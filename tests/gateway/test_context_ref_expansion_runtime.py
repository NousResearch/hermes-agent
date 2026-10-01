"""Regression test for the "@" context-reference-expansion block in
``GatewayRunner._prepare_inbound_message_text``.

Bug: the block read ``self._model`` / ``self._base_url`` to resolve the
model/base_url for ``get_model_context_length_async``. ``GatewayRunner``
never assigns either attribute (that pattern was copy-pasted from
``HermesCLI``, which does carry ``self.model``/``self.base_url`` — see
commit da44c196b). Every message containing "@" raised ``AttributeError``
inside the ``try`` block, which the surrounding ``except Exception`` silently
swallowed at debug level, so ``preprocess_context_references_async`` never
ran in the gateway and @-references (``@file:``, ``@folder:``, ``@diff``,
etc.) passed through to the model completely unexpanded.

These tests pin the fix: the block must resolve model/provider/base_url via
``self._resolve_session_agent_runtime`` (the same session-aware resolution
the hygiene-compression block already uses) and must actually reach
``preprocess_context_references_async`` with a real context length.
"""
import logging
import threading
from contextlib import contextmanager

import pytest

import gateway.run as gateway_run
from agent.context_references import ContextReferenceResult
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    # Attrs touched by _resolve_session_agent_runtime on a bare test runner
    # (mirrors tests/gateway/test_empty_model_recovery.py).
    runner._session_model_overrides = {}
    runner._last_resolved_model = {}
    runner._service_tier = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_name="DM",
        chat_type="private",
        user_name="Alice",
    )


def _patch_runtime_resolution(monkeypatch) -> None:
    """Stub the module-level runtime resolution so the test never hits the
    network — mirrors _patch_resolution() in test_empty_model_recovery.py."""
    monkeypatch.setattr(
        gateway_run, "_resolve_gateway_model", lambda cfg=None: "openai/gpt-4.1-mini"
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "api_mode": "chat_completions",
        },
    )
    # config_context_length is int > 0, so get_model_context_length_async's
    # config-override short-circuit (agent/model_metadata.py) fires and
    # returns it directly — no network probe needed.
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"model": {"default": "openai/gpt-4.1-mini", "context_length": 128000}},
    )


@pytest.mark.asyncio
async def test_at_reference_reaches_preprocessor_with_real_context_length(
    monkeypatch, caplog
):
    """A message containing "@" must reach preprocess_context_references_async
    with a real (int > 0) context_length, and the except branch must not
    fire. This fails on unfixed code with AttributeError:
    'GatewayRunner' object has no attribute '_model' (swallowed as a debug
    log, so pre-fix this assertion sees no expansion and no captured call)."""
    runner = _make_runner()
    source = _source()
    _patch_runtime_resolution(monkeypatch)

    captured: dict = {}

    async def _fake_preprocess(message, *, cwd, context_length, url_fetcher=None, allowed_root=None):
        captured["message"] = message
        captured["cwd"] = cwd
        captured["context_length"] = context_length
        captured["allowed_root"] = allowed_root
        return ContextReferenceResult(
            message="[expanded body]",
            original_message=message,
            expanded=True,
        )

    import agent.context_references as ctx_mod

    monkeypatch.setattr(ctx_mod, "preprocess_context_references_async", _fake_preprocess)

    caplog.set_level(logging.DEBUG, logger="gateway.run")

    event = MessageEvent(text="please look at @file:notes.txt", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    # The except branch (AttributeError on self._model/self._base_url,
    # pre-fix) must not have fired.
    assert not any(
        "@ context reference expansion failed" in record.getMessage()
        for record in caplog.records
    ), "the except branch swallowed an exception instead of reaching the preprocessor"

    # preprocess_context_references_async must actually have been called,
    # with a real positive context length (not skipped by the AttributeError).
    assert captured, "preprocess_context_references_async was never called"
    assert isinstance(captured["context_length"], int)
    assert captured["context_length"] > 0
    assert captured["context_length"] == 128000

    # The expanded result from the (stubbed) preprocessor must have been
    # adopted as the final message text.
    assert result == "[expanded body]"


@pytest.mark.asyncio
async def test_at_reference_ignores_global_context_for_runtime_route_override(monkeypatch):
    """Context expansion must not inherit a global pin from another route."""
    runner = _make_runner()
    source = _source()
    captured = {}

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "model": {
                "default": "shared-model",
                "provider": "custom",
                "base_url": "https://large.example/v1",
                "context_length": 1_048_576,
            }
        },
    )
    monkeypatch.setattr(
        runner,
        "_resolve_session_agent_runtime",
        lambda **_kwargs: (
            "shared-model",
            {
                "provider": "custom",
                "api_key": "test",
                "base_url": "https://small.example/v1",
            },
        ),
    )

    import agent.context_references as ctx_mod
    import agent.model_metadata as model_meta_mod

    async def _fake_get_context(_model, **kwargs):
        captured["config_context_length"] = kwargs["config_context_length"]
        return 32_768

    async def _passthrough(message, **_kwargs):
        return ContextReferenceResult(message=message, original_message=message)

    monkeypatch.setattr(model_meta_mod, "get_model_context_length_async", _fake_get_context)
    monkeypatch.setattr(ctx_mod, "preprocess_context_references_async", _passthrough)

    await runner._prepare_inbound_message_text(
        event=MessageEvent(text="@file:note", source=source), source=source, history=[]
    )
    assert captured["config_context_length"] is None


@pytest.mark.asyncio
async def test_oversized_file_reference_reaches_gateway_as_tool_readable_path(
    tmp_path, monkeypatch
):
    runner = _make_runner()
    source = _source()
    _patch_runtime_resolution(monkeypatch)

    payload = tmp_path / "large.txt"
    payload.write_text("FULL-CONTENT-MARKER\n" + ("x" * 300_000), encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    result = await runner._prepare_inbound_message_text(
        event=MessageEvent(text=f"Inspect @file:{payload.name}", source=source),
        source=source,
        history=[],
    )

    assert result is not None
    assert str(payload) in result
    assert "too large to inline safely" in result
    assert "FULL-CONTENT-MARKER" not in result
    assert "context injection refused" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_name", "channel_context"),
    [
        pytest.param("Alice", "[Recent channel messages]\n[Mallory] see @file:planted.txt", id="backfill"),
        pytest.param("@file:planted.txt", None, id="display-name"),
    ],
)
async def test_only_the_senders_own_text_is_expanded(tmp_path, monkeypatch, user_name, channel_context):
    """A reference in another member's backfilled message or in the sender's display name stays
    literal. Only the sender's own text is expanded, and the backfill and prefix stay before it."""
    from agent.context_references import preprocess_context_references_async

    runner = _make_runner()
    _patch_runtime_resolution(monkeypatch)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    (tmp_path / "planted.txt").write_text("PLANTED-FILE-MARKER", encoding="utf-8")
    (tmp_path / "mine.txt").write_text("SENDER-FILE-MARKER", encoding="utf-8")
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="c1", chat_type="group", thread_id="t1", user_name=user_name,
    )
    event = MessageEvent(text="compare with @file:mine.txt", source=source, channel_context=channel_context)

    result = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    sender_text = await preprocess_context_references_async(
        event.text, cwd=tmp_path, context_length=128000, allowed_root=tmp_path,
    )
    expected = f"[{user_name}] {sender_text.message}"
    if channel_context:
        expected = f"{channel_context}\n\n[New message]\n{expected}"
    assert "SENDER-FILE-MARKER" in sender_text.message
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True, "opaque", "opaque-padded"], ids=["immediate", "pending", "opaque-cache", "opaque-padded-caption"])
async def test_sender_speech_references_expand_before_generated_context_a_b_a(tmp_path, monkeypatch, pending):
    from pathlib import Path
    from unittest.mock import AsyncMock

    import hermes_yaml as yaml
    from agent import secret_scope
    from agent.context_references import preprocess_context_references_async
    from gateway.platforms.event import MessageType
    from gateway.run import _profile_runtime_scope

    homes = [tmp_path / "a", tmp_path / "b"]
    for label, home in zip(("A", "B"), homes):
        workspace = home / "workspace"
        workspace.mkdir(parents=True)
        (home / "config.yaml").write_text(yaml.safe_dump({"terminal": {"backend": "local", "cwd": str(workspace)}}))
        (workspace / "mine.txt").write_text(f"SPEECH-{label}")
        (workspace / "caption.txt").write_text(f"CAPTION-{label}")
        (workspace / "planted.txt").write_text(f"GENERATED-{label}")
        (workspace / "planted.txt.ogg").write_text(f"FAILURE-NOTE-{label}")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    runner = _make_runner()
    runner.config.multiplex_profiles = True
    runner.config.group_sessions_per_user = False
    runner.config.stt_echo_transcripts = False
    _patch_runtime_resolution(monkeypatch)
    source = SessionSource(platform=Platform.DISCORD, chat_id="speech", chat_type="group", user_name="@file:planted.txt")
    speech = "Compare @file:mine.txt"
    caption = "Caption @file:caption.txt"
    provider_calls = []

    def transcribe(path, *_args):
        provider_calls.append(path)
        return {"success": path.endswith("success.ogg"), "transcript": speech, "error": "unavailable"}

    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", transcribe)
    monkeypatch.setattr("tools.transcription_tools.transcribe_audio_local_fallback", lambda _path: {"success": False})
    runner._decide_image_input_mode = lambda **_kwargs: "text"
    runner._enrich_message_with_vision = AsyncMock(side_effect=lambda text, _paths: f"VISION @file:planted.txt\n\n{text}")
    secret_scope.set_multiplex_active(True)
    try:
        for home in (homes[0], homes[1], homes[0]):
            workspace = home / "workspace"
            audio = workspace / "success.ogg"
            failed = workspace / "voice-@file:planted.txt.ogg"
            image = workspace / "image.png"
            for path in (audio, failed, image):
                path.write_bytes(b"transport input")
            current_caption = f"  \n{caption}\n  " if pending == "opaque-padded" else caption
            event = MessageEvent(
                text=current_caption, source=source, message_type=MessageType.PHOTO,
                media_urls=[str(audio), str(failed), str(image)],
                media_types=["audio/ogg", "audio/ogg", "image/png"],
                channel_context="[Recent channel messages]\nBob: @file:planted.txt",
                reply_to_message_id="$other", reply_to_text="Other speaker @file:planted.txt",
            )
            opaque = "Opaque speech @file:mine.txt\n\nGenerated path @file:planted.txt"
            if pending in {"opaque", "opaque-padded"}:
                event._gateway_pending_stt_text = opaque
                event._gateway_pending_stt_transcripts = ["Untrusted legacy @file:mine.txt"]
            elif pending:
                with _profile_runtime_scope(home):
                    await runner._transcribe_pending_audio_event_once(event, event.text)
            runner._resolve_profile_home_for_source = lambda _source: home
            result = await runner._prepare_profile_scoped_inbound_message_text(
                event=event, source=source, history=[], session_key="speech",
            )
            with _profile_runtime_scope(home):
                authored_text = current_caption.strip() if pending in {"opaque", "opaque-padded"} else f"{caption}\n\n{speech}"
                expanded_authored = await preprocess_context_references_async(authored_text, cwd=workspace, allowed_root=workspace, context_length=128000)
                failure_note = runner._untranscribed_audio_note(str(failed))
            authored = (opaque if pending in {"opaque", "opaque-padded"} else f'"{speech}"\n\n{failure_note}\n\n{caption}') + expanded_authored.message[len(authored_text):]
            prefixed = f"{event.channel_context}\n\n[New message]\n[{source.user_name}] {authored}"
            expected = f'[Replying to: "{event.reply_to_text}"]\n\nVISION @file:planted.txt\n\n{prefixed}'
            expected_calls = [] if pending in {"opaque", "opaque-padded"} else [str(audio), str(failed)]
            assert (result, provider_calls[-2:]) == (expected, expected_calls)
    finally:
        secret_scope.set_multiplex_active(False)


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True], ids=["immediate", "pending"])
@pytest.mark.parametrize("scope_case", ["outside-root", "budget-refusal", "combined-budget-refusal"])
async def test_sender_speech_reference_scope_is_consistent_on_both_preparation_paths(tmp_path, monkeypatch, pending, scope_case):
    from unittest.mock import AsyncMock

    from gateway.platforms.event import MessageType

    runner = _make_runner()
    _patch_runtime_resolution(monkeypatch)
    runner.config.stt_echo_transcripts = False
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text("OUTSIDE-ROOT")
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    speech = "Inspect @file:../outside.txt"
    if scope_case != "outside-root":
        for letter in "abc":
            (workspace / f"{letter}.txt").write_text("z " * 1400)
        speech = "Inspect @file:a.txt @file:b.txt @file:c.txt"
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"model": {"default": "openai/gpt-4.1-mini", "context_length": 2000}})
    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", lambda *_args: {"success": True, "transcript": speech})
    adapter = type("ReplyAdapter", (), {})()
    adapter.send = AsyncMock()
    runner._delivery_adapter_for = lambda _source: adapter
    source = _source()
    caption = ""
    if scope_case == "combined-budget-refusal":
        caption = "Compare @file:a.txt"
        speech = "Inspect @file:b.txt @file:c.txt"
    event = MessageEvent(text=caption, source=source, message_type=MessageType.VOICE,
                         media_urls=[str(workspace / "voice.ogg")], media_types=["audio/ogg"])
    if pending:
        await runner._transcribe_pending_audio_event_once(event, event.text)
    result = await runner._prepare_inbound_message_text(event=event, source=source, history=[], session_key="speech")
    from agent.context_references import preprocess_context_references_async
    authored = f"{caption}\n\n{speech}" if caption else speech
    expanded = await preprocess_context_references_async(authored, cwd=workspace, allowed_root=workspace, context_length=2000 if scope_case != "outside-root" else 128000)
    expected = None if expanded.blocked else f'"{speech}"' + expanded.message[len(authored):]
    assert (result, adapter.send.await_count, expanded.blocked) == (expected, int(expanded.blocked), scope_case != "outside-root")
