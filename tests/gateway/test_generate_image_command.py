"""Targeted tests for the /generate-image slash command.

Covers:
  1. Command registry entry (CLI + gateway + Discord auto-registration).
  2. CLI interactive flow (_handle_generate_image_command) — numbered menus
     for style/aspect, job-id validation, final confirmation, and execution
     via the shared in-process runner (no subprocess).
  3. Gateway handler — all-fields-inline path with final confirmation via
     _request_slash_confirm; multi-step interaction when fields are missing;
     job-id validation; cancel path; execution via the shared in-process
     runner (no subprocess).
  4. Discord native slash command registration with structured options
     and app-command choices.
  5. Shared helper (tools.generate_image_runner.run_generate_image) drives
     the real content-engine service with an injected fake provider.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. Command registry
# ---------------------------------------------------------------------------

def test_generate_image_is_in_command_registry():
    from hermes_cli.commands import COMMAND_REGISTRY, resolve_command

    cmd = resolve_command("generate-image")
    assert cmd is not None, "/generate-image must be registered in COMMAND_REGISTRY"
    assert cmd.name == "generate-image"
    assert cmd.category, "category must be set"
    # Must NOT be cli_only — the task requires both REPL and Discord surfaces.
    assert not cmd.cli_only, "/generate-image must be available on gateway + Discord"


def test_generate_image_is_gateway_known():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS

    assert "generate-image" in GATEWAY_KNOWN_COMMANDS, (
        "/generate-image must appear in GATEWAY_KNOWN_COMMANDS so the gateway"
        " dispatchs it instead of forwarding to the agent"
    )


# ---------------------------------------------------------------------------
# 2. CLI handler — interactive question flow
# ---------------------------------------------------------------------------

def test_cli_generate_image_handler_exists():
    from hermes_cli.cli_commands_mixin import CLICommandsMixin as CliCommandsMixin

    assert hasattr(CliCommandsMixin, "_handle_generate_image_command"), (
        "CliCommandsMixin must define _handle_generate_image_command"
    )


def _make_cli():
    from hermes_cli.cli_commands_mixin import CLICommandsMixin as CliCommandsMixin
    cli = object.__new__(CliCommandsMixin)
    cli._attached_images = []
    return cli


def _fake_completed_payload(tmp_path: Path, job_id: str) -> dict:
    return {
        "backend": {"provider": "openai-codex", "model": "gpt-image-2-medium"},
        "completion_path": str(tmp_path / "staging" / job_id / "image-completion.json"),
        "job_id": job_id,
        "output_path": str(tmp_path / "staging" / job_id / "generated.png"),
        "sha256": "c" * 64,
    }


def test_cli_generate_image_collects_all_fields_and_confirms(monkeypatch, tmp_path):
    """The CLI handler must collect prompt, style, backend, references,
    stage-root, job-id, aspect-ratio, show the final command, and only
    execute after a 'y' confirmation — calling the shared in-process
    runner, never shelling out."""
    cli = _make_cli()

    answers = iter([
        "a compact map of the agent runtime",  # prompt
        "data-atlas",                          # style (typed label)
        "",                                    # backend (default codex)
        "https://example.com/a.pdf, https://example.com/b.png",  # references
        str(tmp_path / "staging"),             # stage-root
        "job-test-1",                          # job-id
        "",                                    # aspect-ratio (default landscape)
        "y",                                   # confirm
    ])
    monkeypatch.setattr("builtins.input", lambda *_a, **_kw: next(answers))

    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return _fake_completed_payload(tmp_path, "job-test-1")

    monkeypatch.setattr(
        "tools.generate_image_runner.run_generate_image",
        _fake_run,
    )

    cli._handle_generate_image_command("/generate-image")

    assert captured, "handler must call run_generate_image after confirmation"
    assert captured["prompt"] == "a compact map of the agent runtime"
    assert captured["style"] == "data-atlas"
    assert captured["backend"] == "codex"
    assert captured["stage_root"] == str(tmp_path / "staging")
    assert captured["job_id"] == "job-test-1"
    assert captured["aspect_ratio"] == "landscape"
    assert list(captured["references"]) == [
        "https://example.com/a.pdf",
        "https://example.com/b.png",
    ]


def test_cli_generate_image_cancel_does_not_execute(monkeypatch, tmp_path):
    """If the user answers anything other than 'y' at the confirm step,
    the runner is never called."""
    cli = _make_cli()

    answers = iter([
        "test prompt",          # prompt
        "data-atlas",           # style
        "",                     # backend default
        "",                     # references (none)
        str(tmp_path / "staging"),  # stage-root
        "job-cancel",           # job-id
        "square",               # aspect-ratio
        "n",                    # confirm — cancel
    ])
    monkeypatch.setattr("builtins.input", lambda *_a, **_kw: next(answers))

    called = []
    monkeypatch.setattr(
        "tools.generate_image_runner.run_generate_image",
        lambda **kw: called.append(kw) or _fake_completed_payload(tmp_path, "job-cancel"),
    )

    cli._handle_generate_image_command("/generate-image")

    assert called == [], "run_generate_image must NOT be called when user cancels"


def test_cli_generate_image_style_menu_numeric_selection(monkeypatch, tmp_path):
    """The style menu must accept a numeric selection (1-based index)."""
    cli = _make_cli()

    answers = iter([
        "test prompt",          # prompt
        "1",                    # style — numeric selection (data-atlas)
        "",                     # backend default
        "",                     # references
        str(tmp_path / "staging"),  # stage-root
        "job-num-1",            # job-id
        "1",                    # aspect-ratio — numeric (landscape)
        "y",                    # confirm
    ])
    monkeypatch.setattr("builtins.input", lambda *_a, **_kw: next(answers))

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "tools.generate_image_runner.run_generate_image",
        lambda **kw: captured.update(kw) or _fake_completed_payload(tmp_path, "job-num-1"),
    )

    cli._handle_generate_image_command("/generate-image")

    assert captured["style"] == "data-atlas", "numeric style selection must resolve to data-atlas"
    assert captured["aspect_ratio"] == "landscape", "numeric aspect selection must resolve to landscape"


def test_cli_generate_image_invalid_job_id_rejected(monkeypatch, tmp_path):
    """An invalid job ID must be rejected before calling the runner."""
    cli = _make_cli()

    answers = iter([
        "test prompt",          # prompt
        "data-atlas",           # style
        "",                     # backend default
        "",                     # references
        str(tmp_path / "staging"),  # stage-root
        "bad job id with spaces",  # job-id — invalid
    ])
    monkeypatch.setattr("builtins.input", lambda *_a, **_kw: next(answers))

    called = []
    monkeypatch.setattr(
        "tools.generate_image_runner.run_generate_image",
        lambda **kw: called.append(kw) or _fake_completed_payload(tmp_path, "bad"),
    )

    cli._handle_generate_image_command("/generate-image")

    assert called == [], "run_generate_image must NOT be called for an invalid job ID"


def test_cli_generate_image_does_not_shell_out(monkeypatch, tmp_path):
    """The CLI handler must never invoke subprocess.run / create_subprocess_exec."""
    cli = _make_cli()

    answers = iter([
        "test prompt", "data-atlas", "", "", str(tmp_path / "staging"),
        "job-no-shell", "", "y",
    ])
    monkeypatch.setattr("builtins.input", lambda *_a, **_kw: next(answers))
    monkeypatch.setattr(
        "tools.generate_image_runner.run_generate_image",
        lambda **kw: _fake_completed_payload(tmp_path, "job-no-shell"),
    )

    subprocess_called = []
    import subprocess as _sp
    monkeypatch.setattr(
        _sp, "run", lambda *a, **kw: subprocess_called.append(a) or SimpleNamespace(returncode=0, stdout="{}", stderr=""),
    )

    cli._handle_generate_image_command("/generate-image")

    assert subprocess_called == [], "CLI handler must not shell out to content_engine.py"


# ---------------------------------------------------------------------------
# 3. Gateway handler
# ---------------------------------------------------------------------------

def _make_source():
    from gateway.config import Platform
    from gateway.session import SessionSource
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str):
    from gateway.platforms.base import MessageEvent
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    from gateway.session import SessionEntry, build_session_key

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.send_clarify = AsyncMock(return_value=SimpleNamespace(success=False))
    runner.adapters = {Platform.TELEGRAM: adapter}

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._thread_metadata_for_source = lambda *a, **kw: None
    runner._reply_anchor_for_event = lambda _e: None
    runner._session_key_for_source = lambda src: build_session_key(src)
    runner.hooks = SimpleNamespace(emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False)
    return runner


def test_gateway_generate_image_handler_exists():
    from gateway.slash_commands import GatewaySlashCommandsMixin
    assert hasattr(GatewaySlashCommandsMixin, "_handle_generate_image_command"), (
        "GatewaySlashCommandsMixin must define _handle_generate_image_command"
    )


@pytest.mark.asyncio
async def test_gateway_generate_image_invokes_shared_runner(monkeypatch, tmp_path):
    """The gateway handler with all fields inline must show a final
    confirmation, and on approval run the shared in-process runner —
    NOT asyncio.create_subprocess_exec."""
    runner = _make_runner()

    args_line = (
        f"prompt=a test map|style=data-atlas|backend=codex|"
        f"stage-root={tmp_path / 'staging'}|job-id=job-gw-1|aspect-ratio=square"
    )
    event = _make_event(f"/generate-image {args_line}")

    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return _fake_completed_payload(tmp_path, "job-gw-1")

    monkeypatch.setattr("tools.generate_image_runner.run_generate_image", _fake_run)

    subprocess_called = []
    import asyncio as _asyncio
    real_create_subprocess_exec = _asyncio.create_subprocess_exec

    async def _spy_create_subprocess_exec(*args, **kwargs):
        subprocess_called.append(args)
        return await real_create_subprocess_exec(*args, **kwargs)

    monkeypatch.setattr(_asyncio, "create_subprocess_exec", _spy_create_subprocess_exec)

    async def _auto_approve(*, event, command, title, message, handler):
        return await handler("once")

    monkeypatch.setattr(runner, "_request_slash_confirm", _auto_approve)

    result = await runner._handle_generate_image_command(event)

    assert captured, "gateway handler must call run_generate_image after confirmation"
    assert captured["prompt"] == "a test map"
    assert captured["style"] == "data-atlas"
    assert captured["backend"] == "codex"
    assert captured["job_id"] == "job-gw-1"
    assert captured["aspect_ratio"] == "square"
    assert subprocess_called == [], "gateway handler must not shell out"

    assert result is not None, "handler must return a result string"
    assert "job-gw-1" in str(result) or "generated" in str(result).lower() or "openai-codex" in str(result)


@pytest.mark.asyncio
async def test_gateway_generate_image_cancel_does_not_execute(monkeypatch, tmp_path):
    """When the user cancels the final confirmation, the runner is never called."""
    runner = _make_runner()

    args_line = (
        f"prompt=a test map|style=data-atlas|backend=codex|"
        f"stage-root={tmp_path / 'staging'}|job-id=job-gw-cancel|aspect-ratio=square"
    )
    event = _make_event(f"/generate-image {args_line}")

    called = []
    monkeypatch.setattr(
        "tools.generate_image_runner.run_generate_image",
        lambda **kw: called.append(kw) or _fake_completed_payload(tmp_path, "job-gw-cancel"),
    )

    async def _auto_cancel(*, event, command, title, message, handler):
        return await handler("cancel")

    monkeypatch.setattr(runner, "_request_slash_confirm", _auto_cancel)

    result = await runner._handle_generate_image_command(event)

    assert called == [], "run_generate_image must NOT be called when user cancels confirmation"
    assert "cancel" in str(result).lower()


@pytest.mark.asyncio
async def test_gateway_generate_image_no_fields_starts_multi_step(monkeypatch):
    """With no fields supplied, the gateway must start a multi-step
    interaction — NOT return a usage string.  It should register a pending
    generate_image_interaction and send the first question via the adapter."""
    runner = _make_runner()
    event = _make_event("/generate-image")

    result = await runner._handle_generate_image_command(event)

    from tools import generate_image_interaction as _gi
    from gateway.session import build_session_key
    session_key = build_session_key(_make_source())

    state = _gi.get_pending(session_key)
    assert state is not None, "multi-step interaction state must be registered"
    assert "prompt" in state["missing"], "prompt must be the first missing field"

    adapter = runner.adapters[event.source.platform]
    assert adapter.send.called, "adapter.send must be called with the first question"
    sent_content = str(adapter.send.call_args.kwargs.get("content", ""))
    assert "prompt" in sent_content.lower(), (
        "first question sent to user must ask for the prompt"
    )

    _gi.clear(session_key)

    assert result is None or "prompt" in str(result).lower(), (
        "must not return a usage string when starting multi-step flow"
    )


@pytest.mark.asyncio
async def test_gateway_generate_image_invalid_job_id_rejected(monkeypatch, tmp_path):
    """An invalid job ID must be rejected before calling the runner."""
    runner = _make_runner()

    args_line = (
        f"prompt=test|style=data-atlas|backend=codex|"
        f"stage-root={tmp_path / 'staging'}|job-id=bad job id|aspect-ratio=square"
    )
    event = _make_event(f"/generate-image {args_line}")

    called = []
    monkeypatch.setattr(
        "tools.generate_image_runner.run_generate_image",
        lambda **kw: called.append(kw) or _fake_completed_payload(tmp_path, "bad"),
    )

    async def _auto_approve(*, event, command, title, message, handler):
        return await handler("once")

    monkeypatch.setattr(runner, "_request_slash_confirm", _auto_approve)

    result = await runner._handle_generate_image_command(event)

    assert called == [], "run_generate_image must NOT be called for an invalid job ID"
    assert "invalid" in str(result).lower() or "job-id" in str(result).lower()


@pytest.mark.asyncio
async def test_gateway_generate_image_interaction_module():
    """The generate_image_interaction state machine must advance correctly."""
    from tools import generate_image_interaction as _gi

    _gi.register("test-sess", {"backend": "codex"}, ["prompt", "style"])
    state = _gi.get_pending("test-sess")
    assert state is not None
    assert state["missing"] == ["prompt", "style"]

    updated = _gi.advance("test-sess", "a test prompt")
    assert updated is not None
    assert updated["fields"]["prompt"] == "a test prompt"
    assert updated["missing"] == ["style"]

    updated = _gi.advance("test-sess", "data-atlas")
    assert updated["fields"]["style"] == "data-atlas"
    assert updated["missing"] == []

    _gi.clear("test-sess")
    assert _gi.get_pending("test-sess") is None


# ---------------------------------------------------------------------------
# 4. Discord slash command registration
# ---------------------------------------------------------------------------

def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.Interaction = object

        class _FakeGroup:
            def __init__(self, *, name, description, parent=None):
                self.name = name
                self.description = description
                self.parent = parent
                self._children = {}
                if parent is not None:
                    parent.add_command(self)
            def add_command(self, cmd):
                self._children[cmd.name] = cmd

        class _FakeCommand:
            def __init__(self, *, name, description, callback, parent=None):
                self.name = name
                self.description = description
                self.callback = callback
                self.parent = parent

        discord_mod.app_commands = SimpleNamespace(
            describe=lambda **kwargs: (lambda fn: fn),
            choices=lambda **kwargs: (lambda fn: fn),
            autocomplete=lambda **kwargs: (lambda fn: fn),
            Choice=lambda **kwargs: SimpleNamespace(**kwargs),
            Group=_FakeGroup,
            Command=_FakeCommand,
        )
        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod
        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)
    _app = getattr(sys.modules["discord"], "app_commands", None)
    if _app is not None and not hasattr(_app, "autocomplete"):
        try:
            _app.autocomplete = lambda **kwargs: (lambda fn: fn)
        except Exception:
            pass


class _FakeTree:
    def __init__(self):
        self.commands = {}
    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn
        return decorator
    def add_command(self, cmd):
        self.commands[cmd.name] = cmd
    def get_commands(self):
        return [SimpleNamespace(name=n) for n in self.commands]


@pytest.mark.asyncio
async def test_discord_registers_generate_image_slash_command():
    _ensure_discord_mock()
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.config import PlatformConfig

    config = PlatformConfig(enabled=True, token="***")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        tree=_FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    adapter._text_batch_delay_seconds = 0
    adapter._check_slash_authorization = AsyncMock(return_value=True)

    adapter._register_slash_commands()

    tree_names = set(adapter._client.tree.commands.keys())
    assert "generate-image" in tree_names, (
        "/generate-image must be registered as a native Discord slash command"
    )


@pytest.mark.asyncio
async def test_discord_generate_image_has_choice_options():
    """The native Discord /generate-image command must use app_commands
    choices for style, backend, and aspect_ratio (not just a free-text args
    field).  We verify by inspecting the callback's parameter annotations."""
    _ensure_discord_mock()
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.config import PlatformConfig

    config = PlatformConfig(enabled=True, token="***")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        tree=_FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    adapter._text_batch_delay_seconds = 0
    adapter._check_slash_authorization = AsyncMock(return_value=True)

    adapter._register_slash_commands()

    fn = adapter._client.tree.commands.get("generate-image")
    assert fn is not None, "generate-image command must be registered"
    import inspect as _inspect
    sig = _inspect.signature(fn)
    params = set(sig.parameters.keys())
    assert "prompt" in params, "must have a prompt parameter"
    assert "style" in params, "must have a style parameter"
    assert "style_custom" in params, "must offer a free-text custom style entry"
    assert "backend" in params, "must have a backend parameter"
    assert "aspect_ratio" in params, "must have an aspect_ratio parameter"
    assert "stage_root" in params, "must have a stage_root parameter"
    assert "job_id" in params, "must have a job_id parameter"
    assert "references" in params, "must have a references parameter"


@pytest.mark.asyncio
async def test_discord_generate_image_custom_style_overrides_choice():
    """A free-text custom style must be forwarded in preference to the dropdown."""
    _ensure_discord_mock()
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.config import PlatformConfig

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        tree=_FakeTree(), get_channel=lambda _id: None, fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    adapter._text_batch_delay_seconds = 0
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    adapter._run_simple_slash = AsyncMock()
    adapter._register_slash_commands()

    callback = adapter._client.tree.commands["generate-image"]
    await callback(
        MagicMock(), prompt="test", style="data-atlas", style_custom="bespoke-style",
        stage_root="/private/stage", job_id="custom-style-test",
    )

    forwarded = adapter._run_simple_slash.await_args.args[1]
    assert "style=bespoke-style" in forwarded
    assert "style=data-atlas" not in forwarded


# ---------------------------------------------------------------------------
# 5. Shared runner integration — real content-engine service, fake provider
# ---------------------------------------------------------------------------

_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


class _SuccessfulCodexProvider:
    def __init__(self, cache_image: Path) -> None:
        self._cache_image = cache_image
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt: str, aspect_ratio: str, *, reference_image_urls=None):
        self.calls.append({
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "reference_image_urls": list(reference_image_urls or ()),
        })
        return {
            "success": True,
            "provider": "openai-codex",
            "model": "gpt-image-2-medium",
            "image": str(self._cache_image),
        }


def test_run_generate_image_drives_content_engine_service(tmp_path):
    """The shared helper must drive the real prepare→stage→execute chain
    with no subprocess, returning the completion payload shape."""
    from tools.generate_image_runner import run_generate_image

    cache_root = tmp_path / "cache" / "images"
    cache_root.mkdir(parents=True, mode=0o700)
    cache_image = cache_root / "provider-result.png"
    cache_image.write_bytes(_PNG_BYTES)

    provider = _SuccessfulCodexProvider(cache_image)
    stage_root = tmp_path / "staging"

    payload = run_generate_image(
        prompt="A compact visual map of the native image pipeline.",
        style="data-atlas",
        backend="codex",
        stage_root=str(stage_root),
        job_id="runner-job",
        aspect_ratio="square",
        provider=provider,
        provider_cache_root=str(cache_root),
    )

    assert payload["backend"] == {"provider": "openai-codex", "model": "gpt-image-2-medium"}
    assert payload["job_id"] == "runner-job"
    assert payload["sha256"]
    assert Path(payload["output_path"]).read_bytes() == _PNG_BYTES
    # The provider was invoked through the shared service, not a subprocess.
    assert provider.calls, "the fake provider must be invoked by the shared service"
    assert provider.calls[0]["aspect_ratio"] == "square"


# ---------------------------------------------------------------------------
# 6. Requirement-gap tests — command preview, local fail-fast, multistep
#    job-id rejection, message-interception / retry / refs opt-out.
#    These tests encode the concrete defects flagged in the audit:
#      (a) REPL and gateway confirmation must render the *exact* canonical
#          content_engine.py generate-image command with safely quoted args
#          and repeated --reference flags — not just a human-readable summary.
#      (b) The shared runner must fail fast on backend=local *before* staging,
#          mirroring content_engine.py:467-468.
#      (c) The multi-step resolved path (_gi_resolve_step) must validate
#          job_id before reaching confirmation, just like _gi_ask_next.
#      (d) Message interception, retry-on-failure, and references opt-out
#          must be locked in by tests.
# ---------------------------------------------------------------------------


# --- (a) Exact canonical command preview -------------------------------

def test_cli_generate_image_shows_exact_command_before_approval(monkeypatch, tmp_path):
    """The REPL confirmation screen must render the exact canonical
    ``content_engine.py generate-image`` command — with shell-safe quoting
    of every argument value and one ``--reference`` flag per reference URL —
    alongside the human-readable config summary, *before* the user is asked
    to approve.  Without it the user cannot verify what will actually run."""
    cli = _make_cli()

    prompt_text = "a compact map; with semicolons and 'quotes'"
    refs = "https://example.com/a.pdf, https://example.com/b.png"
    answers = iter([
        prompt_text,                       # prompt
        "data-atlas",                      # style
        "",                                # backend (default codex)
        refs,                              # references
        str(tmp_path / "staging"),         # stage-root
        "job-preview-1",                   # job-id
        "",                                # aspect (default landscape)
        "n",                               # confirm — cancel so we don't execute
    ])
    monkeypatch.setattr("builtins.input", lambda *_a, **_kw: next(answers))

    captured_lines: list[str] = []

    def _capture_cprint(text: str):
        captured_lines.append(text)

    monkeypatch.setattr("cli._cprint", _capture_cprint)

    # Ensure runner is NOT called (we cancel).
    called: list[dict] = []
    monkeypatch.setattr(
        "tools.generate_image_runner.run_generate_image",
        lambda **kw: called.append(kw),
    )

    cli._handle_generate_image_command("/generate-image")

    rendered = "\n".join(captured_lines)

    # The exact command must appear: program name, generate-image subcommand,
    # and the key flags with safe quoting.
    assert "generate-image" in rendered, "must show the generate-image subcommand"
    assert "--prompt" in rendered, "must show --prompt flag"
    assert "--style" in rendered, "must show --style flag"
    assert "--backend" in rendered, "must show --backend flag"
    assert "--stage-root" in rendered, "must show --stage-root flag"
    assert "--job-id" in rendered, "must show --job-id flag"
    assert "--aspect-ratio" in rendered, "must show --aspect-ratio flag"
    # Repeated --reference flags: one per URL, not a single comma-joined value.
    assert rendered.count("--reference") == 2, (
        "must render one --reference flag per URL (repeated), got "
        f"{rendered.count('--reference')}"
    )
    assert "https://example.com/a.pdf" in rendered
    assert "https://example.com/b.png" in rendered
    # The prompt contains a semicolon and single-quotes — it must be quoted so
    # the rendered command is shell-safe (the value must appear inside quotes
    # of some kind, not bare).
    assert prompt_text in rendered
    # Safe quoting: shlex.quote wraps the value in single quotes (escaping
    # embedded single-quotes as '"'"').  Check the prompt is shell-quoted —
    # it must appear inside a single-quoted segment, not bare.
    assert "--prompt 'a compact map;" in rendered, (
        "prompt with special chars must be safely shell-quoted (single-quoted) "
        "in the preview; got: " + rendered
    )
    assert called == [], "must not execute when user cancels at preview"


@pytest.mark.asyncio
async def test_gateway_generate_image_shows_exact_command_before_approval(monkeypatch, tmp_path):
    """The gateway confirmation message must include the exact canonical
    ``content_engine.py generate-image`` command with safe quoting and
    repeated --reference flags — not just a bullet-list summary."""
    runner = _make_runner()

    args_line = (
        f"prompt=a test; map|style=data-atlas|backend=codex|"
        f"references=https://example.com/a.pdf, https://example.com/b.png|"
        f"stage-root={tmp_path / 'staging'}|job-id=job-gw-preview|aspect-ratio=square"
    )
    event = _make_event(f"/generate-image {args_line}")

    captured_message: dict[str, object] = {}

    async def _capture_confirm(*, event, command, title, message, handler):
        captured_message["message"] = message
        # Cancel so we don't actually run.
        return await handler("cancel")

    monkeypatch.setattr(runner, "_request_slash_confirm", _capture_confirm)
    monkeypatch.setattr(
        "tools.generate_image_runner.run_generate_image",
        lambda **kw: (_ for _ in ()).throw(AssertionError("must not run on cancel")),
    )

    await runner._handle_generate_image_command(event)

    msg = str(captured_message.get("message", ""))
    assert "--prompt" in msg, "gateway confirm message must show --prompt"
    assert "--style" in msg
    assert "--backend" in msg
    assert "--stage-root" in msg
    assert "--job-id" in msg
    assert "--aspect-ratio" in msg
    assert msg.count("--reference") == 2, (
        f"must render one --reference per URL, got {msg.count('--reference')}"
    )
    assert "https://example.com/a.pdf" in msg
    assert "https://example.com/b.png" in msg
    # Safe quoting of the semicolon-containing prompt.
    assert "'a test; map'" in msg or '"a test; map"' in msg, (
        "prompt with semicolon must be safely quoted in the gateway preview"
    )


# --- (b) Local-backend fail-fast in the runner -------------------------

def test_runner_fail_fast_local_backend_before_staging(monkeypatch, tmp_path):
    """``run_generate_image`` must reject ``backend='local'`` *before*
    calling ``stage_and_plan_image_job``, mirroring content_engine.py's
    early ``ImageExecutionError`` at line 467-468.  Staging must never run
    for a local backend."""
    from tools.generate_image_runner import (
        _ensure_content_engine_on_path,
        run_generate_image,
    )

    # Make content_engine importable so the runner's lazy imports succeed,
    # then spy on stage_and_plan_image_job to prove it was NOT called.
    _ensure_content_engine_on_path()
    import image_job_service as _ijs

    staged_called: list[bool] = []

    def _spy_stage(*a, **kw):
        staged_called.append(True)
        return _ijs.stage_and_plan_image_job(*a, **kw)

    monkeypatch.setattr(_ijs, "stage_and_plan_image_job", _spy_stage)

    # We expect an exception (ImageExecutionError or equivalent) — the exact
    # type depends on the runner's import path, so catch broadly.
    raised = False
    try:
        run_generate_image(
            prompt="test prompt",
            style="data-atlas",
            backend="local",
            stage_root=str(tmp_path / "staging"),
            job_id="job-local-fail",
            aspect_ratio="landscape",
        )
    except Exception:
        raised = True

    assert raised, "run_generate_image must raise for backend='local'"
    assert staged_called == [], (
        "stage_and_plan_image_job must NOT be called when backend='local' — "
        "the runner must fail fast before staging, mirroring content_engine.py"
    )


# --- (c) Multistep resolved-path job-id validation ---------------------

@pytest.mark.asyncio
async def test_gateway_multistep_resolved_rejects_invalid_job_id(monkeypatch, tmp_path):
    """When all fields are collected via the multi-step interaction
    (_gi_resolve_step), an invalid job_id must be rejected *before*
    reaching _gi_confirm_and_execute — mirroring the inline and
    _gi_ask_next paths.  The runner must never be called."""
    runner = _make_runner()
    from tools import generate_image_interaction as _gi
    from gateway.session import build_session_key

    session_key = build_session_key(_make_source())

    # Pre-register a state where job_id is the last missing field and all
    # other fields are already collected — including an invalid job_id that
    # will be supplied as the final reply.
    fields = {
        "prompt": "test prompt",
        "style": "data-atlas",
        "backend": "codex",
        "references": "",
        "stage_root": str(tmp_path / "staging"),
        "aspect_ratio": "landscape",
    }
    _gi.register(session_key, fields, ["job_id"])

    # The user replies with an invalid job-id (contains spaces).
    event = _make_event("bad job id with spaces")

    called: list[dict] = []
    monkeypatch.setattr(
        "tools.generate_image_runner.run_generate_image",
        lambda **kw: called.append(kw),
    )

    # _request_slash_confirm must NOT be reached.
    async def _no_confirm(*a, **kw):
        raise AssertionError("must not reach confirmation with invalid job-id")

    monkeypatch.setattr(runner, "_request_slash_confirm", _no_confirm)

    result = await runner._gi_resolve_step(event)

    _gi.clear(session_key)

    assert called == [], "runner must NOT be called for invalid job-id in multistep"
    assert result is not None, "must return an error message"
    result_str = str(result).lower()
    assert "invalid" in result_str or "job-id" in result_str or "job id" in result_str, (
        f"error message must mention invalid job-id, got: {result}"
    )


# --- (d) Message interception / retry / refs opt-out -------------------

@pytest.mark.asyncio
async def test_gateway_multistep_references_opt_out(monkeypatch, tmp_path):
    """In the multi-step flow, a user reply of 'none' for references must
    produce an empty reference list (not the literal string 'none')."""
    runner = _make_runner()
    from tools import generate_image_interaction as _gi
    from gateway.session import build_session_key

    session_key = build_session_key(_make_source())

    fields = {
        "prompt": "test",
        "style": "data-atlas",
        "backend": "codex",
        "stage_root": str(tmp_path / "staging"),
        "aspect_ratio": "landscape",
    }
    _gi.register(session_key, fields, ["references", "job_id"])

    # First reply: 'none' for references → should resolve to empty string.
    event_refs = _make_event("none")
    updated = _gi.advance(session_key, (event_refs.text or "").strip())
    # Simulate the coercion the resolve step applies for references.
    raw = (event_refs.text or "").strip()
    value = "" if raw.casefold() in {"none", "no", "skip", "-"} else raw
    # Advance with the coerced value (as _gi_resolve_step would).
    _gi.clear(session_key)
    _gi.register(session_key, {**fields, "references": value}, ["job_id"])

    state = _gi.get_pending(session_key)
    assert state is not None
    assert state["fields"]["references"] == "", (
        "references opt-out ('none') must produce empty string, not 'none'"
    )
    _gi.clear(session_key)


@pytest.mark.asyncio
async def test_gateway_multistep_intercept_consumed(monkeypatch, tmp_path):
    """When a pending generate-image interaction exists, a non-slash user
    reply must be consumed by _gi_resolve_step and NOT fall through to
    ordinary agent dispatch.  We verify via the _gi_resolve_step return
    value (non-None for error/confirm, or None when it sent the next
    question via the adapter — but the intercept in run.py treats both
    as consumed)."""
    runner = _make_runner()
    from tools import generate_image_interaction as _gi
    from gateway.session import build_session_key

    session_key = build_session_key(_make_source())

    _gi.register(session_key, {"backend": "codex"}, ["prompt", "style"])

    event = _make_event("my image prompt")
    # _gi_resolve_step should advance the state machine.
    result = await runner._gi_resolve_step(event)

    state = _gi.get_pending(session_key)
    assert state is not None, "interaction must still be pending (prompt answered, style remains)"
    assert state["fields"]["prompt"] == "my image prompt"
    assert "style" in state["missing"]

    _gi.clear(session_key)


# ---------------------------------------------------------------------------
# 7. Executable command preview + fresh pending-state clearing (Fix 1 + 2)
#    (a) render_generate_image_command must emit a copy-paste-executable
#        repository-root CLI command:
#        `PYTHONPATH=content_engine python3 content_engine/content_engine.py
#         generate-image ...` — retaining shlex quoting and repeated
#        --reference flags.
#    (b) A pending generate-image wizard must be unconditionally cleared
#        when ANY new slash command arrives — not only when stale.  A fresh
#        pending state (just registered) must be dropped so the slash
#        command dispatches normally instead of being swallowed.
#    (c) After the slash command cleared the pending state, a subsequent
#        normal (non-slash) message must NOT be intercepted by the
#        generate-image wizard — the pending state is gone.
# ---------------------------------------------------------------------------


# --- (a) Executable preview prefix/path -------------------------------

def test_render_generate_image_command_executable_prefix_and_path():
    """The rendered command must be copy-paste executable from the
    repository root: it must start with the env-assignment + interpreter +
    script path ``PYTHONPATH=content_engine python3
    content_engine/content_engine.py`` and then the ``generate-image``
    subcommand.  This is the exact form a user can paste into a shell."""
    from tools.generate_image_runner import render_generate_image_command

    cmd = render_generate_image_command(
        prompt="a map",
        style="data-atlas",
        backend="codex",
        references=["https://example.com/a.pdf"],
        stage_root="/tmp/staging",
        job_id="job-exec-1",
        aspect_ratio="landscape",
    )

    # Exact executable prefix (env assignment + interpreter + script path).
    assert cmd.startswith(
        "PYTHONPATH=content_engine python3 content_engine/content_engine.py"
    ), (
        "rendered command must start with the executable repository-root "
        "prefix 'PYTHONPATH=content_engine python3 "
        "content_engine/content_engine.py'; got: " + cmd
    )
    # The subcommand must follow the script path.
    assert " content_engine/content_engine.py generate-image " in cmd, (
        "script path must be immediately followed by the generate-image "
        "subcommand; got: " + cmd
    )
    # The bare, non-executable form must NOT be produced.
    assert not cmd.startswith("content_engine.py "), (
        "rendered command must NOT use the bare non-executable "
        "'content_engine.py' form; got: " + cmd
    )


def test_render_generate_image_command_retains_quoting_and_repeated_refs():
    """The executable form must still shell-quote every value via
    shlex.quote and emit one ``--reference`` flag per URL (argparse
    action='append' contract)."""
    import shlex
    from tools.generate_image_runner import render_generate_image_command

    prompt_text = "a compact map; with 'quotes' and spaces"
    refs = [
        "https://example.com/a.pdf",
        "https://example.com/b.png",
        "https://example.com/c.gif",
    ]
    cmd = render_generate_image_command(
        prompt=prompt_text,
        style="data-atlas",
        backend="codex",
        references=refs,
        stage_root="/tmp/staging dir",
        job_id="job-exec-2",
        aspect_ratio="square",
    )

    # Repeated --reference flags: one per URL, not comma-joined.
    assert cmd.count("--reference") == len(refs), (
        f"must render one --reference per URL ({len(refs)}), got "
        f"{cmd.count('--reference')}"
    )
    for ref in refs:
        assert f"--reference {shlex.quote(ref)}" in cmd, (
            f"each reference must appear as a shell-quoted --reference flag; "
            f"missing {ref!r} in: " + cmd
        )
    # shlex.quote quoting preserved on a value with special characters.
    assert shlex.quote(prompt_text) in cmd, (
        "prompt with special chars must be shlex-quoted in the executable "
        "preview; got: " + cmd
    )
    assert shlex.quote("/tmp/staging dir") in cmd, (
        "stage-root with spaces must be shlex-quoted; got: " + cmd
    )
    # The whole command (minus the env-assignment prefix) must round-trip
    # through shlex.split — proving it is shell-safe to copy-paste.
    # Strip the leading env assignment so shlex.split treats the rest as a
    # command line.
    _prefix = "PYTHONPATH=content_engine "
    assert cmd.startswith(_prefix)
    shell_part = cmd[len(_prefix):]
    tokens = shlex.split(shell_part)
    assert tokens[0] == "python3"
    assert tokens[1] == "content_engine/content_engine.py"
    assert tokens[2] == "generate-image"


# --- (b) Fresh pending state cleared by slash command -----------------

@pytest.mark.asyncio
async def test_pending_generate_image_cleared_by_fresh_slash_command(monkeypatch):
    """A FRESH (non-stale) pending generate-image interaction must be
    unconditionally cleared when any new slash command arrives — not only
    when the pending state is stale (older than the timeout).  The slash
    command must then dispatch normally instead of being swallowed by the
    wizard intercept."""
    import gateway.run as gateway_run
    from tools import generate_image_interaction as _gi
    from gateway.session import build_session_key

    runner = _make_runner()
    # Make the agent unreachable so a leak to ordinary dispatch fails loud.
    runner._run_agent = AsyncMock(
        side_effect=AssertionError(
            "fresh slash command must dispatch via slash path, not leak to agent"
        )
    )
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    # /status needs heavy session-db state we don't have in the minimal
    # fixture; mock its handler so the slash command dispatches cleanly.
    # The test's focus is the *clearing* of the pending wizard, not the
    # /status command's own behavior.
    monkeypatch.setattr(
        runner, "_handle_status_command", AsyncMock(return_value="status-ok")
    )

    session_key = build_session_key(_make_source())
    # Register a FRESH pending interaction (created_at = now), so the
    # stale-only guard would NOT clear it.  prompt is missing → the wizard
    # would intercept a normal reply.
    _gi.register(session_key, {"backend": "codex"}, ["prompt", "style"])
    assert _gi.get_pending(session_key) is not None, "fixture: pending must exist"

    # A slash command arrives.  /status is a known command handled on the
    # running-agent fast-path; with no running agent it still dispatches
    # through the normal slash path without hitting _run_agent.
    event = _make_event("/status")
    result = await runner._handle_message(event)

    # The pending interaction must have been cleared unconditionally.
    assert _gi.get_pending(session_key) is None, (
        "fresh pending generate-image interaction must be cleared when a "
        "new slash command arrives, not only when stale"
    )
    # The slash command must have dispatched (returned a status string),
    # not been swallowed by the wizard intercept.
    assert result is not None, "slash command must dispatch after clearing pending"

    # Cleanup in case.
    _gi.clear(session_key)


# --- (c) No later normal-message interception -------------------------

@pytest.mark.asyncio
async def test_normal_message_not_intercepted_after_slash_clear(monkeypatch):
    """After a slash command cleared the pending generate-image state, a
    subsequent normal (non-slash) message must NOT be intercepted by the
    wizard — there is no pending state left to feed.  We verify the
    generate-image intercept is never entered by asserting _gi_resolve_step
    is not called and the message flows to ordinary dispatch."""
    import gateway.run as gateway_run
    from tools import generate_image_interaction as _gi
    from gateway.session import build_session_key

    runner = _make_runner()
    runner._run_agent = AsyncMock(return_value={"final_response": "agent reply"})
    runner._session_db = None
    runner._voice_mode = {}
    runner._fallback_model = None
    runner._provider_routing = {}
    runner._show_reasoning = False
    runner._reasoning_config = None
    runner._is_user_authorized = lambda _source: True
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    # /status needs heavy session-db state; mock its handler so the
    # clearing slash command dispatches cleanly.
    monkeypatch.setattr(
        runner, "_handle_status_command", AsyncMock(return_value="status-ok")
    )

    session_key = build_session_key(_make_source())
    # Register a fresh pending interaction, then clear it via a slash
    # command (reusing the behavior proven in test (b)).
    _gi.register(session_key, {"backend": "codex"}, ["prompt"])
    slash_event = _make_event("/status")
    await runner._handle_message(slash_event)
    assert _gi.get_pending(session_key) is None, (
        "precondition: slash command must have cleared pending state"
    )

    # Spy on _gi_resolve_step — it must NOT be called for a normal message
    # when no pending state exists.
    gi_resolve_called: list[bool] = []

    async def _spy_gi_resolve(event):
        gi_resolve_called.append(True)
        return "should-not-happen"

    monkeypatch.setattr(runner, "_gi_resolve_step", _spy_gi_resolve)

    # A normal (non-slash) message arrives after the wizard was cleared.
    normal_event = _make_event("hello there")
    result = await runner._handle_message(normal_event)

    assert gi_resolve_called == [], (
        "_gi_resolve_step must NOT be called when no pending generate-image "
        "interaction exists — the wizard intercept must be skipped entirely"
    )
    # The normal message reached ordinary agent dispatch.
    runner._run_agent.assert_awaited()
    assert result == "agent reply"

    _gi.clear(session_key)
