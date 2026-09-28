"""Inbound attachments on a multiplexed gateway live in the ROUTED profile's cache (#101134).

Adapters cache an attachment before the event is routed, so the bytes land under the launch home
while the routed profile's sandbox mounts point at ``profiles/<p>/cache/<kind>`` — a mounted, empty
directory. Real temp homes A (launch) and B (routed), real ``_profile_runtime_scope``, multiplex on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import hermes_yaml as yaml

from agent import secret_scope as ss
from gateway.platforms.base import cache_media_bytes
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import _profile_runtime_scope
from gateway.run_inbound import GatewayInboundMixin
from gateway.run_inbound_media import rehome_inbound_media
from tools.credential_files import from_agent_visible_cache_path, get_cache_directory_mounts


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """Launch home A (HERMES_HOME, local backend) and routed profile B (docker backend)."""
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(a))
    for name, home in (("a", a), ("b", b)):
        (home / "config.yaml").write_text(
            yaml.safe_dump({"terminal": {"backend": "local" if name == "a" else "docker"}}), encoding="utf-8")
    ss.set_multiplex_active(True)
    yield a, b
    ss.set_multiplex_active(False)


def _adapter_cached(home: Path, kind: str, name: str) -> str:
    """What an adapter leaves behind: a file under ``<home>/cache/<kind>`` (host path)."""
    path = home / "cache" / kind / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 x" if kind == "documents" else b"\xff\xd8\xff\xe0 jpeg")
    return str(path)


def test_routed_turn_rehomes_adapter_cached_attachments_a_b_a(two_homes):
    a, b = two_homes
    doc, img = _adapter_cached(a, "documents", "doc_0123456789ab_report.pdf"), _adapter_cached(a, "images", "img_1.jpg")
    event = MessageEvent(text=f"[document 'report.pdf' saved at: {doc}]", message_type=MessageType.DOCUMENT,
                         source=None, media_urls=[doc, img], media_types=["application/pdf", "image/jpeg"])

    with _profile_runtime_scope(b):  # routed turn: B's mounts, B's cache roots
        rehome_inbound_media(event)
        note = GatewayInboundMixin._prepend_inbound_document_notes(event, "")
        assert "/root/.hermes/cache/documents/doc_0123456789ab_report.pdf" in note  # sandbox form
        # The sandbox path names a file B's container actually mounts.
        host = Path(from_agent_visible_cache_path("/root/.hermes/cache/documents/doc_0123456789ab_report.pdf"))
        assert host.is_file() and host == b / "cache" / "documents" / "doc_0123456789ab_report.pdf"
        assert {Path(m["host_path"]) for m in get_cache_directory_mounts()} >= {host.parent, (b / "cache" / "images")}
        assert "/root/.hermes/cache/documents/doc_0123456789ab_report.pdf" in event.text  # baked note repointed
    assert event.media_urls == [str(b / "cache" / "documents" / "doc_0123456789ab_report.pdf"),
                                str(b / "cache" / "images" / "img_1.jpg")]
    assert not Path(doc).exists() and not Path(img).exists()  # moved, not copied: A holds nothing of B's
    assert (b / "cache" / "images" / "img_1.jpg").is_file()

    # A→B→A: a launch-profile turn keeps its own files where they are (no-op, no bleed into B).
    doc_a = _adapter_cached(a, "documents", "doc_aaaaaaaaaaaa_own.pdf")
    own = MessageEvent(text="", message_type=MessageType.DOCUMENT, source=None,
                       media_urls=[doc_a], media_types=["application/pdf"])
    with _profile_runtime_scope(a):
        rehome_inbound_media(own)
    assert own.media_urls == [doc_a] and Path(doc_a).is_file()
    assert not (b / "cache" / "documents" / "doc_aaaaaaaaaaaa_own.pdf").exists()


def test_cache_media_bytes_returns_host_path_and_translates_in_note(two_homes):
    """``media_urls`` holds ONE coordinate system (host paths, like ``cache_image_from_bytes``); the
    sandbox translation happens at render time under the scope that owns the turn."""
    _, b = two_homes
    with _profile_runtime_scope(b):  # docker backend
        cached = cache_media_bytes(b"%PDF-1.4 x", filename="report.pdf", mime_type="application/pdf")
        assert Path(cached.path).is_file() and Path(cached.path).parent == b / "cache" / "documents"
        assert cached.context_note().startswith("[document 'report.pdf' saved at: /root/.hermes/cache/documents/doc_")


@pytest.mark.parametrize("failure", ["missing", "io-error"])
@pytest.mark.parametrize("quoted_failed", [False, True])
def test_failed_transfer_removes_only_the_foreign_attachment(
    two_homes, monkeypatch, failure, quoted_failed,
):
    from gateway.platforms.event import QuotedMediaDependency
    from gateway import run_inbound_media

    launch, routed = two_homes
    failed = _adapter_cached(launch, "images", "failed.jpg")
    owned = _adapter_cached(routed, "images", "owned.jpg")
    shared = _adapter_cached(launch, "images", "shared.jpg")
    if failure == "missing":
        Path(failed).unlink()
    else:
        for operation in ("move", "copy2"):
            original = getattr(run_inbound_media.shutil, operation)

            def transfer(src, dest, *args, _original=original, **kwargs):
                if str(src) == failed:
                    raise OSError("transfer refused")
                return _original(src, dest, *args, **kwargs)

            monkeypatch.setattr(run_inbound_media.shutil, operation, transfer)
    quoted = (QuotedMediaDependency("!room", "$shared", 2, "shared-content"),)
    if quoted_failed:
        quoted += (QuotedMediaDependency("!room", "$failed", 0, "failed-content"),)
    event = MessageEvent(
        text=f"caption [attachment saved at: {failed}]", message_type=MessageType.PHOTO,
        media_urls=[failed, owned, shared], media_types=["image/jpeg"] * 3,
        media_text_inlined=[True, False, True], _quoted_media_dependencies=quoted,
    )
    with _profile_runtime_scope(routed):
        rehome_inbound_media(event)
    shared_copy = str(routed / "cache" / "images" / "shared.jpg")
    expected = MessageEvent(
        text="caption [attachment saved at: [attachment unavailable]]",
        message_type=MessageType.PHOTO, media_urls=[owned, shared_copy],
        media_types=["image/jpeg"] * 2, media_text_inlined=[False, True], timestamp=event.timestamp,
    )
    expected_authored = MessageEvent(
        text=expected.text, message_type=MessageType.PHOTO, media_urls=[owned],
        media_types=["image/jpeg"], media_text_inlined=[False], timestamp=event.timestamp,
    )
    assert (event, event._quoted_media_dependencies, event.authored_media(),
            Path(shared).read_bytes(), Path(shared_copy).read_bytes()) == (
        expected, (QuotedMediaDependency("!room", "$shared", 1, "shared-content"),),
        expected_authored, b"\xff\xd8\xff\xe0 jpeg", b"\xff\xd8\xff\xe0 jpeg",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["priority-interrupt", "busy-interrupt", "pending-monitor", "pending-drain"])
@pytest.mark.parametrize("failure", [None, "missing", "io-error"])
async def test_textless_quoted_media_placeholders_use_the_routed_cache_a_b_a(two_homes, monkeypatch, path, failure):
    import asyncio
    from types import SimpleNamespace

    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import QuotedMediaDependency
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixAdapter
    from gateway import run_inbound_media
    from tools.credential_files import to_agent_visible_cache_path

    launch, routed = two_homes
    original = _adapter_cached(launch, "images", "quoted-parent.jpg")
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="test", extra={
        "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org",
    }))
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
    try:
        runner.adapters[Platform.MATRIX] = adapter
        runner._delivery_adapter_for = lambda _source: adapter
        runner._draining = False
        turns = []
        expected = []

        for home, profile in ((launch, "default"), (routed, "b"), (launch, "default")):
            _adapter_cached(launch, "images", "quoted-parent.jpg")
            failed_copy = home == routed and failure is not None
            received = []
            agent = SimpleNamespace(interrupt=received.append)
            source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="dm", user_id="@sender:example.org")
            event = MessageEvent(
                text="", source=source, message_id="$reply", media_urls=[original], media_types=["image/jpeg"],
                reply_to_message_id="$parent", reply_to_text="quoted image", reply_to_author_id="@other:example.org",
                reply_to_author_name="Other", reply_to_is_own_message=False, reply_to_author_authorized=True,
                _quoted_media_dependencies=(QuotedMediaDependency(source.chat_id, "$parent", 0, "parent-image"),),
            )

            async def dispatch(incoming, key):
                if path == "priority-interrupt":
                    await runner._hm_busy_interrupt(incoming, incoming.source, agent, key)
                    return
                if path == "busy-interrupt":
                    await runner._interrupt_running_agent_for_busy_event(incoming, adapter, agent, key)
                    return
                adapter._pending_messages[key] = incoming
                if path == "pending-monitor":
                    await runner._run_agent_fire_pending_interrupt(
                        adapter, agent, incoming.source, key, asyncio.Event(), [None],
                        log_context="quoted-media", log=lambda: None,
                    )
                    return
                pending_event, text = await runner._run_agent_drain_pending(
                    {"final_response": "previous answer"}, adapter, incoming.source, key,
                )
                assert pending_event is incoming
                received.append(text)

            runner._handle_active_session_busy_message = dispatch
            with monkeypatch.context() as transfer_patch:
                if failed_copy and failure == "missing":
                    Path(original).unlink()
                if failed_copy and failure == "io-error":
                    def refuse_copy(_source, _destination):
                        raise OSError("quoted copy refused")
                    transfer_patch.setattr(run_inbound_media.shutil, "copy2", refuse_copy)
                await runner._make_profile_busy_session_handler(profile)(event, "incoming")
            own_path = home / "cache" / "images" / "quoted-parent.jpg"
            with _profile_runtime_scope(home):
                visible = to_agent_visible_cache_path(str(own_path))
            reply = (event.reply_to_message_id, event.reply_to_text, event.reply_to_author_id,
                     event.reply_to_author_name, event.reply_to_is_own_message, event.reply_to_author_authorized)
            turns.append({"input": received, "paths": event.media_urls,
                          "media_types": event.media_types, "dependencies": event._quoted_media_dependencies,
                          "source_bytes": Path(original).read_bytes() if Path(original).is_file() else None,
                          "routed_bytes": own_path.read_bytes() if own_path.is_file() else None,
                          "reply": reply, "profile": event.source.profile})
            expected.append({"input": ["[attachment unavailable]" if failed_copy else f"[User sent an image: {visible}]"],
                             "paths": [] if failed_copy else [str(own_path)],
                             "media_types": [] if failed_copy else ["image/jpeg"],
                             "dependencies": () if failed_copy else (QuotedMediaDependency(source.chat_id, "$parent", 0, "parent-image"),),
                             "source_bytes": None if failed_copy and failure == "missing" else b"\xff\xd8\xff\xe0 jpeg",
                             "routed_bytes": None if failed_copy else b"\xff\xd8\xff\xe0 jpeg",
                             "reply": ("$parent", "quoted image", "@other:example.org", "Other", False, True),
                             "profile": profile})
        assert turns == expected

    finally:
        runner.session_store.close_all_db_handles()
