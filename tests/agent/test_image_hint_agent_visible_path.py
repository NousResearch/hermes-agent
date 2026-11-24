"""Image attachment hints name the copy the agent's terminal backend can reach.

The gateway retains attachments under its profile cache (``<HERMES_HOME>/cache/documents/native-inputs/
<sha256>/<name>``). An SSH worker runs as another user on another host and only sees the file-synced copy
under ``~/.hermes/cache/…``, so a hint naming the host path sent its ``cp``/``ls``/``read_file`` into
"Permission denied" / "No such file". Hints and text-mode notes are rendered through
``to_agent_visible_cache_path`` (the translation document notes already use); the pixels are still read
from the host path, and readers of a stored hint (vision, replay, retention, the TUI history) map the
agent-visible spelling back to the host file.
"""

import base64
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent.image_routing import build_native_content_parts
from gateway.session_ingress_media import capture_native_media
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg==")
_HINT_RE = re.compile(r"\[Image attached at: (.+)\]")


def _retained(tmp_path, monkeypatch, backend, name="img_x.png"):
    """An image retained in a named profile's store, as every native attach path commits it:
    ``(host_path, sha256)``. The profile home is not ``~/.hermes``, so expanding ``~`` on the
    host can never stand in for the mapping."""
    monkeypatch.setenv("HOME", str(tmp_path / "gateway-user"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "worker"))
    monkeypatch.setenv("TERMINAL_ENV", backend)
    staged = tmp_path / name
    staged.write_bytes(_PNG)
    reference = capture_native_media([staged])[0]
    return Path(reference["path"]), reference["sha256"]


def _visible(sha, name="img_x.png", base="~/.hermes"):
    return f"{base}/cache/documents/native-inputs/{sha}/{name}"


def test_ssh_hint_names_the_synced_copy_and_still_attaches_the_host_bytes(tmp_path, monkeypatch):
    host, sha = _retained(tmp_path, monkeypatch, "ssh")

    parts, skipped = build_native_content_parts("what is in this screenshot?", [str(host)])

    assert skipped == []
    assert parts[0]["text"] == f"what is in this screenshot?\n\n[Image attached at: {_visible(sha)}]"
    url = parts[1]["image_url"]["url"]
    assert base64.b64decode(url.split(",", 1)[1]) == _PNG  # pixels read from the host file


def test_ssh_file_sync_delivers_the_image_where_the_hint_points(tmp_path, monkeypatch):
    """The hint's ``~`` is the remote user's home: the sync list must hold the retained file at exactly
    that remote path, or the hint dangles on the worker."""
    from tools.environments.file_sync import iter_sync_files

    host, _ = _retained(tmp_path, monkeypatch, "ssh")
    parts, _ = build_native_content_parts("look", [str(host)])
    hinted = _HINT_RE.search(parts[0]["text"])[1]

    assert hinted.startswith("~/")
    assert (str(host), "/home/worker" + hinted[1:]) in iter_sync_files("/home/worker/.hermes")


def test_local_backend_keeps_the_host_path(tmp_path, monkeypatch):
    host, _ = _retained(tmp_path, monkeypatch, "local")

    parts, _ = build_native_content_parts("look", [str(host)])

    assert parts[0]["text"] == f"look\n\n[Image attached at: {host}]"


def test_docker_hint_names_the_bind_mount(tmp_path, monkeypatch):
    host, sha = _retained(tmp_path, monkeypatch, "docker")

    parts, _ = build_native_content_parts("look", [str(host)])

    assert parts[0]["text"] == f"look\n\n[Image attached at: {_visible(sha, base='/root/.hermes')}]"


def test_vision_duplicate_guard_recognizes_the_hinted_handle(tmp_path, monkeypatch):
    """``vision_analyze`` on the handle the native turn shows is answered "already in context"."""
    from tools.vision_tools_history_budget import native_turn_duplicate, native_turn_images

    host, _ = _retained(tmp_path, monkeypatch, "ssh")
    parts, _ = build_native_content_parts("look", [str(host)])
    hinted = _HINT_RE.search(parts[0]["text"])[1]

    with native_turn_images(parts):
        duplicate = native_turn_duplicate(hinted, None)

    assert json.loads(duplicate)["already_in_context"] is True


def test_retention_counts_an_agent_visible_hint_as_a_mention(tmp_path, monkeypatch):
    """Retained media is held by any transcript row naming its ``<sha256>/<name>`` tail, so a row whose
    hint uses the agent-visible spelling still keeps the file from being collected."""
    host, sha = _retained(tmp_path, monkeypatch, "ssh")
    parts, _ = build_native_content_parts("look", [str(host)])
    store = storage_handle(tmp_path / "profiles" / "worker")
    seed_session(store, "s", "slack")
    seed_messages(store, "s", {"role": "user", "content": parts[0]["text"]})

    _, mentioned = store.admissions.history_mentions([{"path": str(host), "sha256": sha}])

    assert mentioned == frozenset({str(host)})


def test_delegate_text_mode_hint_names_the_synced_copy(tmp_path, monkeypatch):
    from tools.delegate_tool import _build_child_goal_message

    host, sha = _retained(tmp_path, monkeypatch, "ssh")

    class _Child:
        provider, model = "openrouter", "text-only"

    with patch("agent.image_routing.decide_image_input_mode", return_value="text"):
        msg = _build_child_goal_message("Inspect", [str(host)], _Child())

    assert f"[Image attached at: {_visible(sha)}]" in msg and str(host) not in msg


@pytest.mark.asyncio
async def test_gateway_text_mode_vision_note_names_the_synced_copy(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    host, sha = _retained(tmp_path, monkeypatch, "ssh")
    runner = GatewayRunner.__new__(GatewayRunner)

    with patch("tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock,
               return_value=json.dumps({"success": True, "analysis": "A chart."})) as vision:
        note = await runner._enrich_message_with_vision(user_text="explain", image_paths=[str(host)])

    assert vision.await_args.kwargs["image_url"] == str(host)  # the pre-analysis reads the host file
    assert f"image_url: {_visible(sha)} ~]" in note and str(host) not in note


def test_cli_text_mode_vision_note_names_the_synced_copy(tmp_path, monkeypatch):
    from cli import HermesCLI

    host, sha = _retained(tmp_path, monkeypatch, "ssh")
    cli = HermesCLI.__new__(HermesCLI)

    with patch("tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock,
               return_value=json.dumps({"success": True, "analysis": "A chart."})):
        note = cli._preprocess_images_with_vision("explain", [host], announce=False)

    assert f"image_url: {_visible(sha)}]" in note and str(host) not in note


def test_tui_image_ref_message_names_the_synced_copy(tmp_path, monkeypatch):
    from tui_gateway.server import _build_image_ref_message

    host, sha = _retained(tmp_path, monkeypatch, "ssh")

    note = _build_image_ref_message("explain", [str(host)])

    assert f"image_url: {_visible(sha)}]" in note and str(host) not in note


def test_api_image_hint_names_the_synced_copy(tmp_path, monkeypatch):
    from gateway.session_api_media import commit_api_images, restore_api_images

    _retained(tmp_path, monkeypatch, "ssh")
    url = "data:image/png;base64," + base64.b64encode(_PNG).decode()
    parts = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": url}}]
    media = commit_api_images(parts)

    restored = restore_api_images(parts, media)

    name = Path(media[0]["path"]).name
    assert restored[0]["text"] == f"hi\n\n[Image attached at: {_visible(media[0]['sha256'], name)}]"
    assert restored[1]["image_url"]["url"] == url


def test_tui_history_shows_the_host_file_for_a_synced_hint(tmp_path, monkeypatch):
    from agent.context_references import format_reference_value
    from tui_gateway import server

    host, sha = _retained(tmp_path, monkeypatch, "ssh")
    url = "data:image/png;base64," + base64.b64encode(_PNG).decode()
    row = {"role": "user", "content": [{"type": "text", "text": f"Caption\n\n[Image attached at: {_visible(sha)}]"},
                                       {"type": "image_url", "image_url": {"url": url}}]}

    displayed = server._history_to_messages([row])

    assert displayed == [{"role": "user", "text": f"Caption\n\n@image:{format_reference_value(str(host))}"}]
