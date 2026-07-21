"""Tests for remote MEDIA fetch (issue #466) — gateway/media_fetch.py.

Covers: the remote-path credential/system denylist, the size cap, the
transparent fetch through BasePlatformAdapter.filter_media_delivery_paths,
the undeliverable-notice plumbing, and end-to-end delivery of a fetched
file through TelegramAdapter.send_document (mirrors
tests/gateway/test_telegram_documents.py).
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.media_fetch import (
    MEDIA_FETCH_MAX_BYTES_ENV,
    fetch_remote_media,
    media_fetch_max_bytes,
    remote_path_is_denied,
)
from gateway.platforms.base import (
    BasePlatformAdapter,
    undeliverable_media_notice,
)
from tools.environments.base import FileFetchError


# ---------------------------------------------------------------------------
# Test double for a remote environment
# ---------------------------------------------------------------------------

class _FakeRemoteEnv:
    """Duck-typed environment: serves files from an in-memory dict."""

    def __init__(self, files=None, home="/home/worker", links=None):
        self.files = files or {}
        self.links = links or {}
        self._remote_home = home
        self.fetched = []

    @property
    def remote_home(self):
        return self._remote_home

    def fetch_realpath(self, remote_path):
        return self.links.get(remote_path, remote_path)

    def fetch_file_size(self, remote_path):
        data = self.files.get(remote_path)
        return None if data is None else len(data)

    def fetch_file(self, remote_path, local_dest):
        data = self.files.get(remote_path)
        if data is None:
            raise FileFetchError(f"{remote_path!r} not found")
        self.fetched.append(remote_path)
        with open(local_dest, "wb") as f:
            f.write(data)


@pytest.fixture()
def remote_backend(tmp_path, monkeypatch):
    """Activate a fake ssh backend and point the document cache at tmp_path."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    cache = tmp_path / "doc_cache"
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", cache)

    env = _FakeRemoteEnv(files={"/home/worker/report.pdf": b"%PDF-1.4 remote"})

    def _install(environment=env):
        monkeypatch.setattr(
            "tools.terminal_tool.get_active_env", lambda task_id: environment
        )
        return environment

    return _install


# ---------------------------------------------------------------------------
# Remote-path denylist (security parity with _path_under_denied_prefix)
# ---------------------------------------------------------------------------

class TestRemotePathDenylist:
    @pytest.mark.parametrize("path", [
        "/etc/passwd",
        "/proc/self/environ",
        "/var/log/auth.log",
        "/home/worker/.ssh/id_rsa",
        "/home/worker/.aws/credentials",
        "/home/worker/.hermes/.env",
        "/home/worker/.hermes/auth.json",
        "/home/worker/.hermes/credentials/token",
        "/home/worker/.hermes/mcp-tokens/github.json",
        "/home/worker/.hermes/auth/google_oauth.json",
        "/home/worker/.hermes/cache/bws_cache.json",
        "/home/worker/work/../.ssh/id_rsa",  # traversal is normalized first
    ])
    def test_denied_with_known_home(self, path):
        assert remote_path_is_denied(path, "/home/worker") is True

    @pytest.mark.parametrize("path", [
        "/home/worker/report.pdf",
        "/workspace/build/output.zip",
        "/tmp/chart.png",
        "/home/worker/.hermes/skills/notes.md",  # skills stay deliverable
    ])
    def test_allowed_with_known_home(self, path):
        assert remote_path_is_denied(path, "/home/worker") is False

    def test_root_home_exception(self):
        # /root is a denied system prefix, but on a container whose home IS
        # /root the operator's own artifacts live there (mirrors the local
        # exception in _path_under_denied_prefix).
        assert remote_path_is_denied("/root/report.pdf", "/root") is False
        assert remote_path_is_denied("/root/.ssh/id_rsa", "/root") is True
        assert remote_path_is_denied("/root/report.pdf", "/home/worker") is True

    def test_unknown_home_is_conservative(self):
        assert remote_path_is_denied("/data/.ssh/id_rsa", None) is True
        assert remote_path_is_denied("/srv/app/.hermes/.env", None) is True
        assert remote_path_is_denied("/srv/app/report.pdf", None) is False

    def test_relative_path_denied(self):
        assert remote_path_is_denied("workspace/report.pdf", "/root") is True


# ---------------------------------------------------------------------------
# Size cap
# ---------------------------------------------------------------------------

class TestSizeCap:
    def test_default_is_50mb(self, monkeypatch):
        monkeypatch.delenv(MEDIA_FETCH_MAX_BYTES_ENV, raising=False)
        assert media_fetch_max_bytes() == 50 * 1024 * 1024

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(MEDIA_FETCH_MAX_BYTES_ENV, "1024")
        assert media_fetch_max_bytes() == 1024

    def test_invalid_override_falls_back(self, monkeypatch):
        monkeypatch.setenv(MEDIA_FETCH_MAX_BYTES_ENV, "lots")
        assert media_fetch_max_bytes() == 50 * 1024 * 1024

    def test_oversized_file_rejected_with_clear_reason(
        self, remote_backend, monkeypatch
    ):
        env = remote_backend()
        env.files["/home/worker/huge.bin"] = b"x" * 2048
        monkeypatch.setenv(MEDIA_FETCH_MAX_BYTES_ENV, "1024")

        local_path, reason = fetch_remote_media("/home/worker/huge.bin")
        assert local_path is None
        assert "2.0 KB" in reason and "1.0 KB" in reason
        assert env.fetched == []  # rejected before any bytes moved


# ---------------------------------------------------------------------------
# fetch_remote_media orchestration
# ---------------------------------------------------------------------------

class TestFetchRemoteMedia:
    def test_success_lands_in_document_cache(self, remote_backend, tmp_path):
        env = remote_backend()
        local_path, reason = fetch_remote_media("/home/worker/report.pdf")
        assert reason is None
        assert local_path is not None
        assert local_path.startswith(str(tmp_path / "doc_cache"))
        assert "report.pdf" in local_path
        with open(local_path, "rb") as f:
            assert f.read() == b"%PDF-1.4 remote"
        assert env.fetched == ["/home/worker/report.pdf"]

    def test_missing_file_reports_backend(self, remote_backend):
        remote_backend()
        local_path, reason = fetch_remote_media("/home/worker/nope.pdf")
        assert local_path is None
        assert "ssh" in reason

    def test_denied_remote_path_never_fetches(self, remote_backend):
        env = remote_backend()
        env.files["/home/worker/.ssh/id_rsa"] = b"PRIVATE KEY"
        local_path, reason = fetch_remote_media("/home/worker/.ssh/id_rsa")
        assert local_path is None
        assert "not allowed" in reason
        assert env.fetched == []

    def test_symlink_to_denied_target_rejected(self, remote_backend):
        env = remote_backend()
        env.files["/tmp/innocent.txt"] = b"PRIVATE KEY"
        env.links["/tmp/innocent.txt"] = "/home/worker/.ssh/id_rsa"
        local_path, reason = fetch_remote_media("/tmp/innocent.txt")
        assert local_path is None
        assert "not allowed" in reason
        assert env.fetched == []

    def test_tilde_resolves_against_remote_home(self, remote_backend):
        env = remote_backend()
        local_path, reason = fetch_remote_media("~/report.pdf")
        assert reason is None
        assert env.fetched == ["/home/worker/report.pdf"]

    def test_local_backend_does_not_fetch(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        local_path, reason = fetch_remote_media("/anywhere/file.pdf")
        assert local_path is None
        assert "no remote terminal backend" in reason

    def test_no_active_session(self, remote_backend, monkeypatch):
        remote_backend()
        monkeypatch.setattr("tools.terminal_tool.get_active_env", lambda task_id: None)
        local_path, reason = fetch_remote_media("/home/worker/report.pdf")
        assert local_path is None
        assert "no active ssh terminal session" in reason


# ---------------------------------------------------------------------------
# Transparent MEDIA integration via filter_media_delivery_paths
# ---------------------------------------------------------------------------

class TestFilterMediaDeliveryPathsRemote:
    def test_remote_path_is_fetched_and_substituted(self, remote_backend):
        remote_backend()
        undeliverable = []
        safe = BasePlatformAdapter.filter_media_delivery_paths(
            [("/home/worker/report.pdf", False)], undeliverable
        )
        assert len(safe) == 1
        fetched_path, is_voice = safe[0]
        assert fetched_path != "/home/worker/report.pdf"
        assert "report.pdf" in fetched_path
        assert is_voice is False
        assert undeliverable == []

    def test_unfetchable_path_lands_in_undeliverable(self, remote_backend):
        remote_backend()
        undeliverable = []
        safe = BasePlatformAdapter.filter_media_delivery_paths(
            [("/home/worker/ghost.pdf", False)], undeliverable
        )
        assert safe == []
        assert undeliverable == ["ghost.pdf"]

    def test_local_files_still_pass_untouched(self, remote_backend, tmp_path,
                                              monkeypatch):
        # A path that resolves locally (e.g. host-side cache) must never
        # round-trip through the backend — additive behavior only.
        env = remote_backend()
        local = tmp_path / "doc_cache"
        local.mkdir(parents=True, exist_ok=True)
        host_file = local / "already_local.pdf"
        host_file.write_bytes(b"%PDF local")
        safe = BasePlatformAdapter.filter_media_delivery_paths(
            [(str(host_file), False)]
        )
        assert safe == [(str(host_file.resolve()), False)]
        assert env.fetched == []

    def test_default_call_signature_unchanged(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        assert BasePlatformAdapter.filter_media_delivery_paths([]) == []
        assert BasePlatformAdapter.filter_media_delivery_paths(
            [("/nonexistent/x.pdf", True)]
        ) == []


class TestUndeliverableNotice:
    def test_single_name(self):
        assert undeliverable_media_notice(["report.pdf"]) == (
            "⚠️ Couldn't deliver: report.pdf."
        )

    def test_deduplicates_and_joins(self):
        notice = undeliverable_media_notice(["a.pdf", "b.csv", "a.pdf"])
        assert notice == "⚠️ Couldn't deliver: a.pdf, b.csv."

    def test_empty_names_fall_back(self):
        assert "file attachment" in undeliverable_media_notice([""])


# ---------------------------------------------------------------------------
# End-to-end: fetched file delivered through TelegramAdapter.send_document
# (mirrors tests/gateway/test_telegram_documents.py)
# ---------------------------------------------------------------------------

def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


class TestRemoteFetchTelegramDelivery:
    @pytest.mark.asyncio
    async def test_remote_media_tag_delivers_as_document(self, remote_backend):
        _ensure_telegram_mock()
        from gateway.config import PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter

        remote_backend()
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
        mock_msg = MagicMock()
        mock_msg.message_id = 7
        adapter._bot = AsyncMock()
        adapter._bot.send_document = AsyncMock(return_value=mock_msg)

        media_files, cleaned = adapter.extract_media(
            "Here you go!\nMEDIA:/home/worker/report.pdf"
        )
        media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
        assert len(media_files) == 1

        result = await adapter.send_document(
            chat_id="12345", file_path=media_files[0][0]
        )
        assert result.success is True
        call_kwargs = adapter._bot.send_document.call_args[1]
        # The uuid-prefixed cache name is what lands on disk; the visible
        # filename comes from the file on upload.
        assert call_kwargs["chat_id"] == 12345
