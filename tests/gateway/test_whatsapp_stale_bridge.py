"""Tests for the WhatsApp stale-bridge staleness handshake.

Regression tests for the stale-bridge trap: ``connect()`` reused any
already-running bridge with ``status: connected`` unconditionally, and
``disconnect()`` only kills bridges the adapter spawned itself.  A
long-lived bridge process therefore survived gateway restarts AND
``hermes update``, serving pre-update bridge.js behavior forever (e.g.
no inbound media download → images/voice notes arrive as placeholders).

The fix: bridge.js reports a hash of its own source in ``/health``
(``scriptHash``); the adapter compares it against the bridge.js on disk
and restarts the bridge on mismatch.  Bridges that predate the handshake
report no hash and are treated as stale by definition.

Also covers the npm dependency-refresh stamp: deps are reinstalled when
package.json changes, not only when node_modules is missing.
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform


class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False



@pytest.fixture(autouse=True)
def _pm_node(monkeypatch):
    """Stand-in for PM's Node/npm; the user's PATH copy is never picked up."""
    from plugins.platforms.whatsapp import adapter as whatsapp_adapter
    monkeypatch.setattr(whatsapp_adapter, "find_node_executable", lambda name: f"/pm/{name}")


def _make_adapter(bridge_script: str = "/tmp/test-bridge.js",
                  session_path: Path = Path("/tmp/test-wa-session")):
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = 19876
    adapter._bridge_script = bridge_script
    adapter._session_path = Path(os.path.abspath(os.path.expanduser(str(session_path))))
    adapter._profile_home = adapter._session_path.parent  # mirror __init__
    adapter._foreign_bridge_session = None  # mirror __init__; harness builds via __new__
    adapter._bridge_probe_timed_out = False
    adapter._bridge_holder_unproven = False
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._bridge_token = ""  # mirror __init__; connect() loads it
    adapter._reply_prefix = None
    adapter._send_read_receipts = False
    adapter._dm_policy = adapter._group_policy = "pairing"
    adapter._allow_from = adapter._group_allow_from = set()
    adapter._running = False
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._message_queue = asyncio.Queue()
    adapter._http_session = None
    return adapter


def _mock_health(json_data, raise_on=None, *, knows_token=True, reflects_bearer=False):
    """Mock aiohttp.ClientSession for the challenge-first probe: ``GET /auth-proof`` answers an HMAC over the request's
    challenge with the session's token (``knows_token``) or, for an impostor, with whatever bearer the request disclosed
    (``reflects_bearer``) or nothing; ``GET /health`` returns 200 + *json_data*. ``raise_on`` "headers"/"body" times
    out that phase. The factory exposes ``.requests`` = every ``(url, headers)`` the adapter sent."""
    from plugins.platforms.whatsapp.adapter import _bridge_auth_proof

    requests: list = []

    def _get(url, **kwargs):
        headers = kwargs.get("headers") or {}
        requests.append((url, headers))
        if raise_on == "headers":
            raise TimeoutError
        if url.endswith("/auth-proof"):
            challenge = headers.get("X-Hermes-Bridge-Challenge", "")
            token = _TEST_TOKEN if knows_token else headers.get("Authorization", "").removeprefix("Bearer ")
            payload = {"authProof": _bridge_auth_proof(token, challenge)} if token else {}
        else:
            payload = json_data
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=payload, side_effect=asyncio.TimeoutError if raise_on == "body" else None)
        return _AsyncCM(mock_resp)

    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=_get)
    mock_session.close = AsyncMock()
    factory = MagicMock(return_value=_AsyncCM(mock_session))
    factory.requests = requests
    return factory


_TEST_TOKEN = "test-bridge-token-with-enough-entropy-0123456789"


def _setup_bridge_dir(tmp_path: Path) -> Path:
    """Create a real bridge dir with bridge.js + bridge_auth.js + package.json + creds + this session's token."""
    bridge_dir = tmp_path / "whatsapp-bridge"
    bridge_dir.mkdir()
    (bridge_dir / "bridge.js").write_text("// current bridge code\n", encoding="utf-8")
    (bridge_dir / "bridge_auth.js").write_text("// current auth code\n", encoding="utf-8")
    (bridge_dir / "package.json").write_text('{"name": "bridge"}\n', encoding="utf-8")
    (bridge_dir / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    session_path = tmp_path / "session"
    session_path.mkdir()
    (session_path / "creds.json").write_text("{}", encoding="utf-8")
    (session_path / ".bridge-token").write_text(_TEST_TOKEN, encoding="utf-8")
    return bridge_dir


def _fresh_node_modules(bridge_dir: Path) -> None:
    """Prepare dependencies matching both manifests before exercising bridge ownership."""
    from gateway.platforms.whatsapp_common import record_whatsapp_bridge_dependency_fingerprint

    nm = bridge_dir / "node_modules"
    nm.mkdir()
    assert record_whatsapp_bridge_dependency_fingerprint(bridge_dir)




class TestStaleBridgeHandshake:


    @pytest.mark.asyncio
    async def test_restarts_bridge_when_read_receipt_config_changed(self, tmp_path):
        from plugins.platforms.whatsapp.adapter import _bridge_source_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        adapter._send_read_receipts = True
        disk_hash = _bridge_source_hash(bridge_dir / "bridge.js")
        mock_client = _mock_health(
            {
                "status": "connected",
                "scriptHash": disk_hash,
                "sendReadReceipts": False,
                "session": str(tmp_path / "session"),
            }
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", mock_client), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("plugins.platforms.whatsapp.adapter.spawn_owned", return_value=mock_proc) as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            await adapter.connect()

        mock_popen.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("holder,status,fatal", [
        ("own", "connected", None),
        ("other", "connected", "whatsapp_bridge_foreign_session"),
        ("other", "disconnected", "whatsapp_bridge_foreign_session"),
        ("headers-timeout", None, "whatsapp_bridge_unresponsive"),
        ("body-timeout", None, "whatsapp_bridge_unresponsive"),
        ("body-timeout-windows", None, "whatsapp_bridge_unresponsive"),
    ], ids=["own-session-adopted", "other-session-connected", "other-session-starting",
            "health-headers-timeout", "health-body-timeout", "health-body-timeout-windows"])
    async def test_bridge_is_adopted_or_left_running_by_its_session(self, tmp_path, holder, status, fatal):
        """Two profiles default to one bridge_port; only this profile's own session may be adopted. Another profile's
        bridge is never killed, even while it reports ``disconnected`` (startup, reconnect, QR wait), and neither is a
        port holder that gives no /health answer, which proves no ownership."""
        from plugins.platforms.whatsapp.adapter import (
            _BRIDGE_FINGERPRINT_FILE, _bridge_config_fingerprint, _bridge_source_hash,
        )

        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        if holder == "own":
            # An own-session bridge is adopted only when its recorded spawn-config
            # fingerprint matches the env this adapter would spawn with now (#126824).
            (adapter._session_path / _BRIDGE_FINGERPRINT_FILE).write_text(
                _bridge_config_fingerprint(adapter._bridge_env(), adapter._bridge_package_hash()), encoding="utf-8"
            )
        reported = tmp_path / ("session" if holder == "own" else "other-profile-session")
        mock_client = _mock_health(
            {
                "status": status,
                "scriptHash": _bridge_source_hash(bridge_dir / "bridge.js"),
                "sendReadReceipts": False,
                "session": str(reported),
            },
            raise_on=holder.split("-")[0] if "timeout" in holder else None,
        )
        windows = holder.endswith("-windows")  # the listener scan must be the platform's own, or Windows still kills

        with (
            patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True),
            patch("aiohttp.ClientSession", mock_client),
            patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock),
            patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile") as mock_kill_pidfile,
            patch("plugins.platforms.whatsapp.adapter._kill_port_process") as mock_kill_port,
            patch("plugins.platforms.whatsapp.adapter._IS_WINDOWS", windows),
            patch("plugins.platforms.whatsapp.adapter._listener_pids_on_port", return_value=[] if windows else [4242]),
            patch("plugins.platforms.whatsapp.adapter._windows_listener_pids", return_value=[4242] if windows else []),
            patch("plugins.platforms.whatsapp.adapter.spawn_owned", return_value=MagicMock()) as mock_popen,
            patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True),
        ):
            assert await adapter.connect() is (fatal is None)

        mock_popen.assert_not_called()
        mock_kill_port.assert_not_called()
        # Reaping by this profile's own pidfile is identity-based; only a foreign verdict skips it.
        assert mock_kill_pidfile.called is (fatal == "whatsapp_bridge_unresponsive")
        if fatal:
            assert adapter._fatal_error_code == fatal
            assert adapter._fatal_error_retryable is (fatal == "whatsapp_bridge_unresponsive")
            assert f"nothing on port {adapter._bridge_port} was" in adapter._fatal_error_message
            assert "bridge_port" in adapter._fatal_error_message
        if fatal == "whatsapp_bridge_foreign_session":
            assert str(reported) in adapter._fatal_error_message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("impostor", ["silent", "reflects-bearer"])
    async def test_listener_without_the_session_token_is_never_adopted_and_never_learns_it(self, tmp_path, impostor):
        """#83395: a local process that bound the port first and mirrors /health (scriptHash is derivable from the
        shipped source) must neither be adopted nor be handed the bearer. The adapter's first request carries only a
        challenge; a listener that cannot HMAC it with the session token is replaced, and every request it ever saw
        was credential-free."""
        from plugins.platforms.whatsapp.adapter import _bridge_source_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(bridge_script=str(bridge_dir / "bridge.js"), session_path=tmp_path / "session")
        mock_client = _mock_health(
            {"status": "connected", "scriptHash": _bridge_source_hash(bridge_dir / "bridge.js"),
             "sendReadReceipts": False, "session": str(tmp_path / "session")},
            knows_token=False, reflects_bearer=(impostor == "reflects-bearer"),
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", mock_client), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("plugins.platforms.whatsapp.adapter.spawn_owned", return_value=mock_proc) as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            assert await adapter.connect() is False  # not adopted; the replacement bridge "died" (mock) → False

        mock_popen.assert_called_once()
        # The fresh bridge gets a fresh token (persisted for cron), so whatever the impostor saw is already dead.
        spawned_token = mock_popen.call_args.kwargs["env"]["HERMES_WHATSAPP_BRIDGE_TOKEN"]
        assert spawned_token != _TEST_TOKEN and len(spawned_token) >= 32
        assert (tmp_path / "session" / ".bridge-token").read_text(encoding="utf-8") == spawned_token
        assert all("Authorization" not in headers for _url, headers in mock_client.requests), mock_client.requests
        assert [url.rsplit("/", 1)[1] for url, _ in mock_client.requests] == ["auth-proof"] * len(mock_client.requests)

    @pytest.mark.asyncio
    async def test_other_profiles_bridge_that_cannot_prove_our_token_is_left_running(self, tmp_path):
        """Each profile's bridge holds its own token, so another profile's live bridge on the shared default port fails
        this session's proof. It must be reported, never killed by port and never replaced."""
        from plugins.platforms.whatsapp.adapter import _bridge_source_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(bridge_script=str(bridge_dir / "bridge.js"), session_path=tmp_path / "session")
        mock_client = _mock_health(
            {"status": "connected", "scriptHash": _bridge_source_hash(bridge_dir / "bridge.js"),
             "sendReadReceipts": False, "session": str(tmp_path / "other-profile-session")},
            knows_token=False,
        )

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", mock_client), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile") as mock_kill_pidfile, \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process") as mock_kill_port, \
             patch("plugins.platforms.whatsapp.adapter._port_listener_pids", return_value=[4242]), \
             patch("plugins.platforms.whatsapp.adapter.spawn_owned", return_value=MagicMock()) as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            assert await adapter.connect() is False

        mock_kill_port.assert_not_called()
        mock_popen.assert_not_called()
        mock_kill_pidfile.assert_called_once()  # our own stale bridge may still be reaped, by identity only
        assert adapter._fatal_error_code == "whatsapp_bridge_foreign_session"
        assert adapter._fatal_error_retryable is False
        assert f"nothing on port {adapter._bridge_port} was stopped" in adapter._fatal_error_message
        assert all("Authorization" not in headers for _url, headers in mock_client.requests)
        assert (tmp_path / "session" / ".bridge-token").read_text(encoding="utf-8") == _TEST_TOKEN  # no spawn, no rotation

    @pytest.mark.asyncio
    async def test_adopted_bridge_is_driven_with_the_bearer(self, tmp_path):
        """A healthy own-session bridge that proves the token is adopted and every later request to it is authenticated."""
        from plugins.platforms.whatsapp.adapter import (
            _BRIDGE_FINGERPRINT_FILE, _bridge_config_fingerprint, _bridge_source_hash,
        )

        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(bridge_script=str(bridge_dir / "bridge.js"), session_path=tmp_path / "session")
        (adapter._session_path / _BRIDGE_FINGERPRINT_FILE).write_text(  # adopted only on a matching spawn config
            _bridge_config_fingerprint(adapter._bridge_env(), adapter._bridge_package_hash()), encoding="utf-8")
        mock_client = _mock_health(
            {"status": "connected", "scriptHash": _bridge_source_hash(bridge_dir / "bridge.js"),
             "sendReadReceipts": False, "session": str(tmp_path / "session")})

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", mock_client), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.create_task"), \
             patch("plugins.platforms.whatsapp.adapter.spawn_owned") as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            assert await adapter.connect() is True

        mock_popen.assert_not_called()
        probe, health = mock_client.requests
        assert probe[0].endswith("/auth-proof") and "Authorization" not in probe[1]
        assert health[0].endswith("/health") and health[1]["Authorization"] == f"Bearer {_TEST_TOKEN}"
        # The long-lived session (send/edit/poll) carries the bearer as a default header.
        assert mock_client.call_args.kwargs["headers"] == {"Authorization": f"Bearer {_TEST_TOKEN}"}


class TestDepRefreshStamp:
    @pytest.mark.asyncio
    async def test_skips_install_when_stamp_fresh(self, tmp_path):
        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", _mock_health({"status": "disconnected"})), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("subprocess.run") as mock_run, \
             patch("plugins.platforms.whatsapp.adapter.spawn_owned", return_value=mock_proc), \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            await adapter.connect()

        mock_run.assert_not_called()


class TestCacheDirEnvPassthrough:
    @pytest.mark.asyncio
    async def test_bridge_spawn_env_has_cache_dirs(self, tmp_path):
        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        adapter._send_read_receipts = True
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", _mock_health({"status": "disconnected"})), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("plugins.platforms.whatsapp.adapter.spawn_owned", return_value=mock_proc) as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            await adapter.connect()

        env = mock_popen.call_args.kwargs["env"]
        from gateway.platforms.base import (
            get_audio_cache_dir,
            get_document_cache_dir,
            get_image_cache_dir,
        )
        assert env["HERMES_IMAGE_CACHE_DIR"] == str(get_image_cache_dir())
        assert env["HERMES_AUDIO_CACHE_DIR"] == str(get_audio_cache_dir())
        assert env["HERMES_DOCUMENT_CACHE_DIR"] == str(get_document_cache_dir())
        assert env["WHATSAPP_SEND_READ_RECEIPTS"] == "true"


class TestConfigFingerprintAdoption:
    """Adoption must also verify the spawn-time config: bridge.js reads its policy/allowlist env
    once at startup, so adopting a bridge spawned with older env keeps gating DMs with the stale
    allowlist until the bridge is killed by hand (#126824)."""

    def _record_fingerprint(self, adapter, bridge_env: dict) -> None:
        from plugins.platforms.whatsapp.adapter import (
            _BRIDGE_FINGERPRINT_FILE, _bridge_config_fingerprint,
        )
        (adapter._session_path / _BRIDGE_FINGERPRINT_FILE).write_text(
            _bridge_config_fingerprint(bridge_env, adapter._bridge_package_hash()), encoding="utf-8"
        )

    async def _reuse(self, adapter, bridge_path):
        adapter._bridge_token = _TEST_TOKEN  # connect() loads it from the session dir before probing
        with patch.object(adapter, "_mark_connected"), \
             patch.object(adapter, "_attach_to_bridge"), \
             patch.object(adapter, "_wire_plugin_handlers"):
            return await adapter._reuse_running_bridge(bridge_path)

    @pytest.mark.asyncio
    async def test_adopts_bridge_when_fingerprint_matches(self, tmp_path, monkeypatch, capsys):
        for key in ("WHATSAPP_DM_POLICY", "WHATSAPP_ALLOWED_USERS", "WHATSAPP_MODE", "WHATSAPP_REPLY_PREFIX"):
            monkeypatch.delenv(key, raising=False)
        from plugins.platforms.whatsapp.adapter import _bridge_source_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        self._record_fingerprint(adapter, adapter._bridge_env())

        with patch("aiohttp.ClientSession", _mock_health(
            {"status": "connected", "scriptHash": _bridge_source_hash(bridge_dir / "bridge.js"),
             "sendReadReceipts": False, "uptime": 230000.5})):
            assert await self._reuse(adapter, bridge_dir / "bridge.js") is True

        assert "uptime: 2d 15h" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_restarts_bridge_when_allowlist_changed(self, tmp_path, monkeypatch, capsys):
        for key in ("WHATSAPP_DM_POLICY", "WHATSAPP_ALLOWED_USERS", "WHATSAPP_MODE", "WHATSAPP_REPLY_PREFIX"):
            monkeypatch.delenv(key, raising=False)
        from plugins.platforms.whatsapp.adapter import _bridge_source_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        stale_env = dict(adapter._bridge_env())
        stale_env["WHATSAPP_DM_POLICY"] = "allowlist"
        stale_env["WHATSAPP_ALLOWED_USERS"] = "1111111111@s.whatsapp.net"
        self._record_fingerprint(adapter, stale_env)

        with patch("aiohttp.ClientSession", _mock_health(
            {"status": "connected", "scriptHash": _bridge_source_hash(bridge_dir / "bridge.js"),
             "sendReadReceipts": False})):
            assert await self._reuse(adapter, bridge_dir / "bridge.js") is False

        assert "Running bridge is stale (config changed), restarting" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_restarts_bridge_when_package_json_changed(self, tmp_path, capsys):
        """_ensure_bridge_deps() reinstalls node_modules on package.json drift; the running bridge
        still holds the old modules, so it must not be adopted either."""
        from plugins.platforms.whatsapp.adapter import _bridge_source_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        self._record_fingerprint(adapter, adapter._bridge_env())
        (bridge_dir / "package.json").write_text('{"name": "bridge", "version": "2"}\n', encoding="utf-8")

        with patch("aiohttp.ClientSession", _mock_health(
            {"status": "connected", "scriptHash": _bridge_source_hash(bridge_dir / "bridge.js"),
             "sendReadReceipts": False})):
            assert await self._reuse(adapter, bridge_dir / "bridge.js") is False

        assert "Running bridge is stale (config changed), restarting" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_restarts_bridge_when_reply_prefix_goes_from_unset_to_empty(self, tmp_path, monkeypatch, capsys):
        """bridge.js sends its default prefix when WHATSAPP_REPLY_PREFIX is unset and none when it is "",
        so a bridge spawned without the key must not be adopted once config sets ``reply_prefix: ""``."""
        monkeypatch.delenv("WHATSAPP_REPLY_PREFIX", raising=False)
        from plugins.platforms.whatsapp.adapter import _bridge_source_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        self._record_fingerprint(adapter, adapter._bridge_env())  # spawned with reply_prefix unset
        adapter._reply_prefix = ""

        with patch("aiohttp.ClientSession", _mock_health(
            {"status": "connected", "scriptHash": _bridge_source_hash(bridge_dir / "bridge.js"),
             "sendReadReceipts": False})):
            assert await self._reuse(adapter, bridge_dir / "bridge.js") is False

        assert "Running bridge is stale (config changed), restarting" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_missing_fingerprint_file_is_stale(self, tmp_path):
        """A bridge from before the fingerprint existed (or a lost file) is never adopted."""
        from plugins.platforms.whatsapp.adapter import _bridge_source_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )

        with patch("aiohttp.ClientSession", _mock_health(
            {"status": "connected", "scriptHash": _bridge_source_hash(bridge_dir / "bridge.js"),
             "sendReadReceipts": False})):
            assert await self._reuse(adapter, bridge_dir / "bridge.js") is False

    @pytest.mark.asyncio
    async def test_spawn_writes_fingerprint_of_spawn_env(self, tmp_path):
        from plugins.platforms.whatsapp.adapter import (
            _BRIDGE_FINGERPRINT_FILE, _bridge_config_fingerprint,
        )

        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", _mock_health({"status": "disconnected"})), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("plugins.platforms.whatsapp.adapter.spawn_owned", return_value=mock_proc) as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            await adapter.connect()

        recorded = (tmp_path / "session" / _BRIDGE_FINGERPRINT_FILE).read_text(encoding="utf-8").strip()
        assert recorded == _bridge_config_fingerprint(mock_popen.call_args.kwargs["env"], adapter._bridge_package_hash())

    @pytest.mark.asyncio
    async def test_disconnect_keeps_fingerprint_file_for_adopted_bridge(self, tmp_path):
        """The adopted bridge keeps running with its recorded config; the next gateway start
        still needs the record to decide adoption, so disconnect() must not remove it."""
        from plugins.platforms.whatsapp.adapter import _BRIDGE_FINGERPRINT_FILE

        bridge_dir = _setup_bridge_dir(tmp_path)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        self._record_fingerprint(adapter, adapter._bridge_env())
        (tmp_path / "session" / "bridge.pid").write_text("4242\n12345", encoding="utf-8")
        adapter._bridge_process = None  # adopted: not managed by us
        adapter._poll_task = None

        with patch.object(adapter, "_release_platform_lock", create=True), \
             patch.object(adapter, "_mark_disconnected", create=True):
            await adapter.disconnect()

        assert not (tmp_path / "session" / "bridge.pid").exists()
        assert (tmp_path / "session" / _BRIDGE_FINGERPRINT_FILE).exists()

    # Named from bridge.js's startup reads, not from _BRIDGE_CONSUMED_ENV, so dropping a key from that tuple fails here.
    @pytest.mark.parametrize("key", [
        "WHATSAPP_MODE", "WHATSAPP_DM_POLICY", "WHATSAPP_GROUP_POLICY", "WHATSAPP_ALLOWED_USERS",
        "WHATSAPP_GROUP_ALLOWED_USERS", "WHATSAPP_REPLY_PREFIX", "WHATSAPP_SEND_READ_RECEIPTS", "WHATSAPP_DEBUG",
        "WHATSAPP_FORWARD_OWNER_MESSAGES", "WHATSAPP_MAX_MESSAGE_LENGTH", "WHATSAPP_CHUNK_DELAY_MS",
        "WHATSAPP_SEND_TIMEOUT_MS", "HERMES_IMAGE_CACHE_DIR", "HERMES_AUDIO_CACHE_DIR", "HERMES_DOCUMENT_CACHE_DIR",
    ])
    def test_every_key_bridge_js_reads_at_startup_is_fingerprinted(self, key):
        from plugins.platforms.whatsapp.adapter import _bridge_config_fingerprint

        spawned = _bridge_config_fingerprint({key: "spawned"})
        assert spawned != _bridge_config_fingerprint({key: "edited"})
        assert spawned != _bridge_config_fingerprint({})


class TestFmtUptime:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (None, ""),
            ("not-a-number", ""),
            (0, "0s"),
            (59.9, "59s"),
            (60, "1m"),
            (3661, "1h 1m"),
            (230000.5, "2d 15h"),
        ],
    )
    def test_compact_rendering(self, seconds, expected):
        from plugins.platforms.whatsapp.adapter import _fmt_uptime

        assert _fmt_uptime(seconds) == expected
