"""A2A inbound auth gate: credential hot-reload, diagnostic 401 reasons, and one alert per rejection.

Three requirements are pinned here:

1. **A credential change in the source is honoured without a gateway restart.** Accepted peer
   credentials used to be frozen at adapter construction, so rotating one required a restart
   (docs/a2a-peer-channel-hermes-openclaw.md §7.2). The source (``<HERMES_HOME>/.env``) is now
   re-read through a bounded cache: a change lands within ``A2A_CRED_SOURCE_TTL`` and the file is
   not parsed on every request.
2. **Every rejected authentication carries a diagnostic reason** (``no_credential``,
   ``unknown_credential``, ``malformed_header``, ``method_not_allowed``) in the response body and in
   the audit record.
3. **Exactly one alert record is emitted per rejection**, through a sink list that a future
   webhook/Slack sink can join without touching the auth path. A successful authentication emits none.

Dummy credential values only; no test prints a credential value.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.a2a import protocol, security
from plugins.platforms.a2a.adapter import A2AAdapter

# Dummy credentials. Distinct strings, none a substring of another, so a "value leaked into the
# response/audit" assertion cannot pass by accident.
PEER = "larry"
CRED_OLD = "dummy-old-credential-01"
CRED_NEW = "dummy-new-credential-02"
CRED_NEAR_MISS = "dummy-new-credential-03"
CRED_UNKNOWN = "dummy-wrong-credential-99"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _write_env(tmp_path: Path, **values: str) -> None:
    """Write the operator-editable credential source (the .env file)."""
    (tmp_path / ".env").write_text(
        "".join(f"{name}={value}\n" for name, value in values.items()), encoding="utf-8"
    )


def _audit_records(home: Path) -> list[dict]:
    text = _raw_audit(home)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _raw_audit(home: Path) -> str:
    path = home / "a2a_audit.jsonl"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _alerts(home: Path) -> list[dict]:
    return [r for r in _audit_records(home) if r.get("direction") == "auth_failure"]


def _request(method: str, url: str, body: dict | None = None, headers: dict | None = None):
    """Return (status, raw response text) without raising on 4xx/5xx."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", **(headers or {})}, method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _rpc(text: str = "ping", id_: str = "1") -> dict:
    return {"jsonrpc": "2.0", "id": id_, "method": "message/send",
            "params": {"message": protocol.text_message(protocol.ROLE_USER, text)}}


def _make_adapter(monkeypatch, tmp_path: Path, peers: str | None, ttl: str | None = None) -> tuple[A2AAdapter, str]:
    """A live adapter whose accepted set is ``peers`` (one entry per peer name).

    ``peers=None`` leaves A2A_PEER_TOKENS unset so the credential source is the .env file alone.
    ``ttl`` pins A2A_CRED_SOURCE_TTL; ``None`` keeps the suite hermetic on the default window.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    if peers is not None:
        monkeypatch.setenv("A2A_PEER_TOKENS", peers)
    monkeypatch.setenv("A2A_HOST", "127.0.0.1")
    if ttl is None:
        monkeypatch.delenv("A2A_CRED_SOURCE_TTL", raising=False)
    else:
        monkeypatch.setenv("A2A_CRED_SOURCE_TTL", ttl)
    port = _free_port()
    monkeypatch.setenv("A2A_PORT", str(port))
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": port}))

    async def fake_handle_message(event):
        await adapter.send(event.source.chat_id, "ok", metadata={"notify": True})

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]
    adapter._message_handler = object()  # type: ignore[assignment]  # non-None so dispatch proceeds
    return adapter, f"http://127.0.0.1:{port}"


def _with_server(adapter: A2AAdapter, fn):
    """Run ``fn`` (blocking HTTP calls + assertions) against a connected adapter.

    The event loop must stay alive for the whole exchange — the adapter schedules the
    agent handler on it — so both the connection and the client work are awaited here.
    """
    async def run():
        assert await adapter.connect() is True
        try:
            return await asyncio.to_thread(fn)
        finally:
            await adapter.disconnect()

    return asyncio.run(run())


@pytest.fixture
def clock(monkeypatch):
    """Deterministic monotonic clock plus a cleared credential-source cache."""
    now = [1_000.0]
    monkeypatch.setattr(security, "_monotonic", lambda: now[0])
    security.reset_credential_source_cache()
    yield now
    security.reset_credential_source_cache()


@pytest.fixture(autouse=True)
def _no_registered_sinks():
    """Sinks registered by one test must not observe another test's rejections."""
    reset = getattr(security, "reset_alert_sinks", None)  # absent until the sink list exists
    if reset:
        reset()
    yield
    if reset:
        reset()


# --------------------------------------------------------------------------
# 1. credential rotation without a restart
# --------------------------------------------------------------------------

class TestCredentialHotReload:
    """The context is captured once (what the gateway does at startup); the accepted set must
    follow the source afterwards."""

    def test_rotated_credential_is_honoured_within_the_window_without_restart(
        self, monkeypatch, tmp_path, clock
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")
        monkeypatch.setenv("A2A_CRED_SOURCE_TTL", "30")  # the documented bounded window
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)  # the file is the source, not the env

        _write_env(tmp_path, A2A_PEER_TOKENS=f"{PEER}:{CRED_OLD}")
        security.reset_credential_source_cache()
        ctx = security.A2ASecurityContext.capture()  # == the running process, started once
        assert ctx.authenticate(f"Bearer {CRED_OLD}", "10.0.0.7") == PEER

        # The operator edits the source. No restart, no re-capture.
        _write_env(tmp_path, A2A_PEER_TOKENS=f"{PEER}:{CRED_NEW}")

        # Inside the bounded window the cached set still answers (that is the cache doing its job).
        clock[0] += 5.0
        assert ctx.authenticate(f"Bearer {CRED_OLD}", "10.0.0.7") == PEER
        assert ctx.authenticate(f"Bearer {CRED_NEW}", "10.0.0.7") is None

        # Past the window the new credential is accepted and the retired one is rejected.
        clock[0] += 26.0
        assert ctx.authenticate(f"Bearer {CRED_NEW}", "10.0.0.7") == PEER
        assert ctx.authenticate(f"Bearer {CRED_OLD}", "10.0.0.7") is None

    def test_retired_credential_is_rejected_once_the_window_passes(
        self, monkeypatch, tmp_path, clock
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)

        _write_env(tmp_path, A2A_PEER_TOKENS=f"{PEER}:{CRED_OLD}")
        security.reset_credential_source_cache()
        ctx = security.A2ASecurityContext.capture()

        _write_env(tmp_path, A2A_PEER_TOKENS="")  # credential withdrawn entirely
        clock[0] += 120.0
        assert ctx.authenticate(f"Bearer {CRED_OLD}", "10.0.0.8") is None

    def test_the_source_file_is_not_read_on_every_authentication(
        self, monkeypatch, tmp_path, clock
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("A2A_CRED_SOURCE_TTL", "60")
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        _write_env(tmp_path, A2A_PEER_TOKENS=f"{PEER}:{CRED_OLD}")
        security.reset_credential_source_cache()

        reads: list[str] = []
        real_read = security._read_env_file

        def counting_read(path):
            reads.append(str(path))
            return real_read(path)

        monkeypatch.setattr(security, "_read_env_file", counting_read)
        ctx = security.A2ASecurityContext.capture()
        for _ in range(50):
            assert ctx.authenticate(f"Bearer {CRED_OLD}", "10.0.0.9") == PEER

        assert len(reads) <= 2, f"credential source parsed {len(reads)}x for 50 requests"

    def test_env_provided_credential_is_still_honoured(self, monkeypatch):
        """The .env file is the hot source; a process-env credential (systemd/shell) still works."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.setenv("A2A_PEER_TOKENS", f"{PEER}:{CRED_OLD}")
        ctx = security.A2ASecurityContext.capture()
        assert ctx.authenticate(f"Bearer {CRED_OLD}", "10.0.0.3") == PEER

    def test_rotated_credential_is_accepted_over_http_without_restart(self, monkeypatch, tmp_path):
        """The request thread honours a source rotation: no restart, no adapter rebuild.

        The credential exists only in the .env file, so this also proves the file is the live
        source — the path the operator edits when rotating a peer credential.
        """
        monkeypatch.setenv("A2A_CRED_SOURCE_TTL", "1")
        _write_env(tmp_path, A2A_PEER_TOKENS=f"{PEER}:{CRED_OLD}")
        security.reset_credential_source_cache()
        adapter, base = _make_adapter(monkeypatch, tmp_path, peers=None, ttl="1")  # file is the source

        def probe():
            status, text = _request("POST", base + "/", _rpc(),
                                    {"Authorization": f"Bearer {CRED_OLD}"})
            assert status == 200, text

            # The operator edits the source. No restart, no adapter rebuild.
            _write_env(tmp_path, A2A_PEER_TOKENS=f"{PEER}:{CRED_NEW}")
            time.sleep(1.1)  # let the bounded window (1s) pass

            status, text = _request("POST", base + "/", _rpc(id_="2"),
                                    {"Authorization": f"Bearer {CRED_NEW}"})
            assert status == 200, text
            status, text = _request("POST", base + "/", _rpc(id_="3"),
                                    {"Authorization": f"Bearer {CRED_OLD}"})
            assert status == 401, text
            assert json.loads(text)["error"]["data"]["reason"] == security.AUTH_UNKNOWN_CREDENTIAL
            # The retired credential's rejection alerted; the live one's success did not.
            assert [a["reason"] for a in _alerts(tmp_path)] == [security.AUTH_UNKNOWN_CREDENTIAL]

        _with_server(adapter, probe)


# --------------------------------------------------------------------------
# 2. diagnostic reason codes
# --------------------------------------------------------------------------

class TestDiagnosticReasons:
    def test_reason_codes_are_distinct(self):
        codes = {
            security.AUTH_NO_CREDENTIAL,
            security.AUTH_UNKNOWN_CREDENTIAL,
            security.AUTH_MALFORMED_HEADER,
            security.AUTH_METHOD_NOT_ALLOWED,
        }
        assert len(codes) == 4

    @pytest.fixture(autouse=True)
    def _context(self, monkeypatch):
        monkeypatch.setenv("A2A_PEER_TOKENS", f"{PEER}:{CRED_OLD}")
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_HOST", raising=False)
        self.ctx = security.A2ASecurityContext.capture()

    def test_no_credential(self):
        result = self.ctx.authenticate_detailed(None, "10.0.0.1")
        assert result.identity is None
        assert result.reason == security.AUTH_NO_CREDENTIAL
        assert self.ctx.authenticate_detailed("", "10.0.0.1").reason == security.AUTH_NO_CREDENTIAL
        assert self.ctx.authenticate_detailed("   ", "10.0.0.1").reason == security.AUTH_NO_CREDENTIAL

    def test_malformed_header(self):
        for header in ("Basic dXNlcjpwdw==", "Bearer", "Bearer ", "not-a-scheme-at-all"):
            result = self.ctx.authenticate_detailed(header, "10.0.0.1")
            assert result.identity is None, header
            assert result.reason == security.AUTH_MALFORMED_HEADER, header

    def test_unknown_credential(self):
        result = self.ctx.authenticate_detailed(f"Bearer {CRED_UNKNOWN}", "10.0.0.1")
        assert result.identity is None
        assert result.reason == security.AUTH_UNKNOWN_CREDENTIAL

    def test_near_miss_credential(self):
        """One character off the accepted credential is still an unknown credential."""
        assert CRED_NEAR_MISS[: -1] != CRED_OLD
        result = self.ctx.authenticate_detailed(f"Bearer {CRED_NEAR_MISS[:-1]}", "10.0.0.1")
        assert result.identity is None
        assert result.reason == security.AUTH_UNKNOWN_CREDENTIAL

    def test_wrong_prefix_credential(self):
        result = self.ctx.authenticate_detailed(f"Bearer {CRED_OLD}extra", "10.0.0.1")
        assert result.identity is None
        assert result.reason == security.AUTH_UNKNOWN_CREDENTIAL

    def test_non_ascii_credential_is_a_reason_not_a_crash(self):
        """hmac.compare_digest raises on non-ASCII str; a rejected peer must get a reason, not a crash."""
        result = self.ctx.authenticate_detailed("Bearer klüft-ünïcodé-credential", "10.0.0.1")
        assert result.identity is None
        assert result.reason == security.AUTH_UNKNOWN_CREDENTIAL

    def test_accepted_credential_has_no_reason(self):
        result = self.ctx.authenticate_detailed(f"Bearer {CRED_OLD}", "10.0.0.1")
        assert result.identity == PEER
        assert result.reason == ""

    def test_legacy_authenticate_still_returns_identity(self):
        assert self.ctx.authenticate(f"Bearer {CRED_OLD}", "10.0.0.1") == PEER
        assert self.ctx.authenticate(f"Bearer {CRED_UNKNOWN}", "10.0.0.1") is None


# --------------------------------------------------------------------------
# 3. one alert per rejection, over the real HTTP surface
# --------------------------------------------------------------------------

class TestAuthFailureAlertsOverHttp:
    def test_401_carries_the_reason_and_emits_exactly_one_alert(self, monkeypatch, tmp_path):
        adapter, base = _make_adapter(monkeypatch, tmp_path, f"{PEER}:{CRED_OLD}")

        def probe():
            cases = (
                (None, security.AUTH_NO_CREDENTIAL),
                (f"Bearer {CRED_UNKNOWN}", security.AUTH_UNKNOWN_CREDENTIAL),
                (f"Bearer {CRED_NEAR_MISS}", security.AUTH_UNKNOWN_CREDENTIAL),
                ("Basic dXNlcjpwdw==", security.AUTH_MALFORMED_HEADER),
            )
            for index, (header, expected) in enumerate(cases):
                status, text = _request(
                    "POST", base + "/", _rpc(id_=str(index)),
                    {"Authorization": header} if header else None,
                )
                assert status == 401, text
                body = json.loads(text)
                assert body["error"]["code"] == protocol.ERR_UNAUTHORIZED
                assert body["error"]["data"]["reason"] == expected, text

            alerts = _alerts(tmp_path)
            assert [a["reason"] for a in alerts] == [expected for _, expected in cases]

        _with_server(adapter, probe)

    def test_unsupported_http_method_is_rejected_with_its_own_reason(self, monkeypatch, tmp_path):
        adapter, base = _make_adapter(monkeypatch, tmp_path, f"{PEER}:{CRED_OLD}")

        def probe():
            for method in ("PUT", "DELETE", "PATCH"):
                status, text = _request(
                    method, base + "/", _rpc(), {"Authorization": f"Bearer {CRED_OLD}"}
                )
                assert status == 405, text
                assert json.loads(text)["error"]["data"]["reason"] == security.AUTH_METHOD_NOT_ALLOWED
            assert len(_alerts(tmp_path)) == 3

        _with_server(adapter, probe)

    def test_exactly_one_alert_record_per_rejection(self, monkeypatch, tmp_path):
        """Three rejections, three records — the count is asserted, not the existence."""
        adapter, base = _make_adapter(monkeypatch, tmp_path, f"{PEER}:{CRED_OLD}")

        def probe():
            for index in range(3):
                status, _ = _request("POST", base + "/", _rpc(id_=str(index)))
                assert status == 401
            assert len(_alerts(tmp_path)) == 3

        _with_server(adapter, probe)

    def test_alert_record_carries_reason_and_provenance(self, monkeypatch, tmp_path):
        adapter, base = _make_adapter(monkeypatch, tmp_path, f"{PEER}:{CRED_OLD}")

        def probe():
            status, _ = _request("POST", base + "/", _rpc(), {"Authorization": f"Bearer {CRED_UNKNOWN}"})
            assert status == 401
            records = _alerts(tmp_path)
            assert len(records) == 1
            record = records[0]
            assert record["reason"] == security.AUTH_UNKNOWN_CREDENTIAL
            assert record["event"] == "a2a_auth_failure"
            assert record["http_method"] == "POST"
            assert record["client_ip"]

        _with_server(adapter, probe)

    def test_no_credential_value_ever_reaches_the_response_or_the_audit(self, monkeypatch, tmp_path):
        adapter, base = _make_adapter(monkeypatch, tmp_path, f"{PEER}:{CRED_OLD}")

        def probe():
            for presented in (CRED_UNKNOWN, CRED_NEAR_MISS, CRED_OLD + "x"):
                status, text = _request(
                    "POST", base + "/", _rpc(), {"Authorization": f"Bearer {presented}"}
                )
                assert status == 401, text
                assert presented not in text
                assert CRED_OLD not in text

            raw_audit = _raw_audit(tmp_path)
            for value in (CRED_OLD, CRED_UNKNOWN, CRED_NEAR_MISS):
                assert value not in raw_audit
            for record in _alerts(tmp_path):
                assert CRED_OLD not in json.dumps(record)

        _with_server(adapter, probe)

    def test_successful_authentication_emits_no_alert(self, monkeypatch, tmp_path):
        adapter, base = _make_adapter(monkeypatch, tmp_path, f"{PEER}:{CRED_OLD}")

        def probe():
            status, text = _request(
                "POST", base + "/", _rpc("hello"), {"Authorization": f"Bearer {CRED_OLD}"}
            )
            assert status == 200, text
            assert json.loads(text)["result"]["status"]["state"] == protocol.STATE_COMPLETED
            assert _alerts(tmp_path) == []

        _with_server(adapter, probe)

    def test_public_get_with_a_bad_credential_emits_no_alert(self, monkeypatch, tmp_path):
        """The Agent Card is public: an unauthenticated GET is not a rejected authentication."""
        adapter, base = _make_adapter(monkeypatch, tmp_path, f"{PEER}:{CRED_OLD}")

        def probe():
            status, _ = _request("GET", base + "/.well-known/agent-card.json")
            assert status == 200
            status, _ = _request("GET", base + "/health",
                                 headers={"Authorization": f"Bearer {CRED_UNKNOWN}"})
            assert status == 200
            assert _alerts(tmp_path) == []

        _with_server(adapter, probe)

    def test_registered_sink_receives_exactly_one_call_per_rejection(self, monkeypatch, tmp_path):
        """The sink is pluggable: a webhook/Slack sink can be added without touching the auth path."""
        delivered: list[dict] = []

        def sink(record):
            delivered.append(record)

        security.register_alert_sink(sink)
        adapter, base = _make_adapter(monkeypatch, tmp_path, f"{PEER}:{CRED_OLD}")

        def probe():
            for index in range(2):
                status, _ = _request("POST", base + "/", _rpc(id_=str(index)))
                assert status == 401
            assert len(delivered) == 2
            assert {r["reason"] for r in delivered} == {security.AUTH_NO_CREDENTIAL}
            assert delivered[0]["event"] == "a2a_auth_failure"

        _with_server(adapter, probe)

    def test_a_failing_sink_does_not_break_the_rejection(self, monkeypatch, tmp_path):
        def exploding_sink(record):
            raise RuntimeError("sink down")

        security.register_alert_sink(exploding_sink)
        adapter, base = _make_adapter(monkeypatch, tmp_path, f"{PEER}:{CRED_OLD}")

        def probe():
            status, text = _request("POST", base + "/", _rpc())
            assert status == 401, text
            assert json.loads(text)["error"]["data"]["reason"] == security.AUTH_NO_CREDENTIAL
            assert len(_alerts(tmp_path)) == 1  # the built-in sink still recorded it

        _with_server(adapter, probe)


# --------------------------------------------------------------------------
# 4. one accepted entry per peer name
# --------------------------------------------------------------------------

class TestSinglePeerMapShape:
    def test_credential_map_holds_one_entry_per_peer_name(self, monkeypatch):
        monkeypatch.setenv("A2A_PEER_TOKENS", f"{PEER}:{CRED_OLD}")
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        parsed = dict(security.A2ASecurityContext.capture().peer_tokens)
        assert parsed == {CRED_OLD: PEER}
        assert len(parsed) == 1

    def test_whole_set_works_with_a_single_entry_map(self, monkeypatch, tmp_path, clock):
        """Rotation + rejection reason + alert, all with exactly one accepted credential."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("A2A_CRED_SOURCE_TTL", "1")
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        _write_env(tmp_path, A2A_PEER_TOKENS=f"{PEER}:{CRED_OLD}")
        security.reset_credential_source_cache()
        ctx = security.A2ASecurityContext.capture()
        assert dict(ctx.peer_tokens) == {CRED_OLD: PEER}
        assert len(dict(ctx.peer_tokens)) == 1

        assert ctx.authenticate_detailed(f"Bearer {CRED_OLD}", "10.0.0.1").identity == PEER
        rejected = ctx.authenticate_detailed(f"Bearer {CRED_UNKNOWN}", "10.0.0.1")
        assert rejected.reason == security.AUTH_UNKNOWN_CREDENTIAL
        assert ctx.authenticate_detailed(None, "10.0.0.1").reason == security.AUTH_NO_CREDENTIAL

        security.alert_auth_failure(rejected.reason, client_ip="10.0.0.1", http_method="POST", path="/")
        alerts = _alerts(tmp_path)
        assert len(alerts) == 1
        assert alerts[0]["reason"] == security.AUTH_UNKNOWN_CREDENTIAL
        assert CRED_UNKNOWN not in json.dumps(alerts[0])

        _write_env(tmp_path, A2A_PEER_TOKENS=f"{PEER}:{CRED_NEW}")
        clock[0] += 2.0
        assert ctx.authenticate_detailed(f"Bearer {CRED_NEW}", "10.0.0.1").identity == PEER
        assert ctx.authenticate_detailed(f"Bearer {CRED_OLD}", "10.0.0.1").reason == security.AUTH_UNKNOWN_CREDENTIAL
