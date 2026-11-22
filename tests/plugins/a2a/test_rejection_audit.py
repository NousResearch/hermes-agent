"""A2A rejection audit and bounded JSONL storage regressions."""

import asyncio
import json
import urllib.error

import pytest

from plugins.platforms.a2a import protocol, security
from tests.plugins.test_a2a_plugin import _make_live_adapter, _post_json, _send_body


class TestAudit:

    def test_audit_rejected_record(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(security, "_rejection_events", [])
        security.audit(
            "inbound", None, None, "unauthorized",
            decision="rejected_bad_token", status=401, ip="127.0.0.1",
        )
        audit_file = tmp_path / "a2a_audit.jsonl"
        assert audit_file.exists()
        rec = json.loads(audit_file.read_text().strip().splitlines()[-1])
        assert rec["direction"] == "inbound"
        assert rec["peer"] is None
        assert rec["decision"] == "rejected_bad_token"
        assert rec["status"] == 401
        assert rec["ip"] == "127.0.0.1"

    def test_audit_hashes_oversized_and_non_scalar_task_ids(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        security.audit("inbound", "peer-y", "x" * 400, "ok")
        rec = json.loads((tmp_path / "a2a_audit.jsonl").read_text().splitlines()[-1])
        assert rec["task_id"].startswith("sha256:")
        assert len(rec["task_id"]) < 40

        security.audit("inbound", "mallory", {"id": "x" * 5000}, "untrusted")
        rec = json.loads((tmp_path / "a2a_audit.jsonl").read_text().splitlines()[-1])
        assert rec["task_id"].startswith("sha256:")

    def test_audit_rejection_quota_caps_per_source(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(security, "_rejection_events", [])
        for _ in range(security._REJECTION_PER_SOURCE + 5):
            security.audit(
                "inbound", None, None, "unauthorized",
                decision="rejected_bad_token", status=401, ip="203.0.113.9",
            )
        lines = (tmp_path / "a2a_audit.jsonl").read_text().splitlines()
        assert len(lines) == security._REJECTION_PER_SOURCE

    def test_audit_rejection_quota_caps_distinct_sources(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(security, "_rejection_events", [])
        for source in range(security._REJECTION_GLOBAL + 1):
            security.audit(
                "inbound", None, None, "unauthorized",
                decision="rejected_bad_token", status=401, ip=f"192.0.2.{source + 1}",
            )
        lines = (tmp_path / "a2a_audit.jsonl").read_text().splitlines()
        assert len(lines) == security._REJECTION_GLOBAL

    def test_audit_rotates_full_file_before_appending(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        path = tmp_path / "a2a_audit.jsonl"
        previous = b"x" * security._MAX_AUDIT_FILE_BYTES
        path.write_bytes(previous)
        security.audit("inbound", "peer-y", "task-2", "hello world")
        assert path.with_suffix(".jsonl.1").read_bytes() == previous
        assert json.loads(path.read_text())["task_id"] == "task-2"


class TestRejectionAuditHttp:
    def test_401_unauthorized_is_audited(self, monkeypatch, tmp_path):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        monkeypatch.setenv("A2A_BEARER_TOKEN", "topsecret")
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(security, "_rejection_events", [])
        profile_home = tmp_path / "secondary"
        scope = set_hermes_home_override(profile_home)
        try:
            adapter, base = _make_live_adapter(monkeypatch)
        finally:
            reset_hermes_home_override(scope)

        async def run():
            assert await adapter.connect() is True
            try:
                with pytest.raises(urllib.error.HTTPError) as rejected:
                    await asyncio.to_thread(_post_json, base + "/", _send_body("x"))
                with rejected.value as response:
                    assert response.code == 401
                    assert json.load(response)["error"]["code"] == protocol.ERR_UNAUTHORIZED
            finally:
                await adapter.disconnect()

            assert not (tmp_path / "a2a_audit.jsonl").exists()
            rec = json.loads((profile_home / "a2a_audit.jsonl").read_text())
            assert rec["direction"] == "inbound"
            assert rec["decision"] == "rejected_bad_token"
            assert rec["status"] == 401

        asyncio.run(run())

    def test_403_untrusted_peer_is_audited(self, monkeypatch, tmp_path):
        """A valid token from a non-trusted peer is rejected *and* logged."""
        monkeypatch.setenv("A2A_PEER_TOKENS", "mallory:tok-mallory")
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.setenv("A2A_TRUSTED_PEERS", "alice")
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(security, "_rejection_events", [])
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True

            try:
                with pytest.raises(urllib.error.HTTPError) as rejected:
                    await asyncio.to_thread(
                        _post_json, base + "/", _send_body("do a thing"),
                        {"Authorization": "Bearer tok-mallory"},
                    )
                with rejected.value as response:
                    assert response.code == 403
                    assert json.load(response)["error"]["code"] == protocol.ERR_UNTRUSTED_PEER
            finally:
                await adapter.disconnect()

            audit_file = tmp_path / "a2a_audit.jsonl"
            assert audit_file.exists()
            rec = json.loads(audit_file.read_text().strip().splitlines()[-1])
            assert rec["direction"] == "inbound"
            assert rec["peer"] == "mallory"
            assert rec["decision"] == "rejected_untrusted_peer"
            assert rec["status"] == 403

        asyncio.run(run())
