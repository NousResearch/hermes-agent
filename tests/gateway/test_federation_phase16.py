"""Phase 16 unit tests — security hardening of the federation module.

Covers:
  CRITICAL-1: TLS mandatory (Pillar 2)
  CRITICAL-2: Trust 评级 + Task 敏感度 (Pillar 4)
  CRITICAL-3: 3-failure death rule (Pillar 5)
  CRITICAL-4: Audit log (Pillar 6)
  HIGH-1: Task state HMAC (Pillar 3)
  HIGH-2: Token Keychain (Pillar 8)
  HIGH-3: Heartbeat payload sanitization (Pillar 7)

Each test class is self-contained and can be run independently.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest


# === CRITICAL-1: TLS mandatory ===

class TestTLSSecurity:
    """FederationConfig must enforce TLS by default."""

    def test_default_require_tls_true(self):
        from gateway.config import FederationConfig
        cfg = FederationConfig()
        assert cfg.require_tls is True
        assert cfg.allow_insecure is False

    def test_lan_mode_without_tls_emits_critical(self):
        from gateway.config import FederationConfig
        cfg = FederationConfig(enabled=True, mode="lan")
        issues = cfg.validate_security()
        assert any("TLS" in i for i in issues)

    def test_lan_mode_with_tls_no_issues(self):
        from gateway.config import FederationConfig
        cfg = FederationConfig(
            enabled=True, mode="lan",
            tls_cert="/etc/ssl/cert.pem",
            tls_key="/etc/ssl/key.pem",
        )
        issues = cfg.validate_security()
        assert len(issues) == 0

    def test_allow_insecure_logs_warning(self):
        from gateway.config import FederationConfig
        cfg = FederationConfig(
            enabled=True, mode="lan", allow_insecure=True,
        )
        # First call triggers warning
        issues = cfg.validate_security()
        assert len(issues) == 0  # no hard issues, just warning
        assert cfg._insecure_warned is True

    def test_shared_db_mode_exempt(self):
        from gateway.config import FederationConfig
        cfg = FederationConfig(enabled=True, mode="shared_db")
        issues = cfg.validate_security()
        assert len(issues) == 0

    def test_disabled_federation_no_validation(self):
        from gateway.config import FederationConfig
        cfg = FederationConfig(enabled=False, mode="lan")
        issues = cfg.validate_security()
        assert len(issues) == 0  # disabled = no TLS requirement


# === CRITICAL-2: Trust 评级 + Task 敏感度 ===

class TestTrustPolicy:
    """Trust 评级 + Task 敏感度 enforcement."""

    def test_trust_levels_ordered(self):
        from gateway.federation.trust import TrustLevel
        assert TrustLevel.UNKNOWN.rank < TrustLevel.VERIFIED.rank
        assert TrustLevel.VERIFIED.rank < TrustLevel.TRUSTED.rank
        assert TrustLevel.TRUSTED.rank < TrustLevel.ADMIN.rank

    def test_sensitivity_min_trust(self):
        from gateway.federation.trust import TaskSensitivity, TrustLevel
        assert TaskSensitivity.LOW.min_trust == TrustLevel.VERIFIED
        assert TaskSensitivity.MEDIUM.min_trust == TrustLevel.VERIFIED
        assert TaskSensitivity.HIGH.min_trust == TrustLevel.TRUSTED
        assert TaskSensitivity.CRITICAL.min_trust == TrustLevel.ADMIN

    def test_can_claim_basic(self):
        from gateway.federation.trust import (
            TrustPolicy, TrustLevel, TaskSensitivity,
        )
        p = TrustPolicy()
        assert p.can_claim(TrustLevel.VERIFIED, TaskSensitivity.LOW)
        assert p.can_claim(TrustLevel.VERIFIED, TaskSensitivity.MEDIUM)
        assert not p.can_claim(TrustLevel.VERIFIED, TaskSensitivity.HIGH)
        assert not p.can_claim(TrustLevel.VERIFIED, TaskSensitivity.CRITICAL)
        assert p.can_claim(TrustLevel.TRUSTED, TaskSensitivity.HIGH)
        assert p.can_claim(TrustLevel.ADMIN, TaskSensitivity.CRITICAL)

    def test_can_claim_unknown_string_fail_closed(self):
        from gateway.federation.trust import (
            TrustPolicy, TrustLevel, TaskSensitivity,
        )
        p = TrustPolicy()
        # Garbage trust = no access
        assert not p.can_claim("garbage", TaskSensitivity.LOW)
        # Valid string still works
        assert p.can_claim("verified", TaskSensitivity.LOW)

    def test_can_claim_unknown_sensitivity_fail_closed(self):
        from gateway.federation.trust import TrustPolicy, TrustLevel
        p = TrustPolicy()
        # Garbage sensitivity = no access
        assert not p.can_claim(TrustLevel.ADMIN, "garbage")

    def test_should_alert_on_denial(self):
        from gateway.federation.trust import TrustPolicy, TrustLevel, TaskSensitivity
        p = TrustPolicy()
        assert p.should_alert(TrustLevel.VERIFIED, TaskSensitivity.HIGH)
        assert p.should_alert(TrustLevel.TRUSTED, TaskSensitivity.CRITICAL)

    def test_should_alert_on_critical_touched(self):
        from gateway.federation.trust import TrustPolicy, TrustLevel, TaskSensitivity
        p = TrustPolicy()
        # Even successful critical touch is alerted
        assert p.should_alert(TrustLevel.ADMIN, TaskSensitivity.CRITICAL)

    def test_infer_sensitivity_critical(self):
        from gateway.federation.trust import infer_sensitivity, TaskSensitivity
        assert infer_sensitivity("deploy production cluster") == TaskSensitivity.CRITICAL
        assert infer_sensitivity("delete test files") == TaskSensitivity.CRITICAL
        assert infer_sensitivity("force push to main") == TaskSensitivity.CRITICAL
        assert infer_sensitivity("drop database") == TaskSensitivity.CRITICAL

    def test_infer_sensitivity_high(self):
        from gateway.federation.trust import infer_sensitivity, TaskSensitivity
        assert infer_sensitivity("npm publish new version") == TaskSensitivity.HIGH
        assert infer_sensitivity("apply migration") == TaskSensitivity.HIGH
        assert infer_sensitivity("update user email") == TaskSensitivity.HIGH

    def test_infer_sensitivity_medium_default(self):
        from gateway.federation.trust import infer_sensitivity, TaskSensitivity
        assert infer_sensitivity("read README.md") == TaskSensitivity.MEDIUM
        assert infer_sensitivity("") == TaskSensitivity.MEDIUM


# === CRITICAL-3: 3-failure death rule ===

class TestDeathDetector:
    """3-failure death rule with thread safety."""

    def test_3_failures_confirm_death(self):
        from gateway.federation.death_detector import DeathDetector
        d = DeathDetector(dead_threshold=3)
        assert not d.record_failure("a")
        assert not d.record_failure("a")
        assert d.record_failure("a")  # 3rd failure
        assert d.is_dead("a")

    def test_threshold_configurable(self):
        from gateway.federation.death_detector import DeathDetector
        d = DeathDetector(dead_threshold=5)
        for _ in range(4):
            assert not d.is_dead("a")
        d.record_failure("a")
        d.record_failure("a")
        d.record_failure("a")
        d.record_failure("a")
        assert not d.is_dead("a")
        d.record_failure("a")
        assert d.is_dead("a")

    def test_threshold_zero_rejected(self):
        from gateway.federation.death_detector import DeathDetector
        with pytest.raises(ValueError):
            DeathDetector(dead_threshold=0)

    def test_success_resets_failure_count(self):
        from gateway.federation.death_detector import DeathDetector
        d = DeathDetector(dead_threshold=3)
        d.record_failure("a")
        d.record_failure("a")
        # Reset — peer NOT yet dead, success returns False
        assert d.record_success("a") is False
        assert not d.is_dead("a")
        # Fresh failure cycle starts at 0
        d.record_failure("a")
        d.record_failure("a")
        assert not d.is_dead("a")  # only 2 failures, not dead

    def test_revive_returns_true(self):
        from gateway.federation.death_detector import DeathDetector
        d = DeathDetector(dead_threshold=2)
        d.record_failure("a")
        d.record_failure("a")
        assert d.is_dead("a")
        assert d.record_success("a")  # was dead, now revived
        assert not d.is_dead("a")

    def test_thread_safe_concurrent_failures(self):
        from gateway.federation.death_detector import DeathDetector
        d = DeathDetector(dead_threshold=100)
        def hammer():
            for _ in range(50):
                d.record_failure("x")
        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert d.get_status("x").failure_count == 200

    def test_all_dead(self):
        from gateway.federation.death_detector import DeathDetector
        d = DeathDetector(dead_threshold=2)
        d.record_failure("a")
        d.record_failure("a")
        d.record_failure("b")
        d.record_failure("b")
        d.record_failure("c")  # only 1, alive
        dead = set(d.all_dead())
        assert "a" in dead
        assert "b" in dead
        assert "c" not in dead

    def test_clear_forgets_peer(self):
        from gateway.federation.death_detector import DeathDetector
        d = DeathDetector(dead_threshold=2)
        d.record_failure("a")
        d.record_failure("a")
        assert d.is_dead("a")
        d.clear("a")
        assert not d.is_dead("a")
        assert d.get_status("a") is None


# === CRITICAL-4: Audit log ===

class TestAuditLog:
    """HMAC-chained audit log with tamper detection."""

    def test_token_redaction_short(self):
        from gateway.federation.audit import TokenStr
        assert repr(TokenStr("ab")) == "***"

    def test_token_redaction_long(self):
        from gateway.federation.audit import TokenStr
        assert repr(TokenStr("hermes_abc123def456")) == "herm***f456"

    def test_token_redaction_sk_prefix(self):
        from gateway.federation.audit import TokenStr
        assert repr(TokenStr("sk-1234567890abcdef")) == "sk-1***cdef"

    def test_recursive_redaction(self):
        from gateway.federation.audit import redact
        d = {
            "token": "hermes_abc123def456",
            "name": "mac-a",
            "nested": {"key": "sk-1234567890abcdef"},
            "list": ["plain", "hermes_zzzzz"],
        }
        r = redact(d)
        assert "herm***" in r["token"]
        assert r["name"] == "mac-a"
        assert "sk-1***" in r["nested"]["key"]
        assert r["list"][0] == "plain"
        assert "herm***" in r["list"][1]

    def test_chain_integrity(self, tmp_path):
        from gateway.federation.audit import (
            AuditLog, NodeEvent, TaskEvent, SecurityEvent, UserEvent,
        )
        log_path = tmp_path / "audit.log"
        log = AuditLog(cluster_secret="test-secret", log_path=log_path)
        log.append(NodeEvent.join("mac-a", "unknown"))
        log.append(NodeEvent.join("mac-b", "verified"))
        log.append(TaskEvent.create("t-123", "Analyze", "mac-a"))
        log.append(TaskEvent.claim("t-123", "mac-a", "mac-b"))
        log.append(SecurityEvent.death_confirmed("mac-a", failure_count=3))
        log.append(UserEvent.decision("user", "accept", "t-123"))
        assert log.verify_chain()

    def test_chain_tamper_detection(self, tmp_path):
        from gateway.federation.audit import AuditLog, NodeEvent
        log_path = tmp_path / "audit.log"
        log = AuditLog(cluster_secret="test-secret", log_path=log_path)
        log.append(NodeEvent.join("mac-a", "unknown"))
        log.append(NodeEvent.join("mac-b", "verified"))
        # Tamper with file
        with open(log_path, "r") as f:
            lines = f.readlines()
        lines[1] = lines[1].replace("mac-b", "evil-node")
        with open(log_path, "w") as f:
            f.writelines(lines)
        assert not log.verify_chain()

    def test_query_by_event_type(self, tmp_path):
        from gateway.federation.audit import AuditLog, NodeEvent, TaskEvent
        log_path = tmp_path / "audit.log"
        log = AuditLog(cluster_secret="test-secret", log_path=log_path)
        log.append(NodeEvent.join("a"))
        log.append(NodeEvent.join("b"))
        log.append(TaskEvent.create("t-1", "Test", "a"))
        joins = log.query(event_type="node.join")
        assert len(joins) == 2
        creates = log.query(event_type="task.create")
        assert len(creates) == 1

    def test_query_by_target(self, tmp_path):
        from gateway.federation.audit import AuditLog, NodeEvent, TaskEvent
        log_path = tmp_path / "audit.log"
        log = AuditLog(cluster_secret="test-secret", log_path=log_path)
        log.append(NodeEvent.join("a"))
        log.append(TaskEvent.create("t-1", "Test", "a"))
        log.append(TaskEvent.create("t-2", "Test2", "a"))
        events = log.query(target="t-1")
        assert len(events) == 1
        assert events[0].target == "t-1"

    def test_security_event_alert_severity(self, tmp_path):
        from gateway.federation.audit import AuditLog, SecurityEvent
        log_path = tmp_path / "audit.log"
        log = AuditLog(cluster_secret="test", log_path=log_path)
        log.append(SecurityEvent.death_confirmed("a"))
        log.append(SecurityEvent.signature_invalid("a", "t-1"))
        log.append(SecurityEvent.access_denied("a", "t-1", "trust_low"))
        events = log.query()
        for e in events:
            assert e.severity == "alert"


# === HIGH-1: Task state HMAC ===

class TestSignedTaskState:
    """Task state HMAC integrity."""

    def test_sign_and_verify(self):
        from gateway.federation.task_state import SignedTaskState
        s = SignedTaskState(task_id="t-123", owner="mac-a", step=3, total=10)
        s.sign("cluster-secret")
        assert s.verify("cluster-secret")

    def test_tamper_detection(self):
        from gateway.federation.task_state import SignedTaskState
        s = SignedTaskState(task_id="t-123", owner="mac-a", step=3, total=10)
        s.sign("cluster-secret")
        s.step = 99  # tamper
        assert not s.verify("cluster-secret")

    def test_wrong_secret(self):
        from gateway.federation.task_state import SignedTaskState
        s = SignedTaskState(task_id="t-123", owner="mac-a", step=3, total=10)
        s.sign("correct-secret")
        assert not s.verify("wrong-secret")

    def test_round_trip_json(self):
        from gateway.federation.task_state import SignedTaskState
        s = SignedTaskState(
            task_id="t-123", owner="mac-a", step=3, total=10,
            status="in_progress", required_capability=["cpu"],
            sensitivity="medium",
        )
        s.sign("abc")
        j = s.to_json()
        s2 = SignedTaskState.from_json(j)
        assert s2.verify("abc")
        assert s2.task_id == "t-123"
        assert s2.owner == "mac-a"
        assert s2.step == 3
        assert s2.required_capability == ["cpu"]

    def test_partial_result_not_signed(self):
        """partial_result is NOT part of signature — caller encrypts separately."""
        from gateway.federation.task_state import SignedTaskState
        s = SignedTaskState(
            task_id="t-123", owner="mac-a",
            partial_result={"sensitive": "data"},
        )
        s.sign("secret")
        # Tamper with partial_result
        s.partial_result = {"sensitive": "different"}
        # Signature is still valid (intentional design)
        assert s.verify("secret")

    def test_signature_required(self):
        from gateway.federation.task_state import SignedTaskState
        s = SignedTaskState(task_id="t-123", owner="mac-a")
        # No signature yet
        assert not s.verify("secret")

    def test_sign_requires_secret(self):
        from gateway.federation.task_state import SignedTaskState
        s = SignedTaskState(task_id="t-123", owner="mac-a")
        with pytest.raises(ValueError):
            s.sign("")


# === HIGH-2: Token Keychain ===

class TestSecretStore:
    """Multi-backend secret storage."""

    def test_encrypted_file_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from gateway.federation.secret_store import EncryptedFileBackend
        ef = EncryptedFileBackend()
        ef.set("test/key", "secret-value")
        assert ef.get("test/key") == "secret-value"

    def test_encrypted_file_permissions(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from gateway.federation.secret_store import EncryptedFileBackend
        ef = EncryptedFileBackend()
        ef.set("test/key", "secret")
        ef_path = tmp_path / ".hermes/federation/secrets.json.enc"
        assert ef_path.exists()
        mode = ef_path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_encrypted_no_plaintext_leak(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from gateway.federation.secret_store import EncryptedFileBackend
        ef = EncryptedFileBackend()
        secret = "my-super-secret-cluster-key-1234567890"
        ef.set("cluster_secret", secret)
        ef_path = tmp_path / ".hermes/federation/secrets.json.enc"
        raw = ef_path.read_bytes()
        assert secret.encode() not in raw, "Plaintext leak!"
        assert "cluster_secret" not in raw.decode("utf-8", errors="replace")

    def test_secret_store_rotate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from gateway.federation.secret_store import (
            SecretStore, EncryptedFileBackend,
        )
        store = SecretStore()
        store._backends = [EncryptedFileBackend()]
        store.set("federation.cluster_secret", "old-secret-1234567890")
        old = store.rotate("federation.cluster_secret", "new-secret-abc1234567890")
        assert str(old) == "old-secret-1234567890"
        new = store.get("federation.cluster_secret")
        assert str(new) == "new-secret-abc1234567890"

    def test_secret_store_delete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from gateway.federation.secret_store import (
            SecretStore, EncryptedFileBackend,
        )
        store = SecretStore()
        store._backends = [EncryptedFileBackend()]
        store.set("k", "v")
        assert store.get("k") is not None
        store.delete("k")
        assert store.get("k") is None

    def test_repr_redacted_log(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from gateway.federation.secret_store import (
            SecretStore, EncryptedFileBackend,
        )
        from gateway.federation.audit import TokenStr
        store = SecretStore()
        store._backends = [EncryptedFileBackend()]
        store.set("k", "secret-1234567890abcdef")
        v = store.get("k")
        assert isinstance(v, TokenStr)
        assert str(v) == "secret-1234567890abcdef"  # usable
        assert repr(v) == "secr***cdef"  # redacted in logs


# === HIGH-3: Heartbeat payload sanitization ===

class TestHeartbeatPayload:
    """Heartbeat payload must NOT contain task content."""

    def test_strip_task_payload(self):
        from gateway.federation.heartbeat_payload import sanitize_heartbeat
        raw = {"node_id": "a", "task_payload": "SECRET"}
        clean = sanitize_heartbeat(raw).to_dict()
        assert "task_payload" not in clean

    def test_strip_memory_content(self):
        from gateway.federation.heartbeat_payload import sanitize_heartbeat
        raw = {"node_id": "a", "memory_content": "private"}
        clean = sanitize_heartbeat(raw).to_dict()
        assert "memory_content" not in clean

    def test_strip_user_input(self):
        from gateway.federation.heartbeat_payload import sanitize_heartbeat
        raw = {"node_id": "a", "user_input": "hi"}
        clean = sanitize_heartbeat(raw).to_dict()
        assert "user_input" not in clean

    def test_strip_user_email(self):
        from gateway.federation.heartbeat_payload import sanitize_heartbeat
        raw = {"node_id": "a", "user_email": "a@b.com"}
        clean = sanitize_heartbeat(raw).to_dict()
        assert "user_email" not in clean

    def test_keep_current_task_id(self):
        from gateway.federation.heartbeat_payload import sanitize_heartbeat
        raw = {"node_id": "a", "current_task_id": "t-123"}
        clean = sanitize_heartbeat(raw).to_dict()
        assert clean["current_task_id"] == "t-123"

    def test_keep_current_task_step(self):
        from gateway.federation.heartbeat_payload import sanitize_heartbeat
        raw = {"node_id": "a", "current_task_step": 3, "current_task_total": 10}
        clean = sanitize_heartbeat(raw).to_dict()
        assert clean["current_task_step"] == 3
        assert clean["current_task_total"] == 10

    def test_whitelist_preserved(self):
        from gateway.federation.heartbeat_payload import sanitize_heartbeat
        raw = {
            "node_id": "a",
            "hostname": "mac.local",
            "cpu_cores": 18,
            "memory_gb": 128.0,
            "load_avg": 0.5,
            "version": "0.17.0",
        }
        clean = sanitize_heartbeat(raw).to_dict()
        for k, v in raw.items():
            assert clean[k] == v

    def test_safe_field_check(self):
        from gateway.federation.heartbeat_payload import is_safe_field
        assert is_safe_field("node_id")
        assert is_safe_field("cpu_cores")
        assert not is_safe_field("task_payload")
        assert not is_safe_field("user_input")
        assert not is_safe_field("memory_content")

    def test_assert_safe_raises_on_violation(self):
        from gateway.federation.heartbeat_payload import assert_safe
        with pytest.raises(ValueError):
            assert_safe({"node_id": "a", "task_payload": "secret"})

    def test_assert_safe_passes_on_clean(self):
        from gateway.federation.heartbeat_payload import assert_safe
        assert_safe({"node_id": "a", "status": "online"})

    def test_heuristic_strip_unknown_similar(self):
        """Unknown fields with suspicious keywords are stripped."""
        from gateway.federation.heartbeat_payload import sanitize_heartbeat
        raw = {"node_id": "a", "user_phone": "555-1234"}
        clean = sanitize_heartbeat(raw).to_dict()
        assert "user_phone" not in clean


# === Smoke test ===

class TestAllSmoke:
    """All 7 modules work together."""

    def test_modules_importable(self):
        from gateway.federation.trust import (
            TrustLevel, TaskSensitivity, TrustPolicy,
        )
        from gateway.federation.death_detector import DeathDetector
        from gateway.federation.audit import AuditLog, NodeEvent, SecurityEvent
        from gateway.federation.task_state import SignedTaskState
        from gateway.federation.secret_store import SecretStore
        from gateway.federation.heartbeat_payload import (
            HeartbeatPayload, sanitize_heartbeat,
        )
        from gateway.config import FederationConfig

        cfg = FederationConfig(enabled=True, mode="lan")
        assert cfg.validate_security()

        d = DeathDetector()
        assert not d.record_failure("a")

        p = TrustPolicy()
        assert not p.can_claim(TrustLevel.VERIFIED, TaskSensitivity.CRITICAL)

        s = SignedTaskState(task_id="t-1", owner="a")
        s.sign("secret")
        assert s.verify("secret")

        clean = sanitize_heartbeat({"node_id": "a", "task_payload": "secret"})
        assert "task_payload" not in clean.to_dict()
