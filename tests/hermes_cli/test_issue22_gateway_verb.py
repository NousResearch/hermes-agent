"""Isolated synthetic contracts for hermes_cli.issue22_gateway_verb.

No production DB, config, GitHub or network access; all external calls are mocked.
"""
from __future__ import annotations

import importlib
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

issue22 = importlib.import_module("hermes_cli.issue22_gateway_verb")
ingress = importlib.import_module("hermes_cli.issue_ingress")

ISSUE22_VERB = issue22.ISSUE22_VERB
build_handler = issue22.build_issue22_admit_handler
AdmissionRefused = ingress.AdmissionRefused
AdmissionReceipt = ingress.AdmissionReceipt


def make_receipt(verdict="ACK", kind="PING", body_sha256="0" * 64,
                 repository_id=1, issue_number=22, task_id="t1",
                 session_id="s1", nonce="n"):
    return AdmissionReceipt(verdict, kind, repository_id, issue_number,
                           task_id, session_id, nonce, body_sha256)


class Issue22VerbContract(unittest.TestCase):
    def setUp(self):
        self.cfg_path = Path("/tmp/hermes-issue22-config-fixture.json")  # Path-only sentinel; mocks prevent file/network I/O
        self.admit_mock = MagicMock()
        self.validate_mock = MagicMock(return_value=SimpleNamespace())
        self.conn = MagicMock()

    # ---------- Контракт и константы ----------
    def test_01_verb_constant_value(self):
        self.assertEqual(ISSUE22_VERB, "issue22-admit")

    # ---------- Default disable ----------
    def test_10_disabled_default_returns_reject_disabled(self):
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path)
        out = h({"issue_number": 22, "comment_id": 1})
        self.assertEqual(out, {"verdict": "REJECT", "reason": "INGRESS_DISABLED"})

    def test_11_disabled_even_with_invalid_params(self):
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=False)
        # Disable идёт первым; schema даже не проверяется.
        out = h({"issue_number": 999, "comment_id": -1})
        self.assertEqual(out["reason"], "INGRESS_DISABLED")

    # ---------- REQUEST_SCHEMA: shape ----------
    def test_20_reject_non_dict_params(self):
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        for bad in (None, "x", 1, [1, 2]):
            self.assertEqual(h(bad)["reason"], "REQUEST_SCHEMA", msg=f"input={bad!r}")
        self.validate_mock.assert_not_called()
        self.admit_mock.assert_not_called()

    def test_21_reject_extra_or_missing_keys(self):
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        cases = [
            {},                                  # missing
            {"issue_number": 22},                # missing comment_id
            {"comment_id": 1},                   # missing issue_number
            {"issue_number": 22, "comment_id": 1, "x": 1},  # extra
        ]
        for bad in cases:
            self.assertEqual(h(bad)["reason"], "REQUEST_SCHEMA", msg=f"input={bad!r}")

    def test_22_reject_non_int_values(self):
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        for bad_iss in (22.0, "22", True, None):
            self.assertEqual(
                h({"issue_number": bad_iss, "comment_id": 1})["reason"],
                "REQUEST_SCHEMA", msg=f"issue_number={bad_iss!r}")
        for bad_c in ("1", 1.0, True, None):
            self.assertEqual(
                h({"issue_number": 22, "comment_id": bad_c})["reason"],
                "REQUEST_SCHEMA", msg=f"comment_id={bad_c!r}")

    def test_23_reject_other_issue_numbers(self):
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        for n in (0, 1, 21, 23, 100):
            self.assertEqual(
                h({"issue_number": n, "comment_id": 1})["reason"],
                "REQUEST_SCHEMA", msg=f"issue_number={n}")

    def test_24_reject_comment_id_boundaries(self):
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        # boundary < 1 и > MAX_COMMENT_ID (2^63 - 1)
        max_ok = (1 << 63) - 1
        self.assertEqual(
            h({"issue_number": 22, "comment_id": 0})["reason"], "REQUEST_SCHEMA")
        self.assertEqual(
            h({"issue_number": 22, "comment_id": -1})["reason"], "REQUEST_SCHEMA")
        self.assertEqual(
            h({"issue_number": 22, "comment_id": max_ok + 1})["reason"],
            "REQUEST_SCHEMA")
        # ровно 1 и ровно MAX — должны пройти schema и дойти до validate_config
        self.validate_mock.reset_mock()
        self.admit_mock.return_value = make_receipt()
        h({"issue_number": 22, "comment_id": 1})
        h({"issue_number": 22, "comment_id": max_ok})
        self.assertEqual(self.validate_mock.call_count, 2)

    # ---------- Успешный путь ----------
    def test_30_ack_ping_passes_through(self):
        self.admit_mock.return_value = make_receipt(verdict="ACK", kind="PING")
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        out = h({"issue_number": 22, "comment_id": 7})
        self.assertEqual(out, {"verdict": "ACK", "kind": "PING", "issue_number": 22})
        self.validate_mock.assert_called_once_with(self.cfg_path)
        self.admit_mock.assert_called_once_with(
            self.conn, config_path=self.cfg_path, comment_id=7)

    def test_poll_selects_pinned_typed_task_comment_without_forwarding_body(self):
        cfg = SimpleNamespace(
            owner="ISVGitHub", repo="autonomous-entity", issue_ingress_actor_id=548464,
            request_timeout=30, issue_ingress_issue_number=22, issue_ingress_max_body_bytes=1024,
        )
        fake_auth = MagicMock()
        fake_auth.fetch_installation_token.return_value = SimpleNamespace(
            permissions={"issues": "read"},
        )
        fake_auth.list_issue_comments.return_value = [{
            "id": 81,
            "user": {"id": 548464, "type": "User"},
            "created_at": "2026-10-18T00:00:00Z",
            "updated_at": "2026-10-18T00:00:00Z",
            "body": ('{"kind":"TASK","nonce":"abcdefghijklmnop",'
                     f'"expires_at":{int(time.time()) + 300},"task":"review the release"}}'),
        }]
        accept = MagicMock(return_value={"verdict": "ACK", "state": "awaiting_binding"})
        handler = issue22.build_issue22_poll_handler(
            config_path=self.cfg_path, enabled=True, admit_handler=accept,
        )
        with (patch.object(issue22, "load_config", return_value=cfg),
              patch.object(issue22.dataclasses, "replace", return_value=cfg),
              patch.object(issue22, "GitHubAppAuth", return_value=fake_auth)):
            result = handler({})
        self.assertEqual(result, {"verdict": "ACK", "state": "awaiting_binding"})
        accept.assert_called_once_with({"issue_number": 22, "comment_id": 81})
        fake_auth.list_issue_comments.assert_called_once_with(
            "ISVGitHub", "autonomous-entity", 22, per_page=100,
        )

    def test_31_ack_status_passes_through(self):
        self.admit_mock.return_value = make_receipt(verdict="ACK", kind="STATUS")
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        out = h({"issue_number": 22, "comment_id": 7})
        self.assertEqual(out, {"verdict": "ACK", "kind": "STATUS", "issue_number": 22})

    def test_31a_task_ack_is_explicitly_pending_and_never_dispatched(self):
        self.admit_mock.return_value = make_receipt(verdict="ACK", kind="TASK")
        published = []
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: published.append(kwargs) or 999,
                          record_result=lambda *args, **kwargs: None,
                          bind_task=lambda *_a, **_kw: None)
        out = h({"issue_number": 22, "comment_id": 81})
        self.assertEqual(out, {
            "verdict": "ACK", "kind": "TASK", "issue_number": 22,
            "state": "awaiting_binding",
        })
        self.assertEqual(published, [{
            "comment_id": 81, "kind": "TASK", "state": "awaiting_binding",
        }])
        self.assertEqual(self.admit_mock.call_args.kwargs["comment_id"], 81)

    def test_task_ack_reports_queued_after_atomic_binding(self):
        self.admit_mock.return_value = make_receipt(verdict="ACK", kind="TASK")
        published = []
        h = build_handler(
            connect_owner_db=lambda: self.conn,
            config_path=self.cfg_path, enabled=True,
            validate_config=self.validate_mock, admit=self.admit_mock,
            bind_task=lambda *_args, **_kwargs: {"verdict": "ACK", "state": "queued"},
            publish_result=lambda *args, **kwargs: published.append(kwargs) or 999,
            record_result=lambda *args, **kwargs: None,
        )
        self.assertEqual(h({"issue_number": 22, "comment_id": 81}), {
            "verdict": "ACK", "kind": "TASK", "issue_number": 22, "state": "queued",
        })
        self.assertEqual(published, [{
            "comment_id": 81, "kind": "TASK", "state": "queued",
        }])

    def test_32_receipt_with_wrong_verdict(self):
        self.admit_mock.return_value = make_receipt(verdict="REJECT", kind="PING")
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        out = h({"issue_number": 22, "comment_id": 7})
        self.assertEqual(out, {"verdict": "REJECT", "reason": "OWNER_RECEIPT_INVALID"})

    def test_33_receipt_with_disallowed_kind(self):
        self.admit_mock.return_value = make_receipt(verdict="ACK", kind="EXEC")
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        out = h({"issue_number": 22, "comment_id": 7})
        self.assertEqual(out, {"verdict": "REJECT", "reason": "OWNER_RECEIPT_INVALID"})

    # ---------- Конверсия ошибок ----------
    def test_40_admission_refused_converted_to_reject_reason(self):
        self.admit_mock.side_effect = AdmissionRefused("TRUSTED_INGRESS_PINS_MISSING")
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        out = h({"issue_number": 22, "comment_id": 1})
        self.assertEqual(out, {"verdict": "REJECT",
                               "reason": "TRUSTED_INGRESS_PINS_MISSING"})

    def test_41_unexpected_exception_becomes_owner_error(self):
        self.admit_mock.side_effect = RuntimeError("boom")
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        out = h({"issue_number": 22, "comment_id": 1})
        self.assertEqual(out, {"verdict": "REJECT", "reason": "OWNER_ERROR"})

    def test_42_validate_config_failure_short_circuits(self):
        # validate_config бросает AdmissionRefused ДО admit, потому что
        # идёт раньше в коде handler.
        self.validate_mock.side_effect = AdmissionRefused("TRUSTED_INGRESS_PINS_MISSING")
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        out = h({"issue_number": 22, "comment_id": 1})
        self.assertEqual(out, {"verdict": "REJECT",
                               "reason": "TRUSTED_INGRESS_PINS_MISSING"})
        self.admit_mock.assert_not_called()

    # ---------- Ресурсы ----------
    def test_50_connection_closed_on_success(self):
        self.admit_mock.return_value = make_receipt()
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        h({"issue_number": 22, "comment_id": 1})
        self.conn.close.assert_called_once()

    def test_51_connection_closed_on_error(self):
        self.admit_mock.side_effect = RuntimeError("boom")
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        h({"issue_number": 22, "comment_id": 1})
        self.conn.close.assert_called_once()

    def test_52_connection_closed_on_admission_refused(self):
        self.admit_mock.side_effect = AdmissionRefused("X")
        h = build_handler(connect_owner_db=lambda: self.conn,
                          config_path=self.cfg_path, enabled=True,
                          validate_config=self.validate_mock, admit=self.admit_mock,
                          publish_result=lambda *args, **kwargs: 999,
                          record_result=lambda *args, **kwargs: None)
        h({"issue_number": 22, "comment_id": 1})
        self.conn.close.assert_called_once()

    def test_60_concurrent_requests_are_serialized_fail_closed(self):
        entered = threading.Event()
        release = threading.Event()

        def admit(*args, **kwargs):
            entered.set()
            release.wait(timeout=2)
            return make_receipt()

        h = build_handler(
            connect_owner_db=lambda: MagicMock(), config_path=self.cfg_path,
            enabled=True, validate_config=self.validate_mock, admit=admit,
            publish_result=lambda *args, **kwargs: 999,
            record_result=lambda *args, **kwargs: None,
        )
        first_result = []
        worker = threading.Thread(
            target=lambda: first_result.append(h({"issue_number": 22, "comment_id": 7})),
        )
        worker.start()
        self.assertTrue(entered.wait(timeout=1))
        second = h({"issue_number": 22, "comment_id": 8})
        release.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(second, {"verdict": "REJECT", "reason": "INGRESS_BUSY"})
        self.assertEqual(first_result[0]["verdict"], "ACK")
