"""Tests for the update receipt's ``invoker`` field (self-attributing receipts).

Audit gap this closes: receipts carried argv + pid + ``correlation_id``, but
``correlation_id`` is only populated from ``HERMES_UPDATE_CORRELATION_ID``
(dashboard managed-SSH launcher). Plain CLI runs recorded ``null`` and no
invoker/source at all — the Oct-18 hermes-spare update receipts were
unattributable from the receipt alone (session/kanban log correlation
required). The receipt now carries an ``invoker`` captured at begin:

- ``kanban:<task>`` — ``HERMES_KANBAN_TASK`` is set (dispatcher worker);
- ``session:<id>``  — ``HERMES_SESSION_ID`` is set (agent session);
- ``cli[:tty:<name>[:login:<name>]]`` — anything else (human shell, cron,
  unattended), with tty/login info when available.

Environment-only and stdlib-only, so a sandboxed test drives it with monkeypatched env +
os.ttyname/os.getlogin; no live terminals, no real home.
"""
import json
import os

import pytest

import hermes_cli.update_receipt as ur


@pytest.fixture()
def receipt_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME for receipt writes."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with ur.update_receipt_scope():
        yield home


def _clear_invoker_env(monkeypatch, *, tty=None, login="tester", tty_error=False, login_error=False):
    """Reset the attribution env and pin the stdlib probes to deterministic values.

    ``tty`` — value the tty probe returns (``None`` → the probe is disabled/absent);
    ``tty_error`` — the tty probe raises (no controlling terminal);
    ``login_error`` — the login probe raises. Plain ``OSError`` doubles — never
    ``pytest.fail`` inside the classifier's ``suppress(Exception)``.
    """
    for name in ("HERMES_KANBAN_TASK", "HERMES_SESSION_ID", "HERMES_UPDATE_CORRELATION_ID"):
        monkeypatch.delenv(name, raising=False)
    if tty_error:
        monkeypatch.setattr(os, "ttyname", lambda fd: (_ for _ in ()).throw(OSError(6, "No tty")))
    elif tty is None:
        monkeypatch.delattr(os, "ttyname", raising=False)
    else:
        monkeypatch.setattr(os, "ttyname", lambda fd: tty)
    if login_error:
        monkeypatch.setattr(os, "getlogin", lambda: (_ for _ in ()).throw(OSError(6, "No tty")))
    else:
        monkeypatch.setattr(os, "getlogin", lambda: login)


def _receipt_payload(receipt_home, outcome="success"):
    ur.begin_update_receipt()
    path = ur.finalize_update_receipt(outcome)
    assert path is not None and path.is_file()
    return json.loads(path.read_text(encoding="utf-8"))


class TestClassifyInvoker:
    def test_kanban_worker_names_the_task(self, monkeypatch):
        _clear_invoker_env(monkeypatch)
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_ee576527")
        assert ur._classify_invoker() == "kanban:t_ee576527"

    def test_session_when_no_kanban_task(self, monkeypatch):
        _clear_invoker_env(monkeypatch)
        monkeypatch.setenv("HERMES_SESSION_ID", "abcdef-1234")
        assert ur._classify_invoker() == "session:abcdef-1234"

    def test_kanban_wins_over_session(self, monkeypatch):
        _clear_invoker_env(monkeypatch)
        # A worker running a cron job carries BOTH: the task is the finer attribution.
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
        monkeypatch.setenv("HERMES_SESSION_ID", "sess-1")
        assert ur._classify_invoker() == "kanban:t_abc"

    def test_cli_with_tty_and_login(self, monkeypatch):
        _clear_invoker_env(monkeypatch, tty="/dev/pts/3")
        assert ur._classify_invoker() == "cli:tty:/dev/pts/3:login:tester"

    def test_cli_without_tty_keeps_login(self, monkeypatch):
        _clear_invoker_env(monkeypatch)  # tty probe absent -> no tty in the value
        assert ur._classify_invoker() == "cli:login:tester"

    def test_null_tty_is_not_recorded(self, monkeypatch):
        _clear_invoker_env(monkeypatch, tty="/dev/null")
        assert ur._classify_invoker() == "cli:login:tester"

    def test_tty_probe_failing_degrades_gracefully(self, monkeypatch):
        _clear_invoker_env(monkeypatch, tty_error=True)
        assert ur._classify_invoker() == "cli:login:tester"

    def test_login_failure_degrades_to_bare_cli(self, monkeypatch):
        _clear_invoker_env(monkeypatch, tty_error=True, login_error=True)
        assert ur._classify_invoker() == "cli"


class TestReceiptCarriesInvoker:
    def test_kanban_receipt_is_self_attributing(self, receipt_home, monkeypatch):
        _clear_invoker_env(monkeypatch)
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_ee576527")
        payload = _receipt_payload(receipt_home)
        assert payload["invoker"] == "kanban:t_ee576527"
        # The audit scenario: correlation_id is still null (no launcher env), yet the
        # receipt alone now names its invoker — no log correlation needed.
        assert payload["correlation_id"] is None
        assert payload["update_id"]

    def test_session_receipt(self, receipt_home, monkeypatch):
        _clear_invoker_env(monkeypatch)
        monkeypatch.setenv("HERMES_SESSION_ID", "sess-42")
        payload = _receipt_payload(receipt_home)
        assert payload["invoker"] == "session:sess-42"

    def test_plain_cli_receipt(self, receipt_home, monkeypatch):
        _clear_invoker_env(monkeypatch)
        payload = _receipt_payload(receipt_home)
        assert payload["invoker"] == "cli:login:tester"

    def test_dashboard_launcher_correlation_still_wins(self, receipt_home, monkeypatch):
        # The existing contract is untouched: the managed-SSH launcher's correlation id
        # still lands in correlation_id, and the invoker records the launching context
        # (a CLI shell that has no session binding of its own).
        _clear_invoker_env(monkeypatch)
        monkeypatch.setenv("HERMES_UPDATE_CORRELATION_ID", "desk-abc")
        payload = _receipt_payload(receipt_home)
        assert payload["correlation_id"] == "desk-abc"
        assert payload["invoker"] == "cli:login:tester"

    def test_attribution_failure_never_breaks_the_receipt(self, receipt_home, monkeypatch):
        # The classifier must never break the receipt: if it raises (however exotic the
        # environment), the receipt still writes a complete record — invoker is absent,
        # the pre-field behaviour — and the run still finalizes.
        monkeypatch.setattr(ur, "_classify_invoker", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        payload = _receipt_payload(receipt_home)
        assert payload.get("invoker") is None
        assert payload["outcome"] == "success"
        assert payload["update_id"] and payload["finished_at"]

    def test_running_record_carries_invoker_before_finalize(self, receipt_home, monkeypatch):
        _clear_invoker_env(monkeypatch)
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_running")
        ur.begin_update_receipt()
        stored = ur.read_latest_receipt()
        assert stored is not None and stored.get("invoker") == "kanban:t_running"
        ur.finalize_update_receipt("success")
