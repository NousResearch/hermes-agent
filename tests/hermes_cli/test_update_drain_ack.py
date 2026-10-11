"""The updater's drain ACK is the gateway's ``pausing`` — never ``already_stopping`` (#135878).

``pause_gateway_for_update`` answers ``{"pausing", "already_stopping", "restart_in_flight", ...}``.
``already_stopping`` has two sources the raw flag cannot tell apart: a restart genuinely under way
inside the gateway (``request_restart`` refused this request; the restart's own drain exits the
gateway when done) and a request that never landed (the handler's 5s window closed; the gateway
keeps running its turns). The ACK's ``restart_in_flight`` — snapshotted before the request was
dispatched — splits them: the updater keeps the drain wait for an in-flight restart instead of
cutting its mid-flight turn off (a real regression), and takes the immediate ``_stop_at_once``
escalation only for a missed request, with one observable line saying why.
"""

from __future__ import annotations

from pathlib import Path

from gateway import control_socket
from hermes_cli import update_cmd_posix_pause as m


def _ack(**over):
    ack = {"pausing": False, "already_stopping": True, "pid": 4242, "drain_timeout": 1885}
    ack.update(over)
    return ack


def _entry(home) -> dict:
    return {"pid": 4242, "home": str(home)}


class TestDrainAckClassification:
    def test_pausing_is_the_ack(self, tmp_path, monkeypatch):
        ack = _ack(pausing=True, already_stopping=False, restart_in_flight=False)
        monkeypatch.setattr(control_socket, "pause_gateway_for_update", lambda home, **kw: ack)
        assert m._ask_to_drain(_entry(tmp_path)) == ack

    def test_already_stopping_is_not_the_ack(self, tmp_path, monkeypatch):
        ack = _ack(pausing=False, already_stopping=True, restart_in_flight=False)
        monkeypatch.setattr(control_socket, "pause_gateway_for_update", lambda home, **kw: ack)
        answer = m._ask_to_drain(_entry(tmp_path))
        assert isinstance(answer, dict) and not answer.get("pausing"), (
            "already_stopping means the request was refused or missed — the caller must be able "
            "to tell it apart from a real drain ACK"
        )

    def test_no_answer_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(control_socket, "pause_gateway_for_update", lambda home, **kw: None)
        assert m._ask_to_drain(_entry(tmp_path)) is None
        assert m._ask_to_drain({"pid": 4242, "home": None}) is None  # never asked: no home, no request


class TestAlreadyStoppingSplitsByRestartInFlight:
    def _run_stop(self, tmp_path, monkeypatch, capsys, answers_by_name):
        homes, entries = {}, []
        for name in answers_by_name:
            home = tmp_path / name
            home.mkdir()
            homes[name] = home
            entries.append({"pid": sum(map(ord, name)), "home": str(home)})
        monkeypatch.setattr(
            control_socket, "pause_gateway_for_update",
            lambda home, **kw: answers_by_name[Path(home).name])
        import hermes_cli.update_pause_record as pause_record_module

        monkeypatch.setattr(pause_record_module, "mark_stop_sent", lambda token, pid: None)
        monkeypatch.setattr(m, "_drain_budget", lambda: 1885.0)
        monkeypatch.setattr(m, "_alive", lambda pid, ct=None: False)
        stopped: list[int] = []
        monkeypatch.setattr(m, "_stop_at_once", lambda entry: stopped.append(entry["pid"]))
        m._stop_gateways({}, entries)
        return stopped, {e["pid"]: e for e in entries}

    def test_in_flight_restart_keeps_the_drain_wait(self, tmp_path, monkeypatch, capsys):
        # A restart is draining inside the gateway: it exits on its own when done, so the updater
        # must NOT cut its in-flight turn off — the wait is kept exactly like a pausing gateway.
        stopped, entries = self._run_stop(tmp_path, monkeypatch, capsys, {
            "inflight": _ack(pausing=False, already_stopping=True, restart_in_flight=True),
        })
        assert stopped == [], "an in-flight restart drains itself: no immediate stop may fire"
        out = capsys.readouterr().out
        assert "refused or missed" not in out and "never reached it" not in out, (
            "the refusal is a restart in flight — the refused-or-missed warning would be a false claim"
        )
        assert "Waiting up to" in out, "the gateway is waited for within the drain budget"

    def test_missing_request_stops_now_with_truthful_warning(self, tmp_path, monkeypatch, capsys):
        stopped, entries = self._run_stop(tmp_path, monkeypatch, capsys, {
            "missed": _ack(pausing=False, already_stopping=True, restart_in_flight=False),
        })
        assert stopped == [entries[sum(map(ord, "missed"))]["pid"]], (
            "a request that never landed leaves the gateway running: immediate stop"
        )
        out = capsys.readouterr().out
        assert "already_stopping" in out and "gateway PID" in out, (
            "the misclassified state must be observable in one line, not a silent 31 min wait"
        )
        assert "never reached it" in out, (
            "the warning must say the request never landed — a refusal claim would be false here "
            "(the gateway never answered the request)"
        )

    def test_old_gateway_without_the_field_stops_now(self, tmp_path, monkeypatch, capsys):
        # A gateway predating ``restart_in_flight`` (this PR not yet pulled): no evidence of a
        # restart, so the safe reading stays refused-or-missed → immediate stop (SIGTERM; the bare
        # gateway is then waited for exactly as on the pre-PR path — no wait, no regression).
        stopped, entries = self._run_stop(tmp_path, monkeypatch, capsys, {
            "legacy": _ack(pausing=False, already_stopping=True),
        })
        assert stopped == [entries[sum(map(ord, "legacy"))]["pid"]]
        out = capsys.readouterr().out
        assert "already_stopping" in out, "the legacy gateway gets the refused-or-missed warning"

    def test_pausing_still_waits_and_never_stops(self, tmp_path, monkeypatch, capsys):
        stopped, entries = self._run_stop(tmp_path, monkeypatch, capsys, {
            "pausing": _ack(pausing=True, already_stopping=False, restart_in_flight=False),
        })
        assert stopped == []
        assert "Waiting up to" in capsys.readouterr().out

    def test_no_answer_takes_the_bare_fallback(self, tmp_path, monkeypatch, capsys):
        stopped, entries = self._run_stop(tmp_path, monkeypatch, capsys, {"noack": None})
        assert stopped == [entries[sum(map(ord, "noack"))]["pid"]]
        assert "already_stopping" not in capsys.readouterr().out
