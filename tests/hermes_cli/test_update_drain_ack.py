"""The updater's drain ACK is the gateway's ``pausing`` — never ``already_stopping`` (#135878).

``pause_gateway_for_update`` answers ``{"pausing": accepted, "already_stopping": not accepted}``:
a gateway that refused (or never received) the drain request still reports ``already_stopping``
even though it keeps running its in-flight turns. Treating that flag as a drain ACK parks
``hermes update`` in its full drain budget waiting on a gateway that will not exit on its own
(~31 min on macOS per update); it must take the immediate ``_stop_at_once`` escalation instead,
with one observable line saying why.
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
        ack = _ack(pausing=True, already_stopping=False)
        monkeypatch.setattr(control_socket, "pause_gateway_for_update", lambda home, **kw: ack)
        assert m._ask_to_drain(_entry(tmp_path)) == ack

    def test_already_stopping_is_not_the_ack(self, tmp_path, monkeypatch):
        ack = _ack(pausing=False, already_stopping=True)
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


class TestAlreadyStoppingEscalatesAtOnce:
    def test_an_already_stopping_gateway_is_stopped_now_and_not_waited_on(
        self, tmp_path, monkeypatch, capsys
    ):
        pausing_home = tmp_path / "pausing"
        already_home = tmp_path / "already"
        pausing_home.mkdir()
        already_home.mkdir()
        entries = [
            {"pid": 101, "home": str(pausing_home)},
            {"pid": 202, "home": str(already_home)},
        ]
        answers = {
            "pausing": _ack(pausing=True, already_stopping=False),
            "already": _ack(pausing=False, already_stopping=True),
        }
        monkeypatch.setattr(
            control_socket, "pause_gateway_for_update", lambda home, **kw: answers[Path(home).name]
        )
        import hermes_cli.update_pause_record as pause_record_module

        monkeypatch.setattr(pause_record_module, "mark_stop_sent", lambda token, pid: None)
        monkeypatch.setattr(m, "_drain_budget", lambda: 1885.0)
        monkeypatch.setattr(m, "_alive", lambda pid, ct=None: False)
        stopped: list[int] = []
        monkeypatch.setattr(m, "_stop_at_once", lambda entry: stopped.append(entry["pid"]))

        m._stop_gateways({}, entries)

        assert stopped == [202], (
            "the gateway that answered already_stopping must take the immediate stop; the one "
            "that accepted the drain must keep its wait"
        )
        out = capsys.readouterr().out
        assert "already_stopping" in out and "gateway PID 202" in out, (
            "the misclassified state must be observable in one line, not a silent 31 min wait"
        )
        assert "Waiting up to" in out, "a genuine drain ACK still waits within the budget"
