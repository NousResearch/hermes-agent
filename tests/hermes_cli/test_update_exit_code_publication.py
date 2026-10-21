"""Every writer of ``.update_exit_code`` publishes it atomically.

The gateway /update watcher (gateway/run_notifications_update.py) polls ``.update_exit_code``
every 2 s from the gateway process and reads ``int(text or "1")``. A writer that truncates the
file in place left it empty for an instant; a poll landing there read exit 1, told the user the
update failed although it succeeded, and cleared the markers so no correction followed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.file_write_observer import observe_file_writes


def _watch(marker: Path) -> tuple[list[str], object]:
    """A gateway poll at every write instant: the marker's text whenever it exists."""
    seen: list[str] = []

    def poll() -> None:
        if marker.exists():
            seen.append(marker.read_text(encoding="utf-8-sig"))

    return seen, poll


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(path))
    return path


@pytest.fixture
def quiet_receipt(monkeypatch):
    from hermes_cli import update_receipt

    monkeypatch.setattr(update_receipt, "record_step", lambda *a, **k: None)
    monkeypatch.setattr(update_receipt, "finalize_pending_update_receipt", lambda *a, **k: None)


def test_fleet_writer_never_exposes_an_empty_exit_code(home):
    from hermes_cli.update_cmd_fleet import _write_gateway_update_exit_code

    marker = home / ".update_exit_code"
    seen, poll = _watch(marker)
    with observe_file_writes(poll):
        _write_gateway_update_exit_code(True)
    assert "" not in seen
    assert marker.read_text(encoding="utf-8").strip() == "0"


def test_completion_success_never_exposes_an_empty_exit_code(home):
    from hermes_cli.update_completion import _publish_gateway_success

    marker = home / ".update_exit_code"
    seen, poll = _watch(marker)
    with observe_file_writes(poll):
        _publish_gateway_success({"gateway_mode": True, "home": str(home)})
    assert "" not in seen
    assert marker.read_text(encoding="utf-8").strip() == "0"


def test_takeover_failure_never_exposes_an_empty_exit_code(home, tmp_path, quiet_receipt):
    from hermes_cli._update_takeover import _record_failure

    marker = home / ".update_exit_code"
    seen, poll = _watch(marker)
    with observe_file_writes(poll):
        _record_failure({"gateway_mode": True}, tmp_path / "result.json", 3, "boom")
    assert "" not in seen
    assert marker.read_text(encoding="utf-8").strip() == "3"


def test_takeover_owed_finish_never_exposes_an_empty_exit_code(home, tmp_path, quiet_receipt):
    from hermes_cli._update_takeover import _record_owed_finish

    marker = home / ".update_exit_code"
    seen, poll = _watch(marker)
    with observe_file_writes(poll):
        _record_owed_finish({"gateway_mode": True}, tmp_path / "result.json", "child missing")
    assert "" not in seen
    assert marker.read_text(encoding="utf-8").strip() == "0"


@pytest.mark.parametrize("recorder", ["failure", "owed_finish"])
def test_takeover_records_without_runtime_state_or_pm(home, tmp_path, quiet_receipt, monkeypatch, recorder):
    """The takeover bootstraps before the app is importable; its recorders must still write
    result.json and the gateway exit code when hermes_cli.runtime_state (→ pm) cannot be imported."""
    import sys

    from hermes_cli import _update_takeover

    for module in ("hermes_cli.runtime_state", "pm", "pm.filesystem"):
        monkeypatch.setitem(sys.modules, module, None)
    result = tmp_path / "result.json"
    if recorder == "failure":
        _update_takeover._record_failure({"gateway_mode": True}, result, 3, "boom")
    else:
        _update_takeover._record_owed_finish({"gateway_mode": True}, result, "child missing")

    assert result.exists()
    assert (home / ".update_exit_code").read_text(encoding="utf-8").strip() == ("3" if recorder == "failure" else "0")
