"""Doctor's channel-record hygiene check (_check_channel_record_hygiene).

Report-don't-delete: doctor names the stale config key; the user removes
it. The staleness shapes themselves are covered in
tests/hermes_cli/test_update_channel.py::TestDoctorStaleness — this file
covers the doctor-side wiring: warnings printed, exceptions swallowed
into a warning, silence on a clean config.
"""

from unittest.mock import patch

from hermes_cli.doctor import _check_channel_record_hygiene


def test_clean_config_prints_nothing(capsys):
    with (
        patch("hermes_cli.config.load_config", return_value={}),
    ):
        _check_channel_record_hygiene()
    assert capsys.readouterr().out == ""


def test_stale_records_warned_with_reason_texts(tmp_path, capsys, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    gone = tmp_path / "gone"
    live = tmp_path / "live"
    live.mkdir()
    config = {
        "update": {
            "installs": {
                # missing: nothing at the recorded path
                "deadbeefdeadbeef": {"path": str(gone), "channel": "main"},
                # replaced: path exists but keys to a different sha16
                "0" * 16: {"path": str(live), "channel": "stable"},
            }
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        _check_channel_record_hygiene()
    out = capsys.readouterr().out
    assert "Stale channel record: deadbeefdeadbeef" in out
    assert "safe to remove update.installs.deadbeefdeadbeef" in out
    assert f"Stale channel record: {'0' * 16}" in out
    assert "different install now" in out


def test_unreadable_config_degrades_to_a_warning(capsys):
    with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
        _check_channel_record_hygiene()
    out = capsys.readouterr().out
    assert "Channel-record hygiene unreadable" in out
    assert "boom" in out
