"""`hermes doctor` reports a plugin listed in both plugins.enabled and plugins.disabled.

Such a plugin loads nowhere because the disable wins. Older updates left plugins in that state when
they had to turn them off (#131216), and nothing else said so.
"""
from __future__ import annotations

from hermes_cli import doctor
from hermes_cli.doctor_state import _check_plugins_in_both_lists


def test_plugin_in_both_lists_is_an_issue_with_the_command_that_settles_it(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    (home / "config.yaml").write_text("plugins:\n  enabled: [kept, both]\n  disabled: [both, parked]\n", encoding="utf-8")

    finding = _check_plugins_in_both_lists(False)

    assert len(finding.manual_issues) == 1
    assert "'both'" in finding.manual_issues[0] and "hermes plugins enable both" in finding.manual_issues[0]
