"""Regression test for #102945: ``hermes doctor`` must surface a config.yaml
that fails YAML parsing.

A corrupt config.yaml silently falls back to DEFAULT_CONFIG — every user
override (providers, fallback chain, model settings) is ignored while chat
keeps working, and the loader's one-line stderr warning is gone by the time
anyone investigates. ``hermes doctor`` is the persistent health signal for
that state.
"""

from hermes_cli import doctor_config as doctor_config_mod
from hermes_cli.doctor_report import Finding


def _run_check(monkeypatch, tmp_path, config_text: str) -> Finding:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    # _check_config_file imports HERMES_HOME from hermes_cli.doctor at call time.
    monkeypatch.setattr("hermes_cli.doctor.HERMES_HOME", tmp_path)
    return doctor_config_mod._check_config_file(False)


def test_corrupt_yaml_is_reported_as_running_on_defaults(monkeypatch, tmp_path):
    corrupt = (
        "model:\n"
        "  default: gpt-x\n"
        "  broken_indent\n"
        "    did_not_find_expected_key: true\n"
    )
    f = _run_check(monkeypatch, tmp_path, corrupt)
    assert any("Fix the YAML" in issue for issue in f.issues), (
        "a config.yaml that fails YAML parsing must be a doctor finding — "
        "the silent defaults fallback ignores every user override (#102945)"
    )


def test_valid_yaml_produces_no_parse_finding(monkeypatch, tmp_path):
    f = _run_check(monkeypatch, tmp_path, "model:\n  default: gpt-x\n")
    assert not [i for i in f.issues if "failed to parse" in i]
