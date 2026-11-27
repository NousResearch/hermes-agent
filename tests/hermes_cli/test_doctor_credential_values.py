"""`hermes doctor` names credential values that can never work (#160524):
a redacted display mask or a non-ASCII/lookalike char persisted into config.yaml
fails every request inside the HTTP header encoder, far from the file that caused it."""

from __future__ import annotations

import pytest


@pytest.fixture()
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  api_key: sk-a1B2c3D4a1B2c3D4a1B2c3D4a1B2c3D4\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _run():
    from hermes_cli.doctor_config import _check_credential_values

    return _check_credential_values(False)


def test_clean_config_passes(home):
    finding = _run()
    assert finding.issues == [] and finding.manual_issues == []


def test_non_ascii_credential_is_reported_with_path_and_fix(home, capsys):
    (home / "config.yaml").write_text(
        "model:\n  api_key: sk-proj-abc…xyz1234\n"
        "custom_providers:\n  - name: a\n    api_key: sk-proj-а1b2c3\n",
        encoding="utf-8",
    )
    finding = _run()
    assert any("model.api_key" in i for i in finding.issues)
    assert any("custom_providers" in i for i in finding.issues)
    assert any("hermes config set" in i for i in finding.issues)


def test_mask_shaped_value_is_reported(home):
    (home / "config.yaml").write_text(
        "model:\n  api_key: sk-pr...wxyz\n", encoding="utf-8"
    )
    finding = _run()
    assert any("model.api_key" in i for i in finding.issues)
