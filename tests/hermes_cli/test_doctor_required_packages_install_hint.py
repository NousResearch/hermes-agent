"""Regression tests for the Required Packages check in ``hermes doctor``.

Covers #89121: a missing package's install hint must spell out the *pip
distribution* name, not the import name, for packages whose install name diverges
(``dotenv`` -> ``python-dotenv``, ``yaml`` -> ``pyyaml``, ``telegram`` ->
``python-telegram-bot``, ``discord`` -> ``discord.py``); and an optional package
that is not installed must still surface an install command, not just a bare
"(optional, not installed)" note.
"""

import hermes_cli.doctor_platform as dp


def test_packages_table_uses_diverging_pip_specs():
    """Every import name whose pip distribution differs is mapped explicitly."""
    specs = {module: pip_spec for module, _name, pip_spec, _optional in dp._PACKAGES}
    assert specs["dotenv"] == "python-dotenv"
    assert specs["yaml"] == "pyyaml"
    assert specs["telegram"] == "python-telegram-bot"
    assert specs["discord"] == "discord.py"


def test_missing_required_package_hint_uses_pip_spec(monkeypatch):
    """A missing required dep is reported with ``<install cmd> <pip_spec>``."""
    monkeypatch.setattr(dp, "_python_install_cmd", lambda: "uv pip install")
    monkeypatch.setattr(
        dp, "_PACKAGES",
        (("hermes_missing_required_xyz", "Fake Required", "fake-required-dist", False),),
    )

    finding = dp._check_required_packages(False)

    assert finding.issues, "a missing required package should record an issue"
    fix = finding.issues[0]
    assert "uv pip install fake-required-dist" in fix
    assert "fake-required-dist" in fix
    # The import name must not leak into the hint.
    assert "hermes_missing_required_xyz" not in fix


def test_missing_optional_package_hint_uses_pip_spec(monkeypatch, capsys):
    """A missing optional dep still shows an install command with the pip spec."""
    monkeypatch.setattr(dp, "_python_install_cmd", lambda: "uv pip install")
    monkeypatch.setattr(
        dp, "_PACKAGES",
        (("hermes_missing_optional_xyz", "Fake Optional", "fake-optional-dist", True),),
    )

    finding = dp._check_required_packages(False)

    # Optional deps are warnings, not blocking issues.
    assert not finding.issues
    out = capsys.readouterr().out
    assert "install with: uv pip install fake-optional-dist" in out
    assert "hermes_missing_optional_xyz" not in out
