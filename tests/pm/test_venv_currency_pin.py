"""Venv currency must also track the project's pinned interpreter version."""

import pytest


def _write_base_venv(root, *, version):
    venv = root / ".venv"
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text(
        f"home = test\nversion = {version}\n", encoding="utf-8"
    )


def test_venv_currency_rejects_a_stale_interpreter_pin(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    _write_base_venv(root, version="3.13")
    (root / ".python-version").write_text("3.14\n", encoding="utf-8")

    from pm.install import _runtime_state_matches

    assert not _runtime_state_matches({"stamp": "pin"}, "pin", project_root=root)


def test_venv_currency_accepts_a_matching_interpreter_pin(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    _write_base_venv(root, version="3.14")
    (root / ".python-version").write_text("3.14.7\n", encoding="utf-8")

    from pm.install import _runtime_state_matches

    assert _runtime_state_matches({"stamp": "pin"}, "pin", project_root=root)


def test_venv_currency_without_a_pin_keeps_the_stamp_only_contract(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    _write_base_venv(root, version="3.11")

    from pm.install import _runtime_state_matches

    assert _runtime_state_matches({"stamp": "pin"}, "pin", project_root=root)
