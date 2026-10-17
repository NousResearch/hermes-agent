"""Read-only launcher operations the Desktop SSH helper advertises to Electron.

``inspect`` reports its capability set; ``list-profiles`` and ``install-id`` are pure reads — neither
may create a directory, a file or an identity. The install id is read from the *install root*, so a
HERMES_HOME pointing at ``<root>/profiles/<name>`` must still resolve to ``<root>``.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from hermes_cli import windows_ssh_runtime as rt

INSTALL_ID = "0123456789abcdef0123456789abcdef"
CAPABILITIES = ["launcher-ownership", "profile-inventory"]


def _install_home(monkeypatch, tmp_path):
    """A HERMES_HOME outside the platform native home, so custom-root rules apply."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    return tmp_path / "install"


# --- inspect -------------------------------------------------------------------------------------


def test_inspect_reports_capabilities(monkeypatch, tmp_path):
    exe = tmp_path / "hermes.exe"
    exe.write_text("", encoding="utf-8")

    def fake_run(argv, **_kwargs):
        text = "hermes 9.9.9" if "--version" in argv else "usage: --ssh-session-token-file --ssh-owner-nonce"
        return types.SimpleNamespace(returncode=0, stdout=text, stderr="")

    monkeypatch.setattr(rt, "subprocess", types.SimpleNamespace(run=fake_run))
    result = rt.inspect_hermes(str(exe))
    assert result["capabilities"] == CAPABILITIES
    # The pre-existing fields keep their exact shape.
    assert set(result) == {"path", "version", "supported", "capabilities"}
    assert result["path"] == os.path.abspath(str(exe))
    assert result["version"] == "hermes 9.9.9"
    assert result["supported"] is True


def test_inspect_capabilities_are_stable_without_support(monkeypatch, tmp_path):
    exe = tmp_path / "hermes.exe"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(rt, "subprocess", types.SimpleNamespace(
        run=lambda argv, **_kwargs: types.SimpleNamespace(returncode=1, stdout="", stderr="boom")))
    result = rt.inspect_hermes(str(exe))
    assert result["capabilities"] == CAPABILITIES
    assert result["supported"] is False
    assert result["version"] == ""


# --- list-profiles -------------------------------------------------------------------------------


def test_list_profiles_returns_default_and_live_skips_deleted(monkeypatch, tmp_path):
    home = _install_home(monkeypatch, tmp_path)
    (home / "profiles" / "team").mkdir(parents=True)
    (home / "profiles" / "team" / "config.yaml").write_text("name: team\n", encoding="utf-8")
    # A tombstoned shell: directory present, identity marker present, but deleted.
    (home / "profiles" / "gone").mkdir(parents=True)
    (home / "profiles" / "gone" / "config.yaml").write_text("name: gone\n", encoding="utf-8")
    (home / "profiles" / ".deleted").mkdir(parents=True)
    (home / "profiles" / ".deleted" / "gone").write_text("deleted\n", encoding="utf-8")
    # A marker-less ghost dir must not be listed either.
    (home / "profiles" / "ghost").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert rt.list_profiles_op() == {"profiles": ["default", "team"]}


def test_list_profiles_creates_nothing(monkeypatch, tmp_path):
    home = _install_home(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert rt.list_profiles_op() == {"profiles": ["default"]}
    assert not home.exists()


# --- install-id ----------------------------------------------------------------------------------


def test_install_id_reads_valid_file_and_lowercases(monkeypatch, tmp_path):
    home = _install_home(monkeypatch, tmp_path)
    home.mkdir()
    (home / "install_id").write_text("0123456789ABCDEF0123456789ABCDEF\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert rt.install_id_op() == {"installId": INSTALL_ID}


def test_install_id_missing_file_creates_nothing(monkeypatch, tmp_path):
    home = _install_home(monkeypatch, tmp_path)
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert rt.install_id_op() == {"installId": None}
    assert not (home / "install_id").exists()
    assert not (home / ".install_id.lock").exists()
    assert sorted(entry.name for entry in home.iterdir()) == []


def test_install_id_missing_root_creates_nothing(monkeypatch, tmp_path):
    home = _install_home(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert rt.install_id_op() == {"installId": None}
    assert not home.exists()


@pytest.mark.parametrize("content", ["", "   ", "not-a-hex-id", "0" * 31, "0" * 33, "g" * 32, "0123456789abcdef0123456789abcde"])
def test_install_id_rejects_invalid_content(monkeypatch, tmp_path, content):
    home = _install_home(monkeypatch, tmp_path)
    home.mkdir()
    (home / "install_id").write_text(content, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert rt.install_id_op() == {"installId": None}
    # An invalid file is reported, never rewritten in place.
    assert (home / "install_id").read_text(encoding="utf-8") == content


def test_install_id_resolves_install_root_from_profile_home(monkeypatch, tmp_path):
    home = _install_home(monkeypatch, tmp_path)
    (home / "profiles" / "team").mkdir(parents=True)
    (home / "install_id").write_text(INSTALL_ID + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "team"))
    assert rt.install_id_op() == {"installId": INSTALL_ID}
    assert not (home / "profiles" / "team" / "install_id").exists()


# --- dispatch wiring -----------------------------------------------------------------------------


@pytest.mark.parametrize("operation, expected", [
    ("list-profiles", {"profiles": ["default"]}),
    ("install-id", {"installId": None}),
])
def test_new_operations_dispatch_without_args(monkeypatch, tmp_path, operation, expected):
    monkeypatch.setenv("HERMES_HOME", str(_install_home(monkeypatch, tmp_path)))
    assert rt.dispatch([operation]) == expected


@pytest.mark.parametrize("operation", ["list-profiles", "install-id"])
def test_new_operations_reject_extra_args(monkeypatch, tmp_path, operation):
    monkeypatch.setenv("HERMES_HOME", str(_install_home(monkeypatch, tmp_path)))
    with pytest.raises(ValueError):
        rt.dispatch([operation, "extra"])


def test_install_id_cli_emits_one_json_line(monkeypatch, tmp_path, capsys):
    home = _install_home(monkeypatch, tmp_path)
    home.mkdir()
    (home / "install_id").write_text(INSTALL_ID, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sys, "argv", ["runtime", "install-id"])
    rt.main()
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {"installId": INSTALL_ID}
    assert captured.out.strip().count("\n") == 0
