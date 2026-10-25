"""CLI HERMES_HOME invariant: an explicit ``-p <name>`` re-homes the process before profile state.

With ``HERMES_HOME`` unset in the launching shell, ``main._apply_profile_override`` must export the
profile home, so profile-scoped readers (config, state.db) and every child process agree with the
selected profile, while install-root readers still normalise back to the root. Companion of
tests/pm/test_runtime_environment_home.py, which covers the install-scoped PM bootstrap that runs
before this override.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_constants

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = "homelab-delegator"


@pytest.fixture
def hermes_root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    for name in (PROFILE, "alpha", "beta"):
        (root / "profiles" / name).mkdir(parents=True)
        (root / "profiles" / name / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "active_profile").write_text(f"{PROFILE}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_constants._get_platform_default_hermes_home", lambda: root)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    for var in ("HERMES_SUPERVISED_CHILD", "HERMES_S6_SUPERVISED_CHILD", "INVOCATION_ID",
                "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", False)
    return root


def _launch(monkeypatch, *argv: str) -> str | None:
    """What ``hermes <argv>`` does to the process home before argparse."""
    monkeypatch.setattr(sys, "argv", ["hermes", *argv])
    from hermes_cli.main import _apply_profile_override

    _apply_profile_override()
    return os.environ.get("HERMES_HOME")


def test_explicit_profile_with_unset_home_exports_the_profile_home(hermes_root, monkeypatch, capsys):
    profile = hermes_root / "profiles" / PROFILE
    assert _launch(monkeypatch, "-p", PROFILE, "config", "path") == str(profile)

    from hermes_cli.config import get_config_path
    from hermes_state import _default_db_path

    assert sys.argv == ["hermes", "config", "path"]
    assert hermes_constants.get_hermes_home() == profile
    assert get_config_path() == profile / "config.yaml"
    assert _default_db_path() == profile / "state.db"
    assert "HERMES_HOME fallback" not in capsys.readouterr().err


def test_child_process_inherits_the_selected_profile_home(hermes_root, monkeypatch, tmp_path):
    profile = hermes_root / "profiles" / PROFILE
    _launch(monkeypatch, "-p", PROFILE, "chat")

    child = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); import hermes_constants; "
         "print(hermes_constants.get_hermes_home())"],
        env={**os.environ, "HOME": str(tmp_path)},
        capture_output=True, text=True, check=True,
    )

    assert child.stdout.strip() == str(profile)
    assert "HERMES_HOME fallback" not in child.stderr


def test_default_profile_keeps_the_root(hermes_root, monkeypatch, capsys):
    (hermes_root / "active_profile").write_text("default\n", encoding="utf-8")

    assert _launch(monkeypatch, "chat") is None
    assert hermes_constants.get_hermes_home() == hermes_root
    assert _launch(monkeypatch, "-p", "default", "chat") == str(hermes_root)
    assert hermes_constants.get_hermes_home() == hermes_root
    assert "HERMES_HOME fallback" not in capsys.readouterr().err


@pytest.mark.parametrize("argv", [("chat",), ("-p", PROFILE, "chat")])
def test_profile_home_already_exported_is_kept_without_nesting(hermes_root, monkeypatch, argv):
    profile = hermes_root / "profiles" / PROFILE
    monkeypatch.setenv("HERMES_HOME", str(profile))

    assert _launch(monkeypatch, *argv) == str(profile)


def test_install_root_readers_normalise_a_profile_home(hermes_root, monkeypatch):
    from hermes_cli.profiles import profile_root_for_env_home

    _launch(monkeypatch, "-p", PROFILE, "chat")

    assert hermes_constants.get_default_hermes_root() == hermes_root
    assert profile_root_for_env_home(os.environ["HERMES_HOME"], Path("/unused")) == hermes_root


def test_explicit_switch_replaces_an_inherited_profile_home(hermes_root, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(hermes_root / "profiles" / "alpha"))

    assert _launch(monkeypatch, "-p", "beta", "chat") == str(hermes_root / "profiles" / "beta")


# What `hermes -p <name> gateway run` really does, in order, inside ONE process image: the PM
# bootstrap (pm.runtime_environment) runs first, then the CLI re-homes, then gateway modules
# resolve their profile-scoped paths. A subprocess, because the fallback warning latches per
# process and the gateway's own resolvers must be imported AFTER the re-home.
_GATEWAY_BOOTSTRAP = """
import sys
sys.path.insert(0, {repo!r})
sys.argv = ["hermes", "-p", {profile!r}, "gateway", "run"]
from pm.runtime import runtime_environment
bootstrap_home = runtime_environment()["HERMES_HOME"]
from hermes_cli.main import _apply_profile_override
_apply_profile_override()
from gateway.lifecycle_ledger import get_lifecycle_sentinel_path
print(bootstrap_home)
print(get_lifecycle_sentinel_path())
"""


def test_gateway_start_reaches_the_named_profile_without_the_default_fallback(hermes_root, tmp_path):
    profile = hermes_root / "profiles" / PROFILE
    child = subprocess.run(
        [sys.executable, "-c", _GATEWAY_BOOTSTRAP.format(repo=str(REPO_ROOT), profile=PROFILE)],
        env={k: v for k, v in os.environ.items() if k != "HERMES_HOME"} | {"HOME": str(tmp_path)},
        capture_output=True, text=True, check=True, cwd=str(tmp_path),
    )
    bootstrap_home, lifecycle = child.stdout.strip().split("\n")

    # The install-scoped bootstrap names the root it was launched from, never a profile it is
    # about to be re-homed to; the gateway's own profile-scoped state then lands in the profile.
    assert Path(bootstrap_home) == hermes_root
    assert Path(lifecycle) == profile / "state" / "gateway.lifecycle.json"
    assert "HERMES_HOME fallback" not in child.stderr
