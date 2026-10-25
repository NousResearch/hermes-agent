"""A hand-edited, non-mapping ``secrets:`` section must not take the CLI down.

``secrets: bitwarden`` / ``secrets: [bitwarden]`` used to raise out of ``load_hermes_dotenv`` (every
command exits 1), and ``secrets: {bitwarden: true}`` crashed ``hermes secrets bitwarden status``. These
tests write a real config.yaml into a temp home and check what the user gets: the command runs, the
``.env`` still loads, and Bitwarden reads as disabled. A valid config must still reach the source.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import env_loader

REPO = Path(__file__).resolve().parents[2]
MALFORMED = {
    "scalar": "secrets: bitwarden\n",
    "list": "secrets: [bitwarden]\n",
    "int": "secrets: 42\n",
    "null": "secrets:\n",
    "empty": "secrets: {}\n",
    "source-scalar": "secrets:\n  bitwarden: true\n",
}


@pytest.fixture(autouse=True)
def _clean_secret_state(monkeypatch):
    monkeypatch.delenv("BWS_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ZZ_NONMAP_DOTENV", raising=False)
    monkeypatch.delenv("ZZ_NONMAP_FROM_BW", raising=False)
    env_loader._SECRET_SOURCES.clear()
    env_loader._SECRET_SOURCE_VALUES_BY_HOME.clear()
    env_loader.reset_secret_source_cache()
    yield
    env_loader._SECRET_SOURCES.clear()
    env_loader._SECRET_SOURCE_VALUES_BY_HOME.clear()
    env_loader.reset_secret_source_cache()
    os.environ.pop("ZZ_NONMAP_DOTENV", None)
    os.environ.pop("ZZ_NONMAP_FROM_BW", None)


def _home(tmp_path: Path, config: str) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(config, encoding="utf-8")
    (home / ".env").write_text("ZZ_NONMAP_DOTENV=from-dotenv\n", encoding="utf-8")
    return home


@pytest.mark.parametrize("shape", list(MALFORMED))
def test_malformed_secrets_section_still_loads_dotenv(tmp_path, monkeypatch, shape):
    home = _home(tmp_path, MALFORMED[shape])
    monkeypatch.setenv("HERMES_HOME", str(home))

    loaded = env_loader.load_hermes_dotenv(hermes_home=home)

    assert home / ".env" in loaded
    assert os.environ["ZZ_NONMAP_DOTENV"] == "from-dotenv"


def test_section_cfg_reads_non_mappings_as_empty():
    from hermes_cli._secrets_common import section_cfg

    assert section_cfg({"secrets": "x"}, "bitwarden") == {}
    assert section_cfg({"secrets": ["bitwarden"]}, "bitwarden") == {}
    assert section_cfg({"secrets": {"bitwarden": True}}, "bitwarden") == {}
    assert section_cfg({}, "bitwarden") == {}
    valid = {"enabled": True, "project_id": "p"}
    assert section_cfg({"secrets": {"bitwarden": valid}}, "bitwarden") is valid


@pytest.mark.parametrize("shape", ["scalar", "source-scalar"])
def test_bitwarden_status_reports_disabled_for_a_malformed_section(tmp_path, monkeypatch, capsys, shape):
    from hermes_cli import secrets_cli

    monkeypatch.setenv("HERMES_HOME", str(_home(tmp_path, MALFORMED[shape])))
    monkeypatch.setenv("COLUMNS", "200")

    assert secrets_cli.cmd_status(argparse.Namespace()) == 0
    enabled_row = next(line for line in capsys.readouterr().out.splitlines() if "Enabled" in line)
    assert enabled_row.split("Enabled", 1)[1].split()[0] == "no"


def test_proxy_from_bitwarden_refuses_cleanly_on_a_malformed_section(tmp_path, monkeypatch):
    from rich.console import Console

    from hermes_cli import proxy_cli

    monkeypatch.setenv("HERMES_HOME", str(_home(tmp_path, MALFORMED["scalar"])))
    console = Console(record=True, width=200)

    assert proxy_cli._bitwarden_env_names(console) is None
    assert "secrets.bitwarden.enabled is false" in console.export_text()


def test_valid_bitwarden_section_still_reaches_the_source(tmp_path, monkeypatch):
    import agent.secret_sources.bitwarden as bw_module
    from agent.secret_sources import registry

    home = _home(tmp_path, "secrets:\n  bitwarden:\n    enabled: true\n    project_id: p\n"
                           "    access_token_env: BWS_ACCESS_TOKEN\n")
    (home / ".env").write_text("ZZ_NONMAP_DOTENV=from-dotenv\nBWS_ACCESS_TOKEN=bootstrap\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = []
    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))
    monkeypatch.setattr(bw_module, "fetch_bitwarden_secrets",
                        lambda **kw: calls.append(kw) or ({"ZZ_NONMAP_FROM_BW": "from-bitwarden"}, []))
    registry._reset_registry_for_tests()

    env_loader.load_hermes_dotenv(hermes_home=home)

    assert calls and calls[0]["access_token"] == "bootstrap"
    assert os.environ["ZZ_NONMAP_FROM_BW"] == "from-bitwarden"
    assert os.environ["ZZ_NONMAP_DOTENV"] == "from-dotenv"


@pytest.mark.parametrize("shape,argv", [
    ("scalar", ["status"]),
    ("list", ["config", "show"]),
    ("source-scalar", ["secrets", "bitwarden", "status"]),
])
def test_cli_commands_exit_cleanly_on_a_malformed_section(tmp_path, shape, argv):
    home = _home(tmp_path, MALFORMED[shape])
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_")}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), HERMES_HOME=str(home), COLUMNS="200",
               PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))

    proc = subprocess.run([sys.executable, "-m", "hermes_cli.main", *argv], env=env, cwd=tmp_path,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180, check=False)

    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "Traceback" not in output
    assert "has no attribute" not in output
