"""Regression coverage for Windows updater self-locking native dependencies.

External secret backends are useful during normal Hermes startup, but the
updater must not load them before replacing packages in its own environment.
On Windows, importing Bitwarden's ``cryptography`` dependency maps
``_rust.pyd`` into the updater process and prevents ``uv`` from replacing it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import env_loader


REPO_ROOT = Path(__file__).resolve().parents[2]


def _probe_startup_modules(
    tmp_path: Path, argv: list[str], *, run_main: bool = False
) -> set[str]:
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        """\
secrets:
  bitwarden:
    enabled: true
    project_id: test-project
""",
        encoding="utf-8",
    )

    dispatch = (
        "from hermes_cli import update_cmd\n"
        "update_cmd._cmd_update_check = lambda **kwargs: 0\n"
        "hermes_main.main()\n"
        if run_main
        else ""
    )
    probe = (
        "import json, sys\n"
        f"sys.argv = {argv!r}\n"
        "import hermes_cli.main as hermes_main\n"
        f"{dispatch}"
        "print('LOADED_MODULES=' + json.dumps(sorted(sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        env={**os.environ, "HERMES_HOME": str(home), "BWS_ACCESS_TOKEN": ""},
    )
    assert result.returncode == 0, result.stderr
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("LOADED_MODULES=")
    )
    return set(json.loads(line.removeprefix("LOADED_MODULES=")))




def test_complete_update_dispatch_does_not_import_cryptography(tmp_path):
    """Building every CLI parser used to re-import Bitwarden via secrets_cli."""
    loaded = _probe_startup_modules(
        tmp_path,
        ["hermes", "update", "--check"],
        run_main=True,
    )

    assert "agent.secret_sources.bitwarden" not in loaded
    assert not any(
        name == "cryptography" or name.startswith("cryptography.") for name in loaded
    )


def test_normal_startup_still_loads_enabled_external_secret_source(tmp_path):
    loaded = _probe_startup_modules(tmp_path, ["hermes", "chat"])

    assert "agent.secret_sources.bitwarden" in loaded


@pytest.mark.parametrize("external_secrets", [True, False])
def test_dotenv_loading_is_preserved_when_external_secrets_are_skipped(
    tmp_path, monkeypatch, external_secrets
):
    home = tmp_path / "hermes-home"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("UPDATE_TEST_VALUE=from-dotenv\n", encoding="utf-8")
    applied = []
    monkeypatch.delenv("UPDATE_TEST_VALUE", raising=False)
    monkeypatch.setattr(
        env_loader,
        "_apply_external_secret_sources",
        lambda path: applied.append(path),
    )

    loaded = env_loader.load_hermes_dotenv(
        hermes_home=home,
        load_external_secrets=external_secrets,
    )

    assert loaded == [env_file]
    assert os.environ["UPDATE_TEST_VALUE"] == "from-dotenv"
    assert applied == ([home] if external_secrets else [])


@pytest.mark.platforms("posix")  # the 'command' secret source is POSIX-only
def test_update_probe_children_skip_external_secret_sources(tmp_path):
    """The critical-module import probe imports ``run_agent``, whose dotenv load must not run a
    configured secret helper: a slow helper (op/bws/command, 120s budget) inside the 120s probe
    surfaced as ``timed out before reporting import health`` on a healthy install (#110823)."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    hit = tmp_path / "helper_hit"
    (home / "config.yaml").write_text(
        f"secrets:\n  command:\n    enabled: true\n    command: 'touch {hit}'\n", encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv = ['hermes', 'update']\n"
         "from hermes_cli.update_cmd_validation import _validate_critical_modules_import\n"
         "print('PROBE=' + repr(_validate_critical_modules_import(__import__('os').getcwd())))"],
        capture_output=True, text=True, timeout=180, cwd=REPO_ROOT,
        env={**os.environ, "HERMES_HOME": str(home)},
    )
    assert result.returncode == 0, result.stderr
    assert "PROBE=(True, None, None)" in result.stdout, result.stdout + result.stderr
    assert not hit.exists(), "the import probe resolved external secret sources"
