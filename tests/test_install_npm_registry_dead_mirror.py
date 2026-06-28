"""Regression coverage for #42972: dead npm registry mirrors break desktop install.

A user-level .npmrc can point npm at the long-dead CouchDB replica hosts
``replicate.npmjs.com`` / ``skimdb.npmjs.com``. npm then rewrites lockfile
tarball fetches to that registry and the desktop stage fails with E404 while
fetching packages such as ``globals``.

The installers should keep the fix scoped to Hermes' own npm invocations:
detect only those known-dead registry hosts, pass an explicit public npm
registry for bootstrap installs, and print a remediation hint when npm output
contains that signature.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"

DEAD_HOSTS = ("replicate.npmjs.com", "skimdb.npmjs.com")
PUBLIC_REGISTRY = "https://registry.npmjs.org/"
PRIVATE_OR_NORMAL_REGISTRIES = (
    "https://registry.npmjs.org/",
    "https://registry.corp.internal/npm/replicate.npmjs.com/",
    "https://registry.corp.internal/npm/@replicate.npmjs.com/",
    "https://registry.corp.internal/@skimdb.npmjs.com/",
    "https://replicate.npmjs.com.corp.internal/",
    "https://notreplicate.npmjs.com/",
)


def _extract_install_sh_helpers() -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    start = text.index("log_info() {")
    end = text.index("json_escape() {")
    return text[start:end]


def _run_bash_registry_helper(registry: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fake_npm = tmp_path / "npm"
        fake_npm.write_text(
            "#!/usr/bin/env bash\n"
            "if [ \"$1\" = \"config\" ] && [ \"$2\" = \"get\" ] && [ \"$3\" = \"registry\" ]; then\n"
            f"  printf '%s\\n' {shlex.quote(registry)}\n"
            "  exit 0\n"
            "fi\n"
            "exit 1\n",
            encoding="utf-8",
        )
        fake_npm.chmod(0o755)
        script = f"""
        set -euo pipefail
        RED= GREEN= YELLOW= BLUE= MAGENTA= CYAN= NC= BOLD=
        source "$HELPERS"
        npm_registry_args {shlex.quote(str(fake_npm))}
        """
        env = os.environ.copy()
        helpers_path = tmp_path / "install-sh-registry-helpers.sh"
        helpers_path.write_text(_extract_install_sh_helpers(), encoding="utf-8")
        env["HELPERS"] = str(helpers_path)
        result = subprocess.run(
            ["bash", "-c", script],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return result.stdout.strip()


def test_install_ps1_overrides_known_dead_npm_registry_for_bootstrap_invocations() -> None:
    text = INSTALL_PS1.read_text(encoding="utf-8")

    for host in DEAD_HOSTS:
        assert host in text
    assert PUBLIC_REGISTRY in text
    assert "Get-NpmRegistryHost" in text
    assert "Test-NpmDeadRegistry" in text
    assert "Get-NpmRegistryArgs" in text
    assert "Show-NpmRegistryHint" in text

    # Every PowerShell npm install/ci bootstrap path should receive the same
    # scoped args so the desktop stage and sibling npm stages behave alike.
    assert "& $npm install @npmRegistryArgs -g --prefix" in text
    assert "& $npmPath install @npmRegistryArgs --silent" in text
    assert "& $npmExe ci @npmRegistryArgs" in text
    assert "& $npmExe install @npmRegistryArgs" in text

    # The override must be keyed by exact registry host, not substring, so
    # private/corporate mirrors are not bypassed to public npm.
    assert "$NpmDeadRegistryHosts -contains $registryHost" in text
    assert "$registry -match $NpmDeadRegistryPattern" not in text


def test_install_sh_overrides_known_dead_npm_registry_for_bootstrap_invocations() -> None:
    text = INSTALL_SH.read_text(encoding="utf-8")

    for host in DEAD_HOSTS:
        assert host in text
    assert PUBLIC_REGISTRY in text
    assert "npm_registry_host" in text
    assert "is_dead_npm_registry" in text
    assert "npm_registry_args" in text
    assert "show_npm_registry_hint" in text

    # Shell arrays keep the override correctly quoted and empty for normal
    # registries. Cover browser tools, agent-browser, and desktop workspace.
    assert 'npm install "${npm_registry_args[@]}" --silent' in text
    assert '"$npm_bin" install "${npm_registry_args[@]}" -g --prefix' in text
    assert 'npm ci "${npm_registry_args[@]}"' in text
    assert 'npm install "${npm_registry_args[@]}"' in text


def test_install_sh_registry_override_is_exact_host_only() -> None:
    for host in DEAD_HOSTS:
        assert _run_bash_registry_helper(f"https://{host}/") == f"--registry={PUBLIC_REGISTRY}"

    for registry in PRIVATE_OR_NORMAL_REGISTRIES:
        assert _run_bash_registry_helper(registry) == ""
