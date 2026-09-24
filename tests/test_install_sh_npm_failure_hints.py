"""Tests for install.sh actionable npm failure hints and the npm cache probe.

The installer used to abort with a bare "npm install failed or timed out" and
(at best) a captured snippet that --silent had emptied. npm_failure_hints
classifies the recurring failure signatures from real install reports (cache
EACCES, engine mismatch, native build, network, full disk) and prints the fix;
npm_debug_log_hint points at npm's own debug log, which npm writes even when
console output is captured; prepare_npm_cache probes the cache up front and
falls back to a Hermes-owned cache instead of dying with EACCES deep inside
_cacache. Regression for the diagnosis half of #87093.
"""

import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

STUBS = """\
log_warn() { echo "warn: $*"; }
log_info() { echo "info: $*"; }
log_error() { echo "error: $*"; }
"""

CACHE_FAILURE = """\
npm error code EACCES
npm error syscall mkdir
npm error path /home/u/.npm/_cacache/content-v2/sha512/19/01
npm error errno -13
"""

ENGINE_FAILURE = "npm error code EBADENGINE\nnpm error Unsupported engine (node: >=99)\n"

GYP_FAILURE = "npm error gyp ERR! stack Error: 404 response downloading headers\n"

NETWORK_FAILURE = "npm error code ETIMEDOUT\nnpm error network request to https://registry.npmjs.org timed out\n"

DISK_FAILURE = "npm error code ENOSPC\nnpm error ENOSPC: no space left on device\n"

UNRELATED_FAILURE = "npm error code E404\nnpm error 404 Not Found - GET https://registry.npmjs.org/does-not-exist\n"


def _function_body(name: str) -> str:
    body = re.search(
        rf"^{name}\(\) \{{.*?^\}}", INSTALL_SH.read_text(), re.S | re.M
    )
    assert body, f"{name} not found in install.sh"
    return body.group(0)


def _run_function(
    tmp_path: Path, name: str, log_text: str | None = None, env: dict | None = None
) -> subprocess.CompletedProcess:
    log = tmp_path / "npm.log"
    if log_text is not None:
        log.write_text(log_text)
    script = tmp_path / "harness.sh"
    script.write_text(textwrap.dedent(STUBS) + _function_body(name)
                      + f'\n{name} "{log}"\n')
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env
    )


def test_cache_permission_hint_names_the_chown_fix(tmp_path: Path) -> None:
    result = _run_function(tmp_path, "npm_failure_hints", CACHE_FAILURE)
    assert result.returncode == 0
    assert "cache permissions problem" in result.stdout
    assert "chown -R" in result.stdout


def test_engine_hint_suggests_supported_node(tmp_path: Path) -> None:
    result = _run_function(tmp_path, "npm_failure_hints", ENGINE_FAILURE)
    assert result.returncode == 0
    assert "Node.js version mismatch" in result.stdout
    assert "22.22+, 24.11+, or 26+" in result.stdout


def test_native_build_hint_names_build_tools(tmp_path: Path) -> None:
    result = _run_function(tmp_path, "npm_failure_hints", GYP_FAILURE)
    assert result.returncode == 0
    assert "native-module build" in result.stdout
    assert "build-essential" in result.stdout


def test_network_hint_mentions_proxy(tmp_path: Path) -> None:
    result = _run_function(tmp_path, "npm_failure_hints", NETWORK_FAILURE)
    assert result.returncode == 0
    assert "network problem" in result.stdout
    assert "https-proxy" in result.stdout


def test_disk_hint(tmp_path: Path) -> None:
    result = _run_function(tmp_path, "npm_failure_hints", DISK_FAILURE)
    assert result.returncode == 0
    assert "full disk" in result.stdout


def test_hints_stay_quiet_on_unrelated_failure(tmp_path: Path) -> None:
    result = _run_function(tmp_path, "npm_failure_hints", UNRELATED_FAILURE)
    assert result.returncode == 1
    assert result.stdout == ""


def test_hints_stay_quiet_on_empty_log(tmp_path: Path) -> None:
    result = _run_function(tmp_path, "npm_failure_hints", "")
    assert result.returncode == 1
    assert result.stdout == ""


def test_debug_log_hint_points_at_newest_npm_log(tmp_path: Path) -> None:
    logs = tmp_path / "npm" / "_logs"
    logs.mkdir(parents=True)
    (logs / "2026-09-16T00_00_00_000Z-debug-0.log").write_text("full chain")
    env = {**os.environ, "npm_config_cache": str(tmp_path / "npm")}
    result = _run_function(tmp_path, "npm_debug_log_hint", env=env)
    assert result.returncode == 0
    assert str(logs / "2026-09-16T00_00_00_000Z-debug-0.log") in result.stdout


def test_debug_log_hint_silent_without_logs(tmp_path: Path) -> None:
    env = {**os.environ, "npm_config_cache": str(tmp_path / "npm")}
    result = _run_function(tmp_path, "npm_debug_log_hint", env=env)
    assert result.returncode == 1
    assert result.stdout == ""


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permissions")
def test_unwritable_cache_falls_back_to_hermes_owned(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked-cache"
    blocked.mkdir()
    blocked.chmod(0o500)  # mkdir -p passes, the write probe must fail
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    env = {
        **os.environ,
        "npm_config_cache": str(blocked),
        "HERMES_HOME": str(hermes_home),
    }
    script = tmp_path / "probe.sh"
    script.write_text(textwrap.dedent(STUBS) + _function_body("prepare_npm_cache")
                      + '\nprepare_npm_cache\necho "cache=$npm_config_cache"\n')
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
    blocked.chmod(0o700)
    assert result.returncode == 0
    assert "not writable" in result.stdout
    assert "chown -R" in result.stdout
    assert result.stdout.rstrip().endswith(f"cache={hermes_home}/npm-cache")
    assert (hermes_home / "npm-cache").is_dir()


def test_healthy_cache_needs_no_fallback(tmp_path: Path) -> None:
    env = {**os.environ, "npm_config_cache": str(tmp_path / "fine-cache"),
           "HERMES_HOME": str(tmp_path / "hermes-home")}
    result = _run_function(tmp_path, "prepare_npm_cache", env=env)
    assert result.returncode == 0
    assert result.stdout == ""


def test_failure_branches_wire_all_hints() -> None:
    """Every install-blocking npm failure path prints output, class hints, and
    the debug-log pointer; the node-deps stage probes the cache first."""
    body = _function_body("install_node_deps")
    assert "prepare_npm_cache || return 1" in body
    assert body.count("npm_cert_hint ") >= 2
    assert body.count("npm_failure_hints ") >= 2
    assert body.count("npm_debug_log_hint ") >= 2
