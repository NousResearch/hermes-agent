"""Regression tests for Desktop installer SQLite runtime preservation."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_function(name: str) -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{.*?^\}}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"{name}() not found in install.sh"
    return match.group(0)


def test_setup_venv_preserves_existing_python_311_runtime(tmp_path: Path) -> None:
    """A Desktop update must not delete a usable 3.11 venv before safety repair."""
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/bash\necho 3.11\n", encoding="utf-8")
    venv_python.chmod(0o755)
    sentinel = tmp_path / "venv" / "keep-me"
    sentinel.write_text("preserved", encoding="utf-8")

    uv_log = tmp_path / "uv.log"
    uv_stub = tmp_path / "uv-stub"
    uv_stub.write_text(
        "#!/bin/bash\n"
        f"echo called >> {uv_log!s}\n"
        "exit 97\n",
        encoding="utf-8",
    )
    uv_stub.chmod(0o755)

    harness = f"""
set -e
USE_VENV=true
DISTRO=macos
INSTALL_DIR={str(tmp_path)!r}
PYTHON_VERSION=3.11
UV_CMD={str(uv_stub)!r}
log_info() {{ :; }}
log_success() {{ :; }}
{_extract_function('setup_venv')}
cd "$INSTALL_DIR"
setup_venv
printf 'UV_PYTHON=%s\n' "$UV_PYTHON"
"""
    proc = subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )

    assert proc.returncode == 0, proc.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserved"
    assert not uv_log.exists(), "uv must not recreate an existing Python 3.11 venv"
    assert f"UV_PYTHON={venv_python}" in proc.stdout


def test_desktop_python_deps_stage_repairs_managed_runtime() -> None:
    """Every staged Desktop install must run the managed SQLite repair gate."""
    harness = f"""
set -e
EVENTS=$(mktemp)
trap 'rm -f "$EVENTS"' EXIT
log_event() {{ echo "$1" >> "$EVENTS"; }}
detect_os() {{ :; }}
resolve_install_layout() {{ :; }}
require_install_dir() {{ :; }}
install_uv() {{ :; }}
check_python() {{ :; }}
install_deps() {{ log_event deps; }}
repair_managed_runtime() {{ log_event repair; }}
{_extract_function('run_stage_body')}
run_stage_body python-deps
tr '\n' ' ' < "$EVENTS"
"""
    proc = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "deps repair"


def test_setup_venv_recreates_unprobeable_runtime(tmp_path: Path) -> None:
    """An interpreter that cannot report its version must not be preserved."""
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text(
        "#!/bin/bash\necho 3.11\nexit 42\n",
        encoding="utf-8",
    )
    venv_python.chmod(0o755)
    sentinel = tmp_path / "venv" / "remove-me"
    sentinel.write_text("stale", encoding="utf-8")

    uv_log = tmp_path / "uv.log"
    uv_stub = tmp_path / "uv-stub"
    uv_stub.write_text(
        "#!/bin/bash\n"
        f"echo called >> {uv_log!s}\n"
        'mkdir -p "$2/bin"\n'
        'printf \'#!/bin/bash\\necho 3.11\\n\' > "$2/bin/python"\n'
        'chmod +x "$2/bin/python"\n',
        encoding="utf-8",
    )
    uv_stub.chmod(0o755)

    harness = f"""
set -e
USE_VENV=true
DISTRO=macos
INSTALL_DIR={str(tmp_path)!r}
PYTHON_VERSION=3.11
UV_CMD={str(uv_stub)!r}
log_info() {{ :; }}
log_success() {{ :; }}
{_extract_function('setup_venv')}
cd "$INSTALL_DIR"
setup_venv
printf 'UV_PYTHON=%s\n' "$UV_PYTHON"
"""
    proc = subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )

    assert proc.returncode == 0, proc.stderr
    assert not sentinel.exists()
    assert uv_log.exists(), "uv must recreate an unprobeable venv"
    assert f"UV_PYTHON={venv_python}" in proc.stdout


def test_runtime_repair_failure_reaches_stage_protocol() -> None:
    """The Desktop stage protocol must emit failure when safety is unknown."""
    harness = f"""
set -e
JSON_OUTPUT=true
NON_INTERACTIVE=false
stage_needs_user_input() {{ return 1; }}
emit_stage_json() {{ printf '%s|%s|%s|%s\n' "$1" "$2" "$3" "${{4:-}}"; }}
detect_os() {{ :; }}
resolve_install_layout() {{ :; }}
require_install_dir() {{ :; }}
install_uv() {{ :; }}
check_python() {{ :; }}
install_deps() {{ :; }}
repair_managed_runtime() {{ return 23; }}
{_extract_function('run_stage_body')}
{_extract_function('run_stage_protocol')}
run_stage_protocol python-deps
"""
    proc = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert proc.returncode == 23
    assert proc.stdout.strip() == "python-deps|false|false|exit code 23"


def test_runtime_repair_failure_stops_monolithic_install() -> None:
    """The non-staged installer must not continue after repair uncertainty."""
    noops = " ".join(
        [
            "print_banner",
            "detect_os",
            "resolve_install_layout",
            "install_uv",
            "check_python",
            "check_git",
            "check_node",
            "check_network_prerequisites",
            "install_system_packages",
            "clone_repo",
            "setup_venv",
            "install_deps",
            "install_node_deps",
            "setup_path",
            "copy_config_templates",
            "run_setup_wizard",
            "maybe_start_gateway",
            "install_desktop",
            "print_success",
        ]
    )
    harness = f"""
set -e
INSTALL_DIR=$(mktemp -d)
trap 'rm -rf "$INSTALL_DIR"' EXIT
INCLUDE_DESKTOP=false
for name in {noops}; do eval "$name() {{ :; }}"; done
repair_managed_runtime() {{ return 31; }}
{_extract_function('main')}
main
"""
    proc = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert proc.returncode == 31


def test_runtime_repair_bypasses_unmanaged_and_termux_installs(tmp_path: Path) -> None:
    """The repair gate applies only to managed non-Termux venv installs."""
    function = _extract_function("repair_managed_runtime")
    for use_venv, distro in (("false", "macos"), ("true", "termux")):
        harness = f"""
set -e
USE_VENV={use_venv}
DISTRO={distro}
INSTALL_DIR={str(tmp_path)!r}
UV_CMD=uv
log_error() {{ :; }}
log_info() {{ :; }}
log_success() {{ :; }}
{function}
repair_managed_runtime
"""
        proc = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert proc.returncode == 0, proc.stderr
