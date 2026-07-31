"""`setup_path()` must also install a `hermes-agent` launcher.

The project declares `hermes-agent = run_agent:main` in pyproject.toml. Fresh
installs need a stable PATH-level command next to `hermes`, and the installer
must replace stale symlinks without following them into the venv.
"""

import re
import stat
import subprocess
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parent.parent / "scripts" / "install.sh"

AGENT_BLOCK = re.compile(
    r'(    rm -f "\$command_link_dir/hermes-agent".*?'
    r'log_success "Installed hermes-agent launcher[^\n]*\n'
    r'    fi\n)',
    re.S,
)


def _extract_agent_shim_block() -> str:
    match = AGENT_BLOCK.search(INSTALL_SH.read_text(encoding="utf-8"))
    assert match, (
        "could not locate the hermes-agent launcher block in scripts/install.sh — "
        "if it was renamed, update this test with it"
    )
    return match.group(1)


def test_non_venv_install_writes_hermes_agent_launcher(tmp_path):
    command_link_dir = tmp_path / "local_bin"
    command_link_dir.mkdir()
    external_bin = tmp_path / "external_bin"
    external_bin.mkdir()
    console_script = external_bin / "hermes-agent"
    console_script.write_text("#!/bin/sh\n", encoding="utf-8")
    console_script.chmod(0o755)

    script = (
        "set -e\n"
        f"PATH={external_bin}:$PATH\n"
        f"command_link_dir={command_link_dir}\n"
        f"command_link_display_dir={command_link_dir}\n"
        "USE_VENV=false\n"
        "log_success(){ :; }\n"
        "log_warn(){ printf '%s\\n' \"$*\" >&2; }\n"
        + _extract_agent_shim_block()
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr

    shim = command_link_dir / "hermes-agent"
    assert shim.is_file()
    assert shim.stat().st_mode & stat.S_IXUSR, "launcher must be executable"
    text = shim.read_text(encoding="utf-8")
    assert "unset PYTHONPATH" in text
    assert "unset PYTHONHOME" in text
    assert f'exec "{console_script}" "$@"' in text


def test_hermes_agent_launcher_does_not_follow_a_symlink_into_the_venv(tmp_path):
    command_link_dir = tmp_path / "local_bin"
    command_link_dir.mkdir()
    console_script = tmp_path / "venv" / "bin" / "hermes-agent"
    console_script.parent.mkdir(parents=True)
    marker = "#!/usr/bin/env python\n# real console script\n"
    console_script.write_text(marker, encoding="utf-8")

    shim_path = command_link_dir / "hermes-agent"
    shim_path.symlink_to(console_script)
    assert shim_path.is_symlink()

    hermes_bin = tmp_path / "venv" / "bin" / "python"
    hermes_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    (install_dir / "run_agent.py").write_text("#!/bin/sh\n", encoding="utf-8")

    script = (
        "set -e\n"
        f"HERMES_BIN={hermes_bin}\n"
        f"INSTALL_DIR={install_dir}\n"
        f"command_link_dir={command_link_dir}\n"
        f"command_link_display_dir={command_link_dir}\n"
        "USE_VENV=true\n"
        "log_success(){ :; }\n"
        "log_warn(){ printf '%s\\n' \"$*\" >&2; }\n"
        + _extract_agent_shim_block()
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr

    assert console_script.read_text(encoding="utf-8") == marker
    assert not shim_path.is_symlink(), (
        "command_link_dir/hermes-agent must be replaced with a regular file"
    )