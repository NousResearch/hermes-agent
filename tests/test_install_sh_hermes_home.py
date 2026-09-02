"""--hermes-home must reach skills_sync and the generated launcher (#89231).

The posix installer assigns HERMES_HOME for its own writes but historically
did not export it. Python children and the public `hermes` shim then silently
fell back to ~/.hermes. These tests drive the real install.sh helpers in a
stubbed shell — they do not snapshot script text.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_function(source: str, name: str) -> str:
    """Extract a top-level ``name() { ... }`` body, skipping heredocs.

    A naive ``.*?^}`` cut stops at the first column-0 ``}`` — including one
    that only exists inside a heredoc. Count braces outside heredocs and
    ignore ``${...}`` / quoted spans so a later install.sh edit does not
    silently truncate the helper.
    """
    lines = source.splitlines(keepends=True)
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if re.match(rf"^{re.escape(name)}\(\) \{{", line)
        ),
        None,
    )
    assert start is not None, f"could not extract {name}() from install.sh"
    depth = 0
    heredoc_end: str | None = None
    out: list[str] = []
    for line in lines[start:]:
        out.append(line)
        raw = line.rstrip("\n")
        if heredoc_end is not None:
            if raw == heredoc_end:
                heredoc_end = None
            continue
        heredoc = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        if heredoc and not line.lstrip().startswith("#"):
            heredoc_end = heredoc.group(1)
        code = re.sub(r"\$\{[^{}]*\}", "", line)
        code = re.sub(r"'[^']*'", "", code)
        code = re.sub(r'"[^"]*"', "", code)
        depth += code.count("{") - code.count("}")
        if heredoc_end is None and depth <= 0 and len(out) > 1:
            break
    assert "".join(out).rstrip().endswith("}"), f"unbalanced extract of {name}()"
    return "".join(out)


def test_extract_function_skips_braces_inside_heredoc() -> None:
    """A column-0 `}` in a heredoc must not truncate the extracted function."""
    src = (
        "other() {\n"
        "  true\n"
        "}\n"
        "sample() {\n"
        "  cat <<EOF\n"
        "}\n"
        "EOF\n"
        "  echo done\n"
        "}\n"
    )
    body = _extract_function(src, "sample")
    assert "echo done" in body
    assert body.rstrip().endswith("}")


def _python_probe_script() -> str:
    return (
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "out = os.environ.get('PROBE_OUT')\n"
        "if out:\n"
        "    pathlib.Path(out).write_text(os.environ.get('HERMES_HOME', '<unset>'))\n"
        "    sys.exit(0)\n"
        "sys.stdout.write(os.environ.get('HERMES_HOME', '<unset>'))\n"
    )


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run_copy_config_templates(tmp_path: Path) -> dict[str, str]:
    """Run copy_config_templates the way install.sh does: assign, do not export."""
    source = INSTALL_SH.read_text()
    body = _extract_function(source, "copy_config_templates")
    home = tmp_path / "home"
    hermes_home = tmp_path / "custom-home"
    install_dir = tmp_path / "install"
    probe_out = tmp_path / "child-hermes-home"
    log_out = tmp_path / "install.log"
    stub_python = install_dir / "venv" / "bin" / "python"
    _write_executable(stub_python, _python_probe_script())
    (install_dir / "tools").mkdir(parents=True)
    (install_dir / "tools" / "skills_sync.py").write_text("# probe target\n")

    harness = f"""
set -eu
HOME={str(home)!r}
HERMES_HOME={str(hermes_home)!r}
INSTALL_DIR={str(install_dir)!r}
NO_SKILLS=false
PROBE_OUT={str(probe_out)!r}
export PROBE_OUT
unset HERMES_HOME
HERMES_HOME={str(hermes_home)!r}

log_info() {{ echo "INFO: $*" >>{str(log_out)!r}; }}
log_success() {{ echo "SUCCESS: $*" >>{str(log_out)!r}; }}
configure_browser_env_from_system_browser() {{ :; }}

{body}

copy_config_templates
"""
    env = {k: v for k, v in os.environ.items() if k != "HERMES_HOME"}
    proc = subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    child = probe_out.read_text() if probe_out.exists() else "<missing>"
    logs = log_out.read_text() if log_out.exists() else ""
    return {
        "child_hermes_home": child,
        "logs": logs,
        "hermes_home": str(hermes_home),
    }


def _run_setup_path(
    tmp_path: Path,
    *,
    hermes_home: Path | None = None,
    use_venv: bool = True,
    root_fhs: bool = False,
) -> Path:
    """Generate public launchers via setup_path; return the command-link dir."""
    source = INSTALL_SH.read_text()
    parts = [
        _extract_function(source, "is_termux"),
        _extract_function(source, "get_command_link_dir"),
        _extract_function(source, "get_command_link_display_dir"),
        _extract_function(source, "setup_path"),
    ]
    home = tmp_path / "home"
    if hermes_home is None:
        hermes_home = tmp_path / "custom-home"
    install_dir = tmp_path / "install"
    stub_python = install_dir / "venv" / "bin" / "python"
    entry = install_dir / "hermes"
    _write_executable(stub_python, _python_probe_script())
    entry.write_text("# entrypoint\n")
    command_dir = home / ".local" / "bin"
    command_dir.mkdir(parents=True)
    extra_bin = install_dir / "bin"
    extra_bin.mkdir(parents=True)
    # Non-venv setup_path uses `which hermes`. Keep that binary off the
    # command-link dir so `rm -f "$command_link_dir/hermes"` cannot replace
    # HERMES_BIN with the shim (self-exec loop).
    _write_executable(extra_bin / "hermes", _python_probe_script())

    fhs_override = ""
    if root_fhs:
        # Real FHS writes /usr/local/bin; redirect the link dir so the test
        # never touches the host. The ROOT_FHS_LAYOUT flag still drives bake.
        fhs_override = f"""
get_command_link_dir() {{ echo {str(command_dir)!r}; }}
get_command_link_display_dir() {{ echo '~/.local/bin'; }}
"""

    harness = f"""
set -eu
HOME={str(home)!r}
export HOME
HERMES_HOME={str(hermes_home)!r}
export HERMES_HOME
INSTALL_DIR={str(install_dir)!r}
USE_VENV={"true" if use_venv else "false"}
DISTRO=linux
ROOT_FHS_LAYOUT={"true" if root_fhs else "false"}
PATH={str(extra_bin)!r}:{str(command_dir)!r}:"$PATH"
export PATH

log_info() {{ :; }}
log_success() {{ :; }}
log_warn() {{ :; }}

{chr(10).join(parts)}
{fhs_override}

setup_path
"""
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["HERMES_HOME"] = str(hermes_home)
    proc = subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    launcher = command_dir / "hermes"
    assert launcher.is_file(), launcher
    return command_dir


def _probe_launcher(launcher: Path, *, extra_env: dict[str, str] | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "HERMES_HOME"}
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["bash", str(launcher)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return proc.stdout


def test_copy_config_templates_passes_hermes_home_to_skills_sync(tmp_path: Path) -> None:
    """skills_sync must see --hermes-home even when the shell did not export it."""
    result = _run_copy_config_templates(tmp_path)
    assert result["child_hermes_home"] == result["hermes_home"]


def test_copy_config_templates_logs_requested_home(tmp_path: Path) -> None:
    """Install logs must name the requested home, not a hardcoded ~/.hermes."""
    result = _run_copy_config_templates(tmp_path)
    assert f"Skills synced to {result['hermes_home']}/skills/" in result["logs"]
    assert "Skills synced to ~/.hermes/skills/" not in result["logs"]


def test_generated_launcher_defaults_install_hermes_home(tmp_path: Path) -> None:
    """A login-shell `hermes` with no HERMES_HOME must use the install home."""
    hermes_home = tmp_path / "custom-home"
    command_dir = _run_setup_path(tmp_path)
    assert _probe_launcher(command_dir / "hermes") == str(hermes_home)


def test_generated_launcher_preserves_caller_hermes_home(tmp_path: Path) -> None:
    """A caller-exported HERMES_HOME must win over the baked install default."""
    command_dir = _run_setup_path(tmp_path)
    override = tmp_path / "caller-home"
    assert _probe_launcher(
        command_dir / "hermes", extra_env={"HERMES_HOME": str(override)}
    ) == str(override)


def test_generated_launcher_does_not_pin_default_per_user_home(tmp_path: Path) -> None:
    """Default $HOME/.hermes must not be baked into a user-scoped shim."""
    home = tmp_path / "home"
    default_home = home / ".hermes"
    command_dir = _run_setup_path(tmp_path, hermes_home=default_home)
    assert _probe_launcher(command_dir / "hermes") == "<unset>"


def test_fhs_shim_does_not_bake_inherited_root_home(tmp_path: Path) -> None:
    """A root FHS /usr/local/bin/hermes must not pin inherited HERMES_HOME.

    Root may have HERMES_HOME=/root/experimental in the install environment
    without passing --hermes-home. Baking that path into the world-runnable
    shim would send every later uid into root's directory (#21457).
    """
    inherited = tmp_path / "root-experimental"
    command_dir = _run_setup_path(tmp_path, hermes_home=inherited, root_fhs=True)
    assert _probe_launcher(command_dir / "hermes") == "<unset>"
    assert _probe_launcher(command_dir / "hermes-agent") == "<unset>"
    assert _probe_launcher(command_dir / "hermes-acp") == "<unset>"


def test_non_venv_and_sibling_shims_bake_custom_home(tmp_path: Path) -> None:
    """Non-venv hermes plus hermes-agent / hermes-acp share the same bake."""
    hermes_home = tmp_path / "custom-home"
    command_dir = _run_setup_path(tmp_path, use_venv=False)
    for name in ("hermes", "hermes-agent", "hermes-acp"):
        assert _probe_launcher(command_dir / name) == str(hermes_home), name


def test_baked_home_survives_quote_in_path(tmp_path: Path) -> None:
    """A quote in --hermes-home must not produce a syntactically dead shim."""
    hermes_home = tmp_path / 'quoted"home'
    hermes_home.mkdir()
    command_dir = _run_setup_path(tmp_path, hermes_home=hermes_home)
    assert _probe_launcher(command_dir / "hermes") == str(hermes_home)
