"""Behavioral coverage for Hermes-managed Node exposure and npm prefixes."""

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
NODE_BOOTSTRAP = REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh"
SCRIPT_CASES = (
    pytest.param("installer", INSTALL_SH, id="installer"),
    pytest.param("bootstrap", NODE_BOOTSTRAP, id="lazy-bootstrap"),
)


def _write_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> None:
    path.write_text(body)
    path.chmod(0o755)


def _run_shell(
    script_kind: str,
    script_path: Path,
    body: str,
    *,
    home: Path,
    hermes_home: Path,
    path: str | None = None,
    link_dir: Path | None = None,
) -> None:
    source = (
        'source "$SCRIPT_UNDER_TEST" --manifest >/dev/null'
        if script_kind == "installer"
        else 'source "$SCRIPT_UNDER_TEST"'
    )
    harness = f"""
set -e
{source}
{body}
"""
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "HOME": str(home),
            "HERMES_HOME": str(hermes_home),
            "SCRIPT_KIND": script_kind,
            "SCRIPT_UNDER_TEST": str(script_path),
            "TERMUX_VERSION": "",
            "PREFIX": "",
        }
    )
    if path is not None:
        env["PATH"] = path
    if link_dir is not None:
        env["LINK_DIR"] = str(link_dir)

    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("script_kind", "script_path"), SCRIPT_CASES)
def test_user_install_repairs_legacy_links_and_keeps_npm_private(
    tmp_path: Path,
    script_kind: str,
    script_path: Path,
) -> None:
    home = tmp_path / "home"
    hermes_home = tmp_path / "hermes"
    managed_bin = hermes_home / "node" / "bin"
    link_dir = home / ".local" / "bin"
    fake_bin = tmp_path / "fake-bin"
    managed_bin.mkdir(parents=True)
    link_dir.mkdir(parents=True)
    fake_bin.mkdir()

    for tool in ("node", "npm", "npx"):
        _write_executable(managed_bin / tool)
        (link_dir / tool).symlink_to(managed_bin / tool)
    _write_executable(fake_bin / "node", "#!/bin/sh\necho v22.13.0\n")

    body = """
if [ "$SCRIPT_KIND" = installer ]; then
    ROOT_FHS_LAYOUT=false
    check_node
else
    ensure_node
fi
"""
    _run_shell(
        script_kind,
        script_path,
        body,
        home=home,
        hermes_home=hermes_home,
        path=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    )

    assert (hermes_home / "node" / "etc" / "npmrc").read_text() == (
        f"prefix={hermes_home / 'node'}\n"
    )
    for tool in ("node", "npm", "npx"):
        assert not (link_dir / tool).is_symlink()


@pytest.mark.parametrize(("script_kind", "script_path"), SCRIPT_CASES)
def test_root_fhs_install_exposes_managed_tools_and_shared_prefix(
    tmp_path: Path,
    script_kind: str,
    script_path: Path,
) -> None:
    home = tmp_path / "home"
    hermes_home = tmp_path / "hermes"
    managed_bin = hermes_home / "node" / "bin"
    link_dir = tmp_path / "usr-local" / "bin"
    managed_bin.mkdir(parents=True)

    for tool in ("node", "npm", "npx"):
        _write_executable(managed_bin / tool)

    body = """
if [ "$SCRIPT_KIND" = installer ]; then
    ROOT_FHS_LAYOUT=true
    get_command_link_dir() { printf '%s\n' "$LINK_DIR"; }
    link_managed_node_tools_if_needed
    configure_managed_node_npm_prefix
else
    id() {
        if [ "${1:-}" = -u ]; then printf '0\n'; else command id "$@"; fi
    }
    uname() {
        if [ "${1:-}" = -s ]; then printf 'Linux\n'; else command uname "$@"; fi
    }
    _nb_get_link_dir() { printf '%s\n' "$LINK_DIR"; }
    _nb_link_managed_node_tools_if_needed
    _nb_configure_npm_prefix
fi
"""
    _run_shell(
        script_kind,
        script_path,
        body,
        home=home,
        hermes_home=hermes_home,
        link_dir=link_dir,
    )

    for tool in ("node", "npm", "npx"):
        assert (link_dir / tool).is_symlink()
        assert (link_dir / tool).resolve() == (managed_bin / tool).resolve()
    assert (hermes_home / "node" / "etc" / "npmrc").read_text() == (
        f"prefix={link_dir.parent}\n"
    )


@pytest.mark.parametrize(("script_kind", "script_path"), SCRIPT_CASES)
def test_user_install_preserves_non_hermes_tool_link(
    tmp_path: Path,
    script_kind: str,
    script_path: Path,
) -> None:
    home = tmp_path / "home"
    hermes_home = tmp_path / "hermes"
    link_dir = home / ".local" / "bin"
    user_bin = tmp_path / "user-node" / "bin"
    (hermes_home / "node").mkdir(parents=True)
    link_dir.mkdir(parents=True)
    user_bin.mkdir(parents=True)
    _write_executable(user_bin / "node")
    (link_dir / "node").symlink_to(user_bin / "node")

    body = """
if [ "$SCRIPT_KIND" = installer ]; then
    ROOT_FHS_LAYOUT=false
    remove_private_managed_node_links
else
    _nb_should_link_managed_node_tools() { return 1; }
    _nb_remove_private_managed_node_links
fi
"""
    _run_shell(
        script_kind,
        script_path,
        body,
        home=home,
        hermes_home=hermes_home,
    )

    assert (link_dir / "node").is_symlink()
    assert (link_dir / "node").resolve() == (user_bin / "node").resolve()
