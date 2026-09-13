"""Regression tests for install.sh not depending on a PATH-resolved `env`.

uv's installer writes ``~/.local/bin/env`` -- a PATH helper meant to be
*sourced*, which ignores its arguments and exits 0 -- and puts
``~/.local/bin`` ahead of ``/usr/bin``. Since install.sh installs uv, any
``env VAR=… cmd`` prefix in it can resolve to that script, run nothing, and
return success. install_browser_use_cli used the result of exactly such a
call as its success condition, so the installer logged "Browser Use CLI
installed" for an install that never happened.
"""

import re
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

# `env FOO=bar cmd` where env is resolved through PATH. Excludes /usr/bin/env
# and similar absolute or relative paths, which are not PATH-resolved.
BARE_ENV_PREFIX = re.compile(r"(?:^|[^./\w-])env\s+[A-Z_]+=")
# Comments discuss this pattern by name, so strip them before matching.
COMMENT = re.compile(r"(?:^|\s)#.*$")


def test_no_path_resolved_env_prefix() -> None:
    """No install.sh line may set variables via a PATH-resolved `env`."""
    offenders = [
        (number, line.rstrip())
        for number, line in enumerate(INSTALL_SH.read_text().splitlines(), 1)
        if BARE_ENV_PREFIX.search(COMMENT.sub("", line))
    ]
    assert not offenders, (
        "install.sh must not prefix commands with a PATH-resolved `env` -- a "
        "shadowing ~/.local/bin/env (written by uv, which this installer "
        "installs) swallows the command and returns 0. Use a subshell export "
        "or `sh -c` instead. Offending lines: " + repr(offenders)
    )


def test_browser_use_install_uses_subshell_export() -> None:
    """The Browser Use CLI install must not gate success on an `env` call."""
    text = INSTALL_SH.read_text()
    assert "export UV_NO_CONFIG=1 UV_TOOL_BIN_DIR=" in text
    assert "run_with_timeout 600 env UV_NO_CONFIG=1" not in text


def _write_env_shim(directory: Path) -> None:
    """Drop in a no-op `env` matching the one uv installs."""
    shim = directory / "env"
    shim.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            # add binaries to PATH if they aren't added yet
            case ":${PATH}:" in
                *:"$HOME/.local/bin":*) ;;
                *) export PATH="$HOME/.local/bin:$PATH" ;;
            esac
            """
        )
    )
    shim.chmod(0o755)


def test_env_shim_hides_failure_but_subshell_export_does_not(tmp_path: Path) -> None:
    """Behavioural proof of the bug and of the fix.

    With a shadowing `env` first on PATH, the old construct reports success
    for a command that never ran. The subshell-export form still propagates
    the real exit status, and still delivers the variables.
    """
    _write_env_shim(tmp_path)
    path = f"{tmp_path}:/usr/bin:/bin"

    def run(script: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", script],
            env={"PATH": path, "HOME": str(tmp_path)},
            capture_output=True,
            text=True,
        )

    # The old shape: `false` never runs, yet the caller sees success.
    assert run("env FOO=bar false").returncode == 0

    # The new shape: failure is reported, and the variable is exported.
    assert run("( export FOO=bar; false )").returncode != 0
    delivered = run('( export FOO=bar; printf "%s" "$FOO" )')
    assert delivered.stdout == "bar"
