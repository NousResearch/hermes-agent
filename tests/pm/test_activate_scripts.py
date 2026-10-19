"""``activate`` / ``activate.ps1``: an interactive shell takes on the environment and gives it back.

What a process launched under ``scripts/run-in-hermes-env`` sees is covered in
tests/scripts/test_run_in_hermes_env.py; how the environment is composed, in
test_environment_script.py; the fish port, in test_activate_fish.py.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests.pm.activation_support import (
    ACTIVATE, ACTIVATE_PS1, CANARY, REPO_ROOT, SETUP_HERMES_PS1, SETUP_HERMES_SH, bash, bash_env,
    child_env, fake_store, isolated_checkout, posix, powershell, spawnable_python,
)


def test_bash_scripts_pass_syntax_check():
    for script in (ACTIVATE, SETUP_HERMES_SH):
        result = subprocess.run(
            [bash(), "-n", posix(script)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=child_env(),
            check=False,
        )
        assert result.returncode == 0, f"{script.name}: {result.stderr}"


@pytest.mark.platforms("windows")
def test_source_activate_exports_the_pm_env(tmp_path: Path):
    root = isolated_checkout(tmp_path)
    store, _ = fake_store(tmp_path)
    script = (
        f'source "{posix(root / "activate")}" && '
        f'test -n "$__HERMES_ACTIVATED" && '
        f'printf "%s" "${CANARY}"'
    )
    result = subprocess.run(
        [bash(), "-c", script],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        cwd=posix(tmp_path),
        env=bash_env(store),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "env-ok"


@pytest.mark.platforms("windows")
def test_activate_exports_the_sentinel_to_child_processes(tmp_path: Path):
    """Repo scripts read activation from the environment, so the sentinel must
    survive into an exec'd child (a plain shell variable would not), and
    deactivate must take it back out."""
    root = isolated_checkout(tmp_path)
    store, _ = fake_store(tmp_path)
    script = (
        f'source "{posix(root / "activate")}" && '
        f'"$BASH" -c \'test -n "$__HERMES_ACTIVATED"\' && '
        f'deactivate && '
        f'! "$BASH" -c \'test -n "$__HERMES_ACTIVATED"\' && '
        f'echo exported-then-cleared'
    )
    result = subprocess.run(
        [bash(), "-c", script],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        cwd=posix(tmp_path),
        env=bash_env(store),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "exported-then-cleared"


@pytest.mark.platforms("windows")
def test_deactivate_restores_the_prior_shell(tmp_path: Path):
    root = isolated_checkout(tmp_path)
    store, _ = fake_store(tmp_path)
    script = (
        f'source "{posix(root / "activate")}" && deactivate && '
        f'test -z "${{{CANARY}+set}}" && '
        f'test -z "${{__HERMES_ACTIVATED+set}}" && '
        f"! declare -F deactivate >/dev/null && "
        f'echo restored'
    )
    result = subprocess.run(
        [bash(), "-c", script],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        cwd=posix(tmp_path),
        env=bash_env(store),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "restored"


@pytest.mark.platforms("windows")
def test_activate_leaves_the_shell_paths_in_posix_form(tmp_path: Path):
    """The pm env is read by a native Windows Python, which sees PATH, HOME and
    the temp variables in Windows form. Exported verbatim, `C:\\a;C:\\b` left
    bash without a usable PATH, so every command after activation failed."""
    root = isolated_checkout(tmp_path)
    store, _ = fake_store(tmp_path)
    script = (
        'prior_home="$HOME" prior_tmp="$TMP" && '
        f'source "{posix(root / "activate")}" && '
        'command -v basename >/dev/null && '
        'case "$PATH" in *";"*|*"\\\\"*) exit 3;; esac && '
        'test "$HOME" = "$prior_home" && test "$TMP" = "$prior_tmp" && '
        'echo posix'
    )
    result = subprocess.run(
        [bash(), "-c", script],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        cwd=posix(tmp_path),
        env=bash_env(store),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "posix"


def test_activate_fails_cleanly_without_a_store(tmp_path: Path):
    env = bash_env(tmp_path / "empty-store")
    isolated = isolated_checkout(tmp_path) / "activate"
    script = (
        f'source "{posix(isolated)}" 2>/dev/null; '
        f'test $? -ne 0 && echo refused'
    )
    result = subprocess.run(
        [bash(), "-c", script],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        cwd=posix(tmp_path),
        env=env,
        check=False,
    )
    # Without any provisioned python the source must refuse — never
    # silently no-op with a half-activated shell.
    assert "refused" in result.stdout


def test_powershell_scripts_parse():
    """Parse-check the PowerShell entry points; skip gracefully when no
    PowerShell host is available."""
    ps = powershell()
    if ps is None:
        pytest.skip("no PowerShell host available")
    for script in (ACTIVATE_PS1, SETUP_HERMES_PS1):
        result = subprocess.run(
            [
                ps,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"$errs = $null; $null = [System.Management.Automation.Language.Parser]::ParseFile("
                f"'{script}', [ref]$null, [ref]$errs); "
                f"if ($errs.Count) {{ $errs | ForEach-Object {{ $_.Message }}; exit 1 }}",
            ],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            check=False,
        )
        assert result.returncode == 0, f"{script.name}: {result.stdout}{result.stderr}"


#: The candidate-order block ``activate.ps1`` evaluates when no venv interpreter
#: exists. Delimited by markers in the script, never by a regex over its body, so an
#: edit to the block cannot silently stop being covered.
_CANDIDATES_BEGIN = "# --- BEGIN activation store candidates ---"
_CANDIDATES_END = "# --- END activation store candidates ---"


def _activation_candidates_block() -> str:
    """The marked block from ``activate.ps1``, ready to be evaluated."""
    text = (REPO_ROOT / "activate.ps1").read_text(encoding="utf-8-sig")
    match = re.search(
        re.escape(_CANDIDATES_BEGIN) + r"\r?\n(.*?)\r?\n *" + re.escape(_CANDIDATES_END),
        text, re.DOTALL,
    )
    assert match, "activate.ps1: the activation store-candidate block is missing"
    return match.group(1).strip()


def _as_posix_path(value: str) -> Path:
    """A PowerShell-emitted path as a host path: backslashes become separators.

    The candidate block is Windows shell, so it spells the neighbour with ``\\``
    even when evaluated by pwsh on POSIX.
    """
    return Path(value.replace("\\", "/"))


@pytest.mark.skipif(powershell() is None, reason="needs a PowerShell host")
@pytest.mark.platforms("any")
@pytest.mark.parametrize("home_kind", ["plain", "profile"])
def test_powershell_activate_searches_the_store_root_before_the_neighbour(
    tmp_path: Path, home_kind: str
) -> None:
    """The candidate order is: runtime override, resolved store root, then the neighbour.

    The neighbour used to come SECOND and the store root was hand-rolled, which on
    Windows handed activation a stale interpreter whenever a ``tools/`` directory sat
    beside the checkout. The order IS the fix, so it is asserted directly by
    evaluating the block rather than by watching stand-in interpreters run: on
    Windows a shell-script ``python.exe`` is not a PE, so every candidate fails
    identically and the winner is unobservable.
    """
    ps = powershell()
    checkout = isolated_checkout(tmp_path)
    home = tmp_path / "hermes-home" / ".hermes"
    store_home = home if home_kind == "plain" else home / "profiles" / "coder"
    env = bash_env(tmp_path / "store")
    env["HERMES_HOME"] = posix(store_home)
    env["HOME"] = posix(tmp_path / "hermes-home")
    env.pop("HERMES_RUNTIME_DIR", None)

    script = (
        "$ErrorActionPreference = 'Stop'\n"
        f"$repo = '{posix(checkout)}'\n"
        f". '{posix(checkout / 'scripts' / 'bootstrap' / 'store_root.ps1')}'\n"
        + _activation_candidates_block() + "\n"
        # One line per slot, '' for a skipped slot, so the shape is assertable.
        "foreach ($r in $roots) { if ($r) { Write-Output $r } else { Write-Output '<empty>' } }\n"
        "Write-Output ('store=' + (Join-Path (Get-HermesRoot) 'tools'))\n"
    )
    result = subprocess.run(
        [ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(tmp_path), env=env, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    *slots, store_line = lines
    resolved_store = Path(store_line.removeprefix("store=")).resolve()
    assert len(slots) == 3, f"expected three candidate slots, got {slots}"
    assert slots[0] == "<empty>", f"the runtime override slot must come first: {slots}"
    # The neighbour slot is spelled ``<checkout>\\..\\tools`` — a Windows separator
    # that POSIX Path treats as a file-name character — so normalise it before
    # resolving, or the comparison only holds on the host that wrote the spelling.
    assert _as_posix_path(slots[1]).resolve() == resolved_store, (slots[1], resolved_store)
    assert _as_posix_path(slots[1]).name == "tools", slots
    assert _as_posix_path(slots[2]).resolve() == (Path(checkout).parent / "tools").resolve(), slots


@pytest.mark.platforms("windows")
def test_powershell_activate_exports_and_deactivates(tmp_path: Path):
    # Native venv redirectors resolve the base DLLs/stdlib without copying CPython.
    root = isolated_checkout(tmp_path)
    env = bash_env(tmp_path / "store")
    subprocess.run(
        [str(spawnable_python()), "-m", "venv", "--without-pip", str(root / ".venv")],
        check=True, capture_output=True, env=env, timeout=60,
    )
    ps = powershell()
    assert ps, "native Windows test requires PowerShell"
    result = subprocess.run(
        [ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
         f"$ErrorActionPreference='Stop'; $env:PYTHONPATH='caller-original'; "
         f". '{root / 'activate.ps1'}'; "
         "Write-Output ('active=' + $env:PYTHONPATH); deactivate; "
         "Write-Output ('after=' + $env:PYTHONPATH)"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(tmp_path), env=env, timeout=40,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Nothing is committed, so the checkout alone: the bootstrap .venv's packages never leak in.
    assert f"active={root}\n" in result.stdout
    assert "after=caller-original" in result.stdout
