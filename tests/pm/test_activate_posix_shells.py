"""``source activate`` in bash and zsh: same repo resolution, same save/restore, one prompt prefix.

zsh is a supported shell for ``activate`` (its header says so), but it shares none of the bash-only
mechanisms the script once relied on: ``BASH_SOURCE`` is empty there (activation fell back to the
caller's cwd as the repository), unquoted ``$list`` does not word-split (the whole key list became
one bogus name and the save loop died with ``bad substitution``), ``type -t`` does not exist (a
second ``source`` snapshotted the already-activated shell, so the prompt prefix doubled and
``deactivate`` restored the activated state) and ``PROMPT_COMMAND`` is ignored. Every case runs in
both shells, from the repository root and from outside it with an absolute path.

Only named canaries are printed: the composed environment is the caller's whole environment, so
these scripts never echo ``env`` or trace with ``set -x``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pm.environments import install_key
from tests.pm.activation_support import CANARY, bash, bash_env, fake_store, isolated_checkout, posix

pytestmark = pytest.mark.platforms("posix")

SHELLS = ("bash", "zsh")
PLACEMENTS = ("repo-root", "outside-absolute")
GENERATION = "gen-" + "0" * 32


def _shell(name: str) -> list[str]:
    if name == "bash":
        return [bash(), "--noprofile", "--norc", "-c"]
    found = shutil.which("zsh")
    if found is None:
        pytest.skip("zsh is not installed")
    # -f: no startup files, so the developer's own zshrc cannot change the result.
    return [found, "-f", "-c"]


def _checkout(tmp_path: Path) -> tuple[Path, dict]:
    root = isolated_checkout(tmp_path)
    # A git worktree, so the prompt prefix (which names the worktree) is drawn.
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    store, entry = fake_store(tmp_path)
    # Like the real python package: its bin directory goes on PATH, holding python and python3.
    (entry / "bin" / "python").symlink_to("python3")
    facts = json.loads((store / "facts.json").read_text(encoding="utf-8"))
    facts["packages"]["python"]["env"]["PATH"] = [f"{{{{store}}}}/{entry.name}/bin"]
    (store / "facts.json").write_text(json.dumps(facts), encoding="utf-8")
    env = bash_env(store)
    env.pop("PROMPT_COMMAND", None)
    env.pop("__HERMES_TEST_PYTHON", None)
    # The suite's interpreter, laid out the way pm.testenv selects it.
    testenv = Path(env["HERMES_HOME"]) / "installs" / install_key(root) / "test-environment"
    venv = testenv / GENERATION / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    (venv / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (testenv / "active.json").write_text(json.dumps({"generation": GENERATION}), encoding="utf-8")
    return root, env


def _run(shell: str, placement: str, tmp_path: Path, body: str) -> tuple[subprocess.CompletedProcess, Path]:
    root, env = _checkout(tmp_path)
    if placement == "repo-root":
        cwd, activate = root, "./activate"
    else:
        cwd, activate = tmp_path, posix(root / "activate")
    script = body.replace("@ACTIVATE@", f'"{activate}"').replace("@ROOT@", f'"{posix(root)}"')
    result = subprocess.run([*_shell(shell), script], capture_output=True, text=True, cwd=posix(cwd), env=env)
    return result, root


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize("shell", SHELLS)
def test_activate_resolves_the_checkout_and_exports_its_environment(shell, placement, tmp_path):
    result, root = _run(shell, placement, tmp_path, (
        'source @ACTIVATE@ || { echo "activate-failed:$?"; exit 1; }\n'
        f'printf "canary=%s\\n" "${CANARY}"\n'
        'printf "worktree=%s\\n" "$__HERMES_WORKTREE"\n'
        'printf "test_python=%s\\n" "$__HERMES_TEST_PYTHON"\n'
        'printf "activated=%s\\n" "${__HERMES_ACTIVATED:+yes}"\n'
    ))
    assert result.returncode == 0, result.stderr
    assert "bad substitution" not in result.stderr
    lines = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert lines["canary"] == "env-ok"
    assert Path(lines["worktree"]).resolve() == root.resolve()
    assert lines["test_python"].endswith(f"/test-environment/{GENERATION}/venv/bin/python")
    assert lines["activated"] == "yes"


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize("shell", SHELLS)
def test_deactivate_restores_set_and_unset_variables_exactly(shell, placement, tmp_path):
    result, _ = _run(shell, placement, tmp_path, (
        f"export {CANARY}='prior value with spaces $HOME * \"q\"'\n"
        "unset PYTHONPATH\n"
        'saved_path="$PATH"\n'
        'source @ACTIVATE@ || exit 1\n'
        f'test "${CANARY}" = env-ok || {{ echo "not-applied"; exit 2; }}\n'
        'test -n "${PYTHONPATH+set}" || { echo "pythonpath-not-applied"; exit 3; }\n'
        'deactivate\n'
        f"test \"${CANARY}\" = 'prior value with spaces $HOME * \"q\"' || {{ echo \"set-not-restored\"; exit 4; }}\n"
        'test -z "${PYTHONPATH+set}" || { echo "unset-not-restored"; exit 5; }\n'
        'test "$PATH" = "$saved_path" || { echo "path-not-restored"; exit 6; }\n'
        'test -z "${__HERMES_ACTIVATED+set}" || { echo "sentinel-left"; exit 7; }\n'
        'test -z "${__HERMES_TEST_PYTHON+set}" || { echo "test-python-left"; exit 8; }\n'
        'if typeset -f deactivate >/dev/null 2>&1 || typeset -f hermes >/dev/null 2>&1; then echo "functions-left"; exit 9; fi\n'
        'echo restored\n'
    ))
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip().endswith("restored")


@pytest.mark.parametrize("shell", SHELLS)
def test_sourcing_twice_then_deactivate_restores_the_original_shell(shell, tmp_path):
    result, _ = _run(shell, "outside-absolute", tmp_path, (
        f"export {CANARY}=original\n"
        "PS1='base$ '\n"
        'source @ACTIVATE@ || exit 1\n'
        'source @ACTIVATE@ || exit 1\n'
        'deactivate\n'
        f'test "${CANARY}" = original || {{ echo "canary-not-restored"; exit 2; }}\n'
        "test \"$PS1\" = 'base$ ' || { echo \"ps1-not-restored\"; exit 3; }\n"
        'echo restored\n'
    ))
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip().endswith("restored")


@pytest.mark.parametrize("shell", SHELLS)
def test_prompt_prefix_is_drawn_exactly_once(shell, tmp_path):
    # Whatever the shell runs before each prompt, run it twice, as two prompts would.
    if shell == "zsh":
        hook_count = 'print -r -- "hooks=${#${(@M)precmd_functions:#_hermes_prompt}}"\n'
        draw = 'for f in $precmd_functions; do "$f"; done\n'
        uses_prompt_command = 'printf "prompt_command=%s\\n" "${PROMPT_COMMAND-unset}"\n'
    else:
        hook_count = 'printf "hooks=%s\\n" "$(printf "%s" "$PROMPT_COMMAND" | grep -o _hermes_prompt | wc -l | tr -d " ")"\n'
        draw = 'eval "$PROMPT_COMMAND"\n'
        uses_prompt_command = ""
    result, root = _run(shell, "outside-absolute", tmp_path, (
        "PS1='base$ '\n"
        'source @ACTIVATE@ || exit 1\n'
        'source @ACTIVATE@ || exit 1\n'
        'cd @ROOT@ || exit 1\n'
        f"{draw}{draw}"
        'printf "ps1=%s\\n" "$PS1"\n'
        f"{hook_count}{uses_prompt_command}"
        'deactivate\n'
        f"{hook_count}"
        'printf "after=%s\\n" "$PS1"\n'
    ))
    assert result.returncode == 0, (result.stdout, result.stderr)
    lines = [line.split("=", 1) for line in result.stdout.splitlines() if "=" in line]
    ps1 = next(value for key, value in lines if key == "ps1")
    assert ps1 == f"({root.name}) base$ ", ps1
    hooks = [value for key, value in lines if key == "hooks"]
    assert hooks == ["1", "0"], hooks
    if shell == "zsh":
        assert dict(lines)["prompt_command"] == "unset"
    assert dict(lines)["after"] == "base$ "


@pytest.mark.parametrize("shell", SHELLS)
def test_python_resolves_to_the_managed_interpreter(shell, tmp_path):
    result, _ = _run(shell, "outside-absolute", tmp_path, (
        'source @ACTIVATE@ || exit 1\n'
        'printf "python=%s\\n" "$(command -v python)"\n'
        'printf "python3=%s\\n" "$(command -v python3)"\n'
    ))
    assert result.returncode == 0, (result.stdout, result.stderr)
    store = (tmp_path / "store").resolve()
    found = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    for name in ("python", "python3"):
        assert Path(found[name]).parent.resolve().is_relative_to(store), found


@pytest.mark.parametrize("shell", SHELLS)
def test_activation_prints_no_environment_values(shell, tmp_path):
    """A secret in the caller's environment is re-exported by activation; it must never be echoed."""
    secret = "hermes-activation-canary-" + "9" * 24
    root, env = _checkout(tmp_path)
    env["HERMES_FAKE_PROVIDER_API_KEY"] = secret
    result = subprocess.run([*_shell(shell), f'source "{posix(root / "activate")}" && deactivate'],
                            capture_output=True, text=True, cwd=posix(tmp_path), env=env)
    assert result.returncode == 0, result.stderr.replace(secret, "<redacted>")
    assert secret not in result.stdout
    assert secret not in result.stderr
