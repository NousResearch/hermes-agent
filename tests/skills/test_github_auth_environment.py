"""Exercise the bundled auth helper without any retired GitHub skills."""

from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest


def test_credential_store_fallback_works_with_only_consolidated_skill(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("The bundled shell helper requires Bash")

    user_home = tmp_path / "user"
    user_home.mkdir()
    (user_home / ".git-credentials").write_text(
        "https://octocat:fixture-token@github.com\n", encoding="utf-8"
    )
    hermes_home = tmp_path / "hermes"
    skill = hermes_home / "skills/software-development/github"
    shutil.copytree(
        Path(__file__).resolve().parents[2] / "skills/software-development/github",
        skill,
    )

    # Stub only external services and the uv launcher. The shipped shell
    # helper must discover and execute the real credential parser itself.
    result = subprocess.run(
        [
            bash, "--noprofile", "--norc", "-c",
            r'''
gh() { return 1; }
git() { return 1; }
uv() {
    [ "$1" = run ] || return 1
    shift 2
    "$TEST_PYTHON" "$@"
}
curl() {
    [ "$1" = -s ] && [ "$2" = -H ] || return 1
    [ "$3" = "Authorization: token fixture-token" ] || return 1
    [ "$4" = https://api.github.com/user ] || return 1
    printf '%s\n' '{"login":"octocat"}'
}
source "$1"
[ "$GH_AUTH_METHOD" = curl ] &&
[ "$GITHUB_TOKEN" = fixture-token ] &&
[ "$GH_USER" = octocat ]
''',
            "github-auth-test", str(skill / "scripts/gh-env.sh"),
        ],
        cwd=tmp_path,
        env={
            "HOME": str(user_home),
            "HERMES_HOME": str(hermes_home),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
            "TEST_PYTHON": sys.executable,
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "GitHub Auth: curl" in result.stdout
    assert "User: octocat" in result.stdout
    assert result.stderr == ""
