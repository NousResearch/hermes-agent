"""Execute GitHub reference setup and API gates with isolated auth fixtures.

Git-only setup must stay usable. API routes must reject Git-store-only or
missing API authentication. These are document contracts, not live API tests.
"""
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

REFERENCES = Path(__file__).resolve().parents[2] / "skills/software-development/github/references"
DOCUMENTS = ("pr-workflow.md", "repo-management.md")


def _blocks(document):
    return re.findall(r"```bash\n(.*?)\n```", (REFERENCES / document).read_text(), re.DOTALL)


def _environment(tmp_path, method):
    home = tmp_path / "home"
    helper = home / "skills/software-development/github/scripts/gh-env.sh"
    helper.parent.mkdir(parents=True)
    helper.write_text(f"export GH_AUTH_METHOD={method} GH_USER=fixture\n")
    bindir = home / "bin"
    bindir.mkdir()
    for name in ("gh", "curl"):
        command = bindir / name
        command.write_text('#!/bin/sh\nprintf "api-call\\n" >> "$CALLS_LOG"\nprintf \'{"login":"fixture"}\\n\'\n')
        command.chmod(0o700)
    (bindir / "python").symlink_to(sys.executable)
    env = {"HOME": str(home), "HERMES_HOME": str(home),
           "PATH": str(bindir) + os.pathsep + "/usr/bin:/bin",
           "CALLS_LOG": str(home / "api-calls"), "AUTH": method}
    return env


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("document", DOCUMENTS)
@pytest.mark.parametrize("method", ("git", "none", "gh", "curl"))
def test_general_detection_allows_local_git(tmp_path, document, method):
    env = _environment(tmp_path, method)
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "--quiet", str(repo)], env=env, check=True)
    setup = _blocks(document)[0]
    result = subprocess.run(["/bin/bash", "-c", setup + '\ngit -C "$1" rev-parse --is-inside-work-tree',
                             "fixture", str(repo)], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "true", result.stdout
    if method in ("git", "none"):
        assert not Path(env["CALLS_LOG"]).exists(), "Git-only setup invoked an API command"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("document", DOCUMENTS)
@pytest.mark.parametrize("method", ("git", "none", "gh", "curl"))
def test_api_gate_rejects_non_api_authentication(tmp_path, document, method):
    env = _environment(tmp_path, method)
    gates = [block for block in _blocks(document) if 'case "$AUTH"' in block and "exit 1" in block]
    assert gates, "No fail-closed API gate documented"
    result = subprocess.run(["/bin/bash", "-c", gates[0] + '\nprintf "API_ROUTE_ALLOWED\\n"'],
                            env=env, capture_output=True, text=True, check=False)
    if method in ("git", "none"):
        assert result.returncode == 1
        assert "API_ROUTE_ALLOWED" not in result.stdout
        assert not Path(env["CALLS_LOG"]).exists(), "Rejected gate invoked an API command"
    else:
        assert result.returncode == 0, result.stderr
        assert "API_ROUTE_ALLOWED" in result.stdout


@pytest.mark.platforms("posix")
def test_git_store_detection_never_exports_token(tmp_path):
    home = tmp_path / "home"
    bindir = home / "bin"
    bindir.mkdir(parents=True)
    (home / ".git-credentials").write_text("https://fixture:fixture-only@github.com\n")
    scripts = {
        "gh": "#!/bin/sh\nexit 1\n",
        "uv": '#!/bin/sh\nprintf "extract\\n" >> "$CALLS_LOG"\nprintf "fixture-only\\n"\n',
        "curl": "#!/bin/sh\nprintf 'api\\n' >> \"$CALLS_LOG\"\nprintf '{\"login\":\"fixture\"}\\n'\n",
    }
    for name, text in scripts.items():
        executable = bindir / name
        executable.write_text(text)
        executable.chmod(0o700)
    env = {
        "HOME": str(home), "HERMES_HOME": str(home),
        "PATH": str(bindir) + os.pathsep + "/usr/bin:/bin",
        "CALLS_LOG": str(home / "calls"),
    }
    helper = REFERENCES.parent / "scripts/gh-env.sh"
    result = subprocess.run(
        ["/bin/bash", "-c", 'source "$1" >/dev/null; printf "%s:%s" "$GH_AUTH_METHOD" "${GITHUB_TOKEN:-}"', "fixture", str(helper)],
        cwd=home, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "git:", result.stdout
    assert not Path(env["CALLS_LOG"]).exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("document", DOCUMENTS)
def test_setup_uses_shipped_helper_when_legacy_install_exists(tmp_path, document):
    env = _environment(tmp_path, "git")
    legacy = Path(env["HOME"]) / "skills/github/github-auth/scripts/gh-env.sh"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('printf "legacy-called\\n" >> "$CALLS_LOG"\nexport GH_AUTH_METHOD=git\n')
    result = subprocess.run(
        ["/bin/bash", "-c", _blocks(document)[0]],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not Path(env["CALLS_LOG"]).exists(), "Setup invoked an unreviewed legacy helper"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("document", (*DOCUMENTS, "issues.md", "code-review.md", "github-api-cheatsheet.md"))
@pytest.mark.parametrize("inherited", ("gh", "curl"))
def test_missing_helper_never_authorizes_inherited_auth(tmp_path, document, inherited):
    env = _environment(tmp_path, inherited)
    (Path(env["HERMES_HOME"]) / "skills/software-development/github/scripts/gh-env.sh").unlink()
    env["GH_AUTH_METHOD"] = inherited
    blocks = _blocks(document)
    gate = next((b for b in blocks[1:] if 'case "$AUTH"' in b), "") if document in DOCUMENTS else ""
    result = subprocess.run(
        ["/bin/bash", "-c", blocks[0] + "\n" + gate + '\nprintf "API_ROUTE_ALLOWED\\n"'],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "API_ROUTE_ALLOWED" not in result.stdout
    assert not Path(env["CALLS_LOG"]).exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("document", DOCUMENTS)
@pytest.mark.parametrize("method", ("gh", "curl"))
def test_real_helper_git_only_setup_makes_no_remote_calls(tmp_path, document, method):
    env = _environment(tmp_path, method)
    helper = Path(env["HERMES_HOME"]) / "skills/software-development/github/scripts/gh-env.sh"
    helper.write_bytes((REFERENCES.parent / "scripts/gh-env.sh").read_bytes())
    gh = Path(env["HOME"]) / "bin/gh"
    gh.write_text('#!/bin/sh\nprintf "gh-command\\n" >> "$CALLS_LOG"\nexit "${GH_STATUS:-1}"\n')
    env["GH_STATUS"] = "0" if method == "gh" else "1"
    if method == "curl":
        env["GITHUB_TOKEN"] = "fixture-only"
    result = subprocess.run(
        ["/bin/bash", "-c", _blocks(document)[0] + '\nprintf "GIT_READY\\n"'],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "GIT_READY" in result.stdout
    assert not Path(env["CALLS_LOG"]).exists(), "Git-only setup made a remote call"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("document", DOCUMENTS)
@pytest.mark.parametrize("method", ("gh", "curl", "git", "none"))
def test_real_helper_api_gate_refreshes_authentication(tmp_path, document, method):
    env = _environment(tmp_path, method)
    home = Path(env["HOME"])
    helper = home / "skills/software-development/github/scripts/gh-env.sh"
    helper.write_bytes((REFERENCES.parent / "scripts/gh-env.sh").read_bytes())
    gh = home / "bin/gh"
    gh.write_text('#!/bin/sh\nif [ "$1 $2" = "auth status" ]; then exit "$GH_STATUS"; fi\nprintf "fixture\\n"\n')
    env["GH_STATUS"] = "0" if method == "gh" else "1"
    if method == "curl":
        env["GITHUB_TOKEN"] = "fixture-only"
    if method == "git":
        (home / ".git-credentials").write_text("https://fixture:***@github.com\n")
    env["GH_AUTH_METHOD"] = "gh"
    env["AUTH"] = "gh"
    gate = next(b for b in _blocks(document)[1:] if 'case "$AUTH"' in b)
    result = subprocess.run(
        ["/bin/bash", "-c", _blocks(document)[0] + "\n" + gate + '\nprintf "API_ROUTE_ALLOWED:%s\\n" "$AUTH"'],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    if method in ("gh", "curl"):
        assert result.returncode == 0, result.stderr
        assert "API_ROUTE_ALLOWED:" + method in result.stdout
    else:
        assert result.returncode != 0
        assert "API_ROUTE_ALLOWED" not in result.stdout
        assert not Path(env["CALLS_LOG"]).exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("document", DOCUMENTS)
def test_missing_helper_does_not_block_local_git_setup(tmp_path, document):
    env = _environment(tmp_path, "gh")
    (Path(env["HERMES_HOME"]) / "skills/software-development/github/scripts/gh-env.sh").unlink()
    env["GH_AUTH_METHOD"] = "gh"
    result = subprocess.run(
        ["/bin/bash", "-c", _blocks(document)[0] + '\nprintf "LOCAL:%s\\n" "$AUTH"'],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "LOCAL:none" in result.stdout
    assert not Path(env["CALLS_LOG"]).exists()
