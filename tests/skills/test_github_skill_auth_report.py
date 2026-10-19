"""Behaviour contract for skills/software-development/github/scripts/gh-env.sh.

The helper is a *reporter*: it may print only a verdict a probe actually produced. Two
false negatives shipped on main:

* The detection ladder used ``&>``, which is a bashism. Under ``/bin/sh`` (dash)
  ``command -v gh &>/dev/null`` backgrounds the test, the condition reports success, and
  the helper printed ``GitHub Auth: gh`` on a machine with no ``gh`` installed at all.
* ``$HOME/.git-credentials`` was the last word, so in a repo carrying its own absolute
  ``credential.helper store --file=<path>`` the helper printed ``GitHub Auth: none``
  while git authentication worked fine -- and a reader who believed "none" sets up a
  second credential or gives up.

The helper is always sourced from the installed layout (``$HERMES_HOME/skills/...``), which
is how every caller reaches it. No live network: ``curl`` and ``gh`` are stubbed on PATH.
"""

from pathlib import Path
import os
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "skills/software-development/github/scripts"

# Assembled at runtime so no credential-shaped literal sits in this file's source.
FIXTURE_TOKEN = "ghp_" + "TESTONLY" + "000000"

SHELLS = ["/bin/bash", "/bin/sh"]
SHELLS = [s for s in SHELLS if Path(s).exists()]


def _install_skill(tmp_path: Path) -> tuple[Path, Path]:
    """Stage the shipped scripts at the path a real install puts them on. Returns
    (hermes_home, helper)."""
    hermes_home = tmp_path / "hermes-home"
    dest = hermes_home / "skills" / "software-development" / "github" / "scripts"
    dest.mkdir(parents=True)
    for name in ("gh-env.sh", "git-credential-token.py"):
        shutil.copy2(SCRIPTS / name, dest / name)
    return hermes_home, dest / "gh-env.sh"


def _stub_bin(tmp_path: Path, *, gh: str | None) -> Path:
    """A PATH front directory: curl that never touches the network, plus the gh fixture
    this case asks for ('auth' = authenticated, 'refused' = installed but logged out,
    None = absent, and the stub dir shadows any real gh on the runner)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    curl = bin_dir / "curl"
    curl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    curl.chmod(0o755)
    gh_path = bin_dir / "gh"
    if gh == "auth":
        gh_path.write_text(
            "#!/bin/sh\n"
            'case "$1 $2" in\n'
            '"auth status") echo "Logged in to github.com as fixtureuser"; exit 0 ;;\n'
            '"auth token") printf %s "' + FIXTURE_TOKEN + '"; exit 0 ;;\n'
            '"api user") echo fixtureuser; exit 0 ;;\n'
            "esac\nexit 0\n",
            encoding="utf-8",
        )
    elif gh == "refused":
        gh_path.write_text(
            '#!/bin/sh\necho "gh: not logged in" >&2\nexit 1\n', encoding="utf-8"
        )
    else:
        # Shadow any real gh: exit 127 the way a missing binary would.
        gh_path.write_text('#!/bin/sh\nexit 127\n', encoding="utf-8")
    gh_path.chmod(0o755)
    return bin_dir


def _source(shell: str, tmp_path: Path, cwd: Path, *, gh: str | None = "absent"):
    hermes_home, helper = _install_skill(tmp_path)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "PATH": str(_stub_bin(tmp_path, gh=gh)) + os.pathsep + os.environ["PATH"],
        "HOME": str(home),
        "HERMES_HOME": str(hermes_home),
    }
    env.pop("GITHUB_TOKEN", None)
    runner = tmp_path / "runner.sh"
    runner.write_text(
        f'. "{helper}"\n'
        'printf "METHOD=%s\\n" "$GH_AUTH_METHOD"\n'
        'printf "SOURCE=%s\\n" "$GH_TOKEN_SOURCE"\n'
        'printf "CRED=%s\\n" "$GH_CRED_SOURCE"\n',
        encoding="utf-8",
    )
    return subprocess.run(
        [shell, str(runner)], capture_output=True, text=True, cwd=cwd, env=env, timeout=60
    )


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shell", SHELLS)
def test_reports_gh_only_when_gh_is_the_mechanism(tmp_path, shell):
    """No usable gh, no credentials: the verdict may never be "gh" (the `&>` bashism)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    result = _source(shell, tmp_path, repo, gh="absent")

    assert result.returncode == 0, result.stderr
    assert "METHOD=gh" not in result.stdout, (
        "the helper claimed gh without an authenticated gh -- `&>` under dash backgrounds "
        "the test and reports success (the bug this test exists for)"
    )
    assert "METHOD=none" in result.stdout


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shell", SHELLS)
def test_repo_local_store_helper_is_named_never_tolerated_as_none(tmp_path, shell):
    """A repo-local `store --file=` authorises git here; printing "none" is a lie."""
    if not shutil.which("git"):
        pytest.skip("git required")
    repo = tmp_path / "repo"
    repo.mkdir()
    store = tmp_path / "repo-credentials"
    store.write_text(
        f"https://fixtureuser:{FIXTURE_TOKEN}@github.com\n", encoding="utf-8", newline=""
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "credential.helper", f"store --file={store}"],
        check=True,
        capture_output=True,
    )

    result = _source(shell, tmp_path, repo, gh="absent")

    assert result.returncode == 0, result.stderr
    assert "METHOD=none" not in result.stdout, (
        "git authorises from a repo-local store in this directory; 'none' is the false "
        "negative this test exists for"
    )
    assert "METHOD=curl" in result.stdout
    assert str(store) in result.stdout, "the reporter must name the credential path it read"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shell", SHELLS)
def test_authenticated_gh_is_the_source_of_truth(tmp_path, shell):
    """Where gh authenticates, callers get gh's own token -- and never see it printed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    result = _source(shell, tmp_path, repo, gh="auth")

    assert result.returncode == 0, result.stderr
    assert "METHOD=gh" in result.stdout
    assert "SOURCE=gh auth token" in result.stdout
    assert FIXTURE_TOKEN not in result.stdout, "evidence lines must never carry the token"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shell", SHELLS)
def test_installed_but_unauthenticated_gh_falls_through_and_says_so(tmp_path, shell):
    """`gh` present but logged out must be reported as such, not treated as auth."""
    repo = tmp_path / "repo"
    repo.mkdir()
    result = _source(shell, tmp_path, repo, gh="refused")

    assert result.returncode == 0, result.stderr
    assert "METHOD=gh" not in result.stdout
    assert "METHOD=none" in result.stdout


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shell", SHELLS)
def test_sourcing_under_set_e_survives_outside_a_work_tree(tmp_path, shell):
    """`git remote get-url` exits 128 outside a repo; a `set -eu` caller must not die."""
    hermes_home, helper = _install_skill(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "PATH": str(_stub_bin(tmp_path, gh="absent")) + os.pathsep + os.environ["PATH"],
        "HOME": str(home),
        "HERMES_HOME": str(hermes_home),
    }
    env.pop("GITHUB_TOKEN", None)
    runner = tmp_path / "strict.sh"
    runner.write_text(f'set -eu\n. "{helper}"\nprintf "SURVIVED\\n"\n', encoding="utf-8")
    result = subprocess.run(
        [shell, str(runner)], capture_output=True, text=True, cwd=outside, env=env, timeout=60
    )

    assert "SURVIVED" in result.stdout, (
        f"sourcing the helper aborted a set -eu caller: rc={result.returncode} "
        f"out={result.stdout!r} err={result.stderr!r}"
    )
