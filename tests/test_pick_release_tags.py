import json
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "sandbox" / "pick-release-tags.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _commit(repo: Path, message: str) -> None:
    marker = repo / "marker.txt"
    marker.write_text(message, encoding="utf-8")
    _git(repo, "add", "marker.txt")
    _git(
        repo,
        "-c", "user.name=E2E Test",
        "-c", "user.email=e2e@example.invalid",
        "commit", "-m", message,
    )


def test_picker_excludes_release_tag_pointing_at_update_destination(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")

    _commit(repo, "old release")
    _git(repo, "tag", "v2026.8.31")
    _commit(repo, "new release")
    _git(repo, "tag", "v2026.9.11")

    command = [str(SCRIPT), "--count", "1", "--repo", str(repo)]
    bash_major = int(
        subprocess.run(
            ["bash", "-c", "echo ${BASH_VERSINFO[0]}"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    )
    if bash_major < 4:
        # The production picker runs on Ubuntu. macOS ships Bash 3.2, so
        # provide only the Bash 4 `mapfile -t ARRAY` primitive it uses.
        shim = r'''mapfile() {
  [ "${1-}" = "-t" ] && shift
  local array="${1:-MAPFILE}" line escaped i=0
  eval "$array=()"
  while IFS= read -r line; do
    printf -v escaped '%q' "$line"
    eval "$array[$i]=$escaped"
    i=$((i + 1))
  done
}
export -f mapfile
exec "$@"'''
        command = ["bash", "-c", shim, "bash", *command]

    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )

    assert json.loads(result.stdout) == ["v2026.8.31"]
