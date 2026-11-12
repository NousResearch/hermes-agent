"""scripts/ci/check_blob_size.py on real temp git repos: what a PR may add, and what it may not."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ci" / "check_blob_size.py"
sys.path.insert(0, str(SCRIPT.parent))

import check_blob_size as cbs

LIMIT = 4096


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return env


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, env=_env(), capture_output=True, text=True,
                          encoding="utf-8", stdin=subprocess.DEVNULL, timeout=60, check=False)
    assert proc.returncode == 0, (args, proc.stderr)
    return proc.stdout.strip()


def _binary(n: int, seed: int = 0) -> bytes:
    return bytes((i * 7 + seed) % 256 for i in range(n))  # contains NUL bytes: git calls it binary


def _text(n: int, seed: str = "x") -> bytes:
    return (seed * 63 + "\n").encode() * (n // 64 + 1)


def _commit(repo: Path, files: dict[str, bytes | None], msg: str = "step") -> str:
    for rel, data in files.items():
        path = repo / rel
        if data is None:
            _git(repo, "rm", "-q", "--", rel)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        _git(repo, "add", "--", rel)
    _git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _commit(root, {
        "big.lock": _text(LIMIT * 3, "l"),          # legacy text file already over the limit
        "assets/icon.icns": _binary(LIMIT * 2, 1),  # legacy binary already over the limit
        "assets/small.png": _binary(100, 2),
    }, "base")
    return root


def _offenders(repo: Path, base: str, head: str | None, allow: frozenset[str] | set[str] = frozenset()):
    return [(o.path, o.kind, o.status) for o in cbs.find_offenders(repo, base, head, LIMIT, allow)]


@pytest.mark.parametrize("files, expected", [
    # added binary over the limit
    ({"docs/shot.png": _binary(LIMIT + 1)}, [("docs/shot.png", "binary", "A")]),
    # added binary at exactly the limit, and a small one
    ({"docs/a.png": _binary(LIMIT), "docs/b.png": _binary(10)}, []),
    # a modified binary is judged on its new size, legacy or not
    ({"assets/icon.icns": _binary(LIMIT * 2, 9)}, [("assets/icon.icns", "binary", "M")]),
    ({"assets/small.png": _binary(LIMIT + 1, 3)}, [("assets/small.png", "binary", "M")]),
    # shrinking a legacy binary under the limit passes
    ({"assets/icon.icns": _binary(LIMIT - 1, 9)}, []),
    # a new text file over the limit (fuzz corpus, dumped JSON)
    ({"r318C_fuzz_new.json": _text(LIMIT + 64)}, [("r318C_fuzz_new.json", "new file", "A")]),
    # growing an existing text file is the FILE_LINES ratchet's job, not this check's
    ({"big.lock": _text(LIMIT * 4, "m")}, []),
    # deleting a big file is always fine
    ({"assets/icon.icns": None}, []),
])
def test_verdict(repo, files, expected):
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, files)
    assert _offenders(repo, base, head) == expected


def test_pure_move_adds_no_bytes_but_move_with_edit_is_judged(repo):
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "packaging").mkdir()
    (repo / "deps").mkdir()
    _git(repo, "mv", "assets/icon.icns", "packaging/icon.icns")
    _git(repo, "mv", "big.lock", "deps/big.lock")
    moved = _commit(repo, {})
    assert _offenders(repo, base, moved) == []
    edited = _commit(repo, {"packaging/icon.icns": _binary(LIMIT * 2, 5)})
    assert _offenders(repo, base, edited) == [("packaging/icon.icns", "binary", "R")]


def test_content_the_base_already_holds_adds_no_bytes(repo):
    # A byte-identical copy shares the blob; so does reverting to a version main still tracks.
    base = _git(repo, "rev-parse", "HEAD")
    copied = _commit(repo, {"other/icon.icns": (repo / "assets/icon.icns").read_bytes()})
    assert _offenders(repo, base, copied) == []
    tweaked = _commit(repo, {"other/icon.icns": _binary(LIMIT * 2, 7)})
    assert _offenders(repo, copied, tweaked) == [("other/icon.icns", "binary", "M")]


def test_allowlist_exempts_exact_paths_only(repo):
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, {"assets/icon.icns": _binary(LIMIT * 2, 9), "assets/icon-dark.icns": _binary(LIMIT * 2, 8)})
    allow = {"assets/icon.icns", "assets/"}
    assert _offenders(repo, base, head, allow) == [("assets/icon-dark.icns", "binary", "A")]


def test_allowlist_file_ignores_comments_and_blanks(tmp_path):
    path = tmp_path / "allow.txt"
    path.write_text("# why\n\n  apps/a.icns  \n# b.png\n", encoding="utf-8")
    assert cbs.read_allowlist(path) == {"apps/a.icns"}
    assert cbs.read_allowlist(tmp_path / "missing.txt") == set()


def test_working_tree_mode_sees_unstaged_and_untracked_files(repo):
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "assets/small.png").write_bytes(_binary(LIMIT + 5, 4))  # unstaged edit
    (repo / "log.db").write_bytes(_binary(LIMIT + 5, 6))  # untracked
    (repo / "dump.txt").write_bytes(_text(LIMIT + 64))  # untracked text
    (repo / ".gitignore").write_text("ignored.bin\n", encoding="utf-8")
    (repo / "ignored.bin").write_bytes(_binary(LIMIT * 2))
    assert sorted(_offenders(repo, base, None)) == [
        ("assets/small.png", "binary", "M"), ("dump.txt", "new file", "A"), ("log.db", "binary", "A"),
    ]


def test_untracked_copy_of_base_content_adds_no_bytes(repo):
    # Staging the copy would pass (its blob is already in the base); so must leaving it untracked.
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "copies").mkdir()
    (repo / "copies/icon.icns").write_bytes((repo / "assets/icon.icns").read_bytes())
    (repo / "copies/big.lock").write_bytes((repo / "big.lock").read_bytes())
    (repo / "copies/icon-edited.icns").write_bytes(_binary(LIMIT * 2, 11))
    assert _offenders(repo, base, None) == [("copies/icon-edited.icns", "binary", "A")]
    _git(repo, "add", "--", "copies/icon.icns", "copies/big.lock")
    assert _offenders(repo, base, None) == [("copies/icon-edited.icns", "binary", "A")]


def test_cli_with_head_reads_the_allowlist_from_head_not_the_checkout(repo):
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, {".github/blob-size-allowlist.txt": b"pr-assets/kept.png\n",
                          "pr-assets/kept.png": _binary(LIMIT * 2, 1), "pr-assets/shot.png": _binary(LIMIT * 2, 2)})
    # The checkout now exempts the other file and not the one head exempts.
    _git(repo, "checkout", "-q", base)
    (repo / ".github").mkdir(exist_ok=True)
    (repo / ".github/blob-size-allowlist.txt").write_text("pr-assets/shot.png\n", encoding="utf-8")

    def run(*extra: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(SCRIPT), "--base", base, *extra, "--limit", str(LIMIT)],
                              cwd=repo, env=_env(), capture_output=True, text=True, encoding="utf-8",
                              timeout=60, check=False)

    proc = run("--head", head)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "::error file=pr-assets/shot.png::" in proc.stdout
    assert "pr-assets/kept.png" not in proc.stdout
    # A head without the list exempts nothing, whatever the checkout says.
    bare = _commit(repo, {"pr-assets/shot.png": _binary(LIMIT * 2, 2)})
    assert run("--head", bare).returncode == 1
    # Working-tree mode keeps reading the list from disk.
    assert run().returncode == 0


def test_cli_fails_with_fix_text_and_writes_step_summary(repo, tmp_path):
    base = _git(repo, "rev-parse", "HEAD")
    (repo / ".github").mkdir()
    (repo / ".github/blob-size-allowlist.txt").write_text("assets/icon.icns\n", encoding="utf-8")
    head = _commit(repo, {".github/blob-size-allowlist.txt": (repo / ".github/blob-size-allowlist.txt").read_bytes(),
                          "assets/icon.icns": _binary(LIMIT * 2, 9), "pr-assets/shot.png": _binary(LIMIT * 3)})
    summary = tmp_path / "summary.md"
    env = {**_env(), "GITHUB_STEP_SUMMARY": str(summary)}
    proc = subprocess.run([sys.executable, str(SCRIPT), "--base", base, "--head", head, "--limit", str(LIMIT)],
                          cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "::error file=pr-assets/shot.png::" in proc.stdout
    assert "assets/icon.icns" not in proc.stdout  # allowlisted by the head's own list
    assert "blob-size-allowlist.txt" in proc.stdout and "python scripts/ci/check_blob_size.py" in proc.stdout
    assert "| `pr-assets/shot.png` | binary | A |" in summary.read_text(encoding="utf-8")

    clean = subprocess.run([sys.executable, str(SCRIPT), "--base", head, "--head", head], cwd=repo, env=_env(),
                           capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert clean.returncode == 0, clean.stdout + clean.stderr


def test_bad_revision_is_a_usage_error_not_a_pass(repo):
    proc = subprocess.run([sys.executable, str(SCRIPT), "--base", "no-such-rev", "--head", "HEAD"], cwd=repo,
                          env=_env(), capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert proc.returncode == 2 and "check_blob_size:" in proc.stderr
