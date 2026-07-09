"""Tests for pr-mirror.py."""
import sys
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\pr-mirror.py")
sys.path.insert(0, str(SCRIPT.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("pm", SCRIPT)
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)


# --- find_repos ---

def test_find_repos_empty_dir(tmp_path):
    """Empty dir returns empty list."""
    result = pm.find_repos(tmp_path)
    assert result == []


def test_find_repos_nonexistent_dir(tmp_path):
    """Nonexistent dir returns empty list."""
    nonexistent = tmp_path / "nope"
    result = pm.find_repos(nonexistent)
    assert result == []


def test_find_repos_finds_git_dirs(tmp_path):
    """Finds .git directories under root."""
    (tmp_path / "repo1" / ".git").mkdir(parents=True)
    (tmp_path / "repo2" / ".git").mkdir(parents=True)
    (tmp_path / "not_a_repo").mkdir()  # no .git
    repos = pm.find_repos(tmp_path)
    assert len(repos) == 2


# --- get_remotes ---

def test_get_remotes_parses_standard_output(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    wmic_output = """origin\thttps://github.com/upstream/repo.git (fetch)
upstream\thttps://github.com/upstream/repo.git (fetch)
origin\thttps://github.com/upstream/repo.git (push)
"""
    with patch.object(pm.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=wmic_output, stderr="")
        remotes = pm.get_remotes(repo)
    assert "origin" in remotes
    assert "upstream" in remotes


def test_get_remotes_returns_empty_on_failure(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    with patch.object(pm.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        remotes = pm.get_remotes(repo)
    assert remotes == {}


# --- identify_upstream_and_fork ---

def test_identify_upstream_and_fork_basic():
    remotes = {
        "origin": "https://github.com/upstream/repo.git",
        "fork": "https://github.com/bbasketballer75/repo.git",
    }
    upstream, fork = pm.identify_upstream_and_fork(remotes)
    assert upstream == "origin"
    assert fork == "fork"


def test_identify_upstream_and_fork_friendlycity():
    remotes = {
        "origin": "https://github.com/FriendlyCityJohnstown/jadas-jazz-cafe-web.git",
        "upstream": "https://github.com/upstream/jadas-jazz-cafe-web.git",
    }
    upstream, fork = pm.identify_upstream_and_fork(remotes)
    assert upstream == "upstream"
    assert fork == "origin"


def test_identify_upstream_and_fork_no_fork():
    """No fork remote returns None for fork."""
    remotes = {
        "origin": "https://github.com/upstream/repo.git",
    }
    upstream, fork = pm.identify_upstream_and_fork(remotes)
    assert upstream == "origin"
    assert fork is None


def test_identify_upstream_and_fork_no_upstream():
    """No upstream returns None for upstream."""
    remotes = {
        "origin": "https://github.com/bbasketballer75/repo.git",
    }
    upstream, fork = pm.identify_upstream_and_fork(remotes)
    assert upstream is None
    assert fork == "origin"


# --- is_behind ---

def test_is_behind_true():
    """Output '0\t5' means 0 ahead, 5 behind."""
    with patch.object(pm.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="0\t5\n", stderr="")
        assert pm.is_behind(Path("dummy"), "origin", "main") is True


def test_is_behind_false():
    """Output '0\t0' means 0 behind."""
    with patch.object(pm.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="0\t0\n", stderr="")
        assert pm.is_behind(Path("dummy"), "origin", "main") is False


def test_is_behind_ahead():
    """Output '3\t0' means 3 ahead, 0 behind."""
    with patch.object(pm.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="3\t0\n", stderr="")
        assert pm.is_behind(Path("dummy"), "origin", "main") is False


def test_is_behind_invalid_output():
    """Invalid output returns False (not behind)."""
    with patch.object(pm.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="garbage", stderr="")
        assert pm.is_behind(Path("dummy"), "origin", "main") is False


# --- sync_repo ---

def test_sync_repo_local_branch_missing(tmp_path):
    """Returns False if local branch doesn't exist."""
    repo = tmp_path / "r"
    repo.mkdir()
    with patch.object(pm.subprocess, "run") as mock_run:
        # git rev-parse --verify main returns non-zero
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not found")
        ok, msg = pm.sync_repo(repo, "upstream", "fork", "main")
    assert ok is False
    assert "not found" in msg


def test_sync_repo_up_to_date(tmp_path):
    """Returns True with 'up to date' if not behind."""
    repo = tmp_path / "r"
    repo.mkdir()
    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get('args', [])
        if "rev-parse --verify" in cmd:
            return MagicMock(returncode=0, stdout="abc", stderr="")
        if "fetch" in cmd:
            return MagicMock(returncode=0, stdout="", stderr="")
        if "rev-list" in cmd:
            return MagicMock(returncode=0, stdout="0\t0", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(pm.subprocess, "run", side_effect=fake_run):
        ok, msg = pm.sync_repo(repo, "upstream", "fork", "main")
    assert ok is True
    assert msg == "up to date"


def test_sync_repo_local_changes_block(tmp_path):
    """Returns False if there are uncommitted local changes."""
    repo = tmp_path / "r"
    repo.mkdir()
    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get('args', [])
        if "rev-parse --verify" in cmd:
            return MagicMock(returncode=0, stdout="abc", stderr="")
        if "fetch" in cmd:
            return MagicMock(returncode=0, stdout="", stderr="")
        if "rev-list" in cmd:
            # Indicate local is behind (3 commits) so we proceed past is_behind
            return MagicMock(returncode=0, stdout="0\t3", stderr="")
        if "--porcelain" in cmd:  # match the actual arg "--porcelain"
            return MagicMock(returncode=0, stdout="M file.txt\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(pm.subprocess, "run", side_effect=fake_run):
        ok, msg = pm.sync_repo(repo, "upstream", "fork", "main")
    assert ok is False
    assert "local changes" in msg


def test_sync_repo_rebase_failure_aborts(tmp_path):
    """If rebase fails, runs git rebase --abort to clean up."""
    repo = tmp_path / "r"
    repo.mkdir()
    call_log = []
    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get('args', [])
        call_log.append(cmd)
        if "rev-parse --verify" in cmd:
            return MagicMock(returncode=0, stdout="abc", stderr="")
        if "fetch" in cmd:
            return MagicMock(returncode=0, stdout="", stderr="")
        if "status --porcelain" in cmd:
            return MagicMock(returncode=0, stdout="", stderr="")
        if "rev-list" in cmd:
            # Indicate local is behind so we proceed past is_behind
            return MagicMock(returncode=0, stdout="0\t3", stderr="")
        if "rebase" in cmd and "--abort" not in cmd:
            return MagicMock(returncode=1, stdout="", stderr="conflict")
        if "rebase --abort" in cmd:
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(pm.subprocess, "run", side_effect=fake_run):
        ok, msg = pm.sync_repo(repo, "upstream", "fork", "main")
    assert ok is False
    assert "rebase failed" in msg
    # Verify rebase --abort was called
    assert any("rebase" in cmd and "--abort" in cmd for cmd in call_log)


def test_sync_repo_full_success(tmp_path):
    """Full happy path: fetch, rebase, push."""
    repo = tmp_path / "r"
    repo.mkdir()
    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get('args', [])
        if "rev-parse --verify" in cmd:
            return MagicMock(returncode=0, stdout="abc", stderr="")
        if "fetch" in cmd:
            return MagicMock(returncode=0, stdout="", stderr="")
        if "status --porcelain" in cmd:
            return MagicMock(returncode=0, stdout="", stderr="")
        if "rev-list" in cmd:
            return MagicMock(returncode=0, stdout="0\t3", stderr="")
        if "rebase" in cmd and "--abort" not in cmd:
            return MagicMock(returncode=0, stdout="", stderr="")
        if "push" in cmd:
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(pm.subprocess, "run", side_effect=fake_run):
        ok, msg = pm.sync_repo(repo, "upstream", "fork", "main")
    assert ok is True
    assert msg == "synced"


# --- Integration: script --help ---

def test_script_help():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=10
    )
    assert r.returncode == 0


# --- Integration: dry run with no repos ---

def test_script_runs_with_no_repos(tmp_path, monkeypatch):
    """Script should run cleanly with no repos to process."""
    # Set HERMES_HOME to a temp dir so logs go there
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    r = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, timeout=30
    )
    # Exit 0 (nothing to do) or 1 (at least one repo synced) - both ok
    assert r.returncode in (0, 1)
    assert "Traceback" not in r.stderr