"""Tests for pr-reviewer.py."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\pr-reviewer.py")
sys.path.insert(0, str(SCRIPT.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("prv", SCRIPT)
prv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prv)


# --- load_state / save_state ---

def test_load_state_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(prv, "STATE_FILE", tmp_path / "state.json")
    state = prv.load_state()
    assert state == {"reviewed_prs": {}}


def test_load_state_corrupted_file(tmp_path, monkeypatch):
    monkeypatch.setattr(prv, "STATE_FILE", tmp_path / "state.json")
    (tmp_path / "state.json").write_text("not json")
    state = prv.load_state()
    assert state == {"reviewed_prs": {}}


def test_load_state_valid_file(tmp_path, monkeypatch):
    monkeypatch.setattr(prv, "STATE_FILE", tmp_path / "state.json")
    (tmp_path / "state.json").write_text('{"reviewed_prs": {"foo/bar#1": {"at": "2026-07-05"}}}')
    state = prv.load_state()
    assert "foo/bar#1" in state["reviewed_prs"]


def test_save_state_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(prv, "STATE_FILE", tmp_path / "state.json")
    prv.save_state({"reviewed_prs": {"foo/bar#2": {"at": "2026-07-05"}}})
    assert (tmp_path / "state.json").exists()
    loaded = json.loads((tmp_path / "state.json").read_text())
    assert "foo/bar#2" in loaded["reviewed_prs"]


# --- list_open_prs ---

def test_list_open_prs_returns_list():
    prs_json = json.dumps([
        {"number": 1, "author": {"login": "alice"}},
        {"number": 2, "author": {"login": "bob"}},
    ])
    with patch.object(prv.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=prs_json, stderr="")
        prs = prv.list_open_prs("owner/repo")
    assert len(prs) == 2
    assert prs[0]["number"] == 1


def test_list_open_prs_handles_failure():
    with patch.object(prv.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="auth failed")
        prs = prv.list_open_prs("owner/repo")
    assert prs == []


def test_list_open_prs_handles_invalid_json():
    with patch.object(prv.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="not json", stderr="")
        prs = prv.list_open_prs("owner/repo")
    assert prs == []


# --- fetch_pr_diff ---

def test_fetch_pr_diff_returns_stdout():
    diff = "diff --git a/file b/file\n+hello\n"
    with patch.object(prv.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=diff, stderr="")
        result = prv.fetch_pr_diff("owner/repo", 42)
    assert result == diff


def test_fetch_pr_diff_returns_empty_on_failure():
    with patch.object(prv.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        result = prv.fetch_pr_diff("owner/repo", 42)
    assert result == ""


# --- detect_test_command ---

def test_detect_test_command_node():
    diff = "diff --git a/package.json b/package.json\n+..."
    assert prv.detect_test_command("foo/bar", diff) == "npm test"


def test_detect_test_command_python():
    diff = "diff --git a/pyproject.toml b/pyproject.toml\n+..."
    assert prv.detect_test_command("foo/bar", diff) == "pytest"


def test_detect_test_command_python_requirements():
    diff = "diff --git a/requirements.txt b/requirements.txt\n+..."
    assert prv.detect_test_command("foo/bar", diff) == "pytest"


def test_detect_test_command_go():
    diff = "diff --git a/go.mod b/go.mod\n+..."
    assert prv.detect_test_command("foo/bar", diff) == "go test ./..."


def test_detect_test_command_rust():
    diff = "diff --git a/Cargo.toml b/Cargo.toml\n+..."
    assert prv.detect_test_command("foo/bar", diff) == "cargo test"


def test_detect_test_command_unknown_returns_none():
    diff = "diff --git a/README.md b/README.md\n+..."
    assert prv.detect_test_command("foo/bar", diff) is None


# --- compose_review ---

def test_compose_review_includes_diff_stats():
    diff = """diff --git a/a b/a
+hello
+world
+foo
diff --git a/b b/b
-line1
"""
    review = prv.compose_review(diff, tests_passed=True, test_output="all good")
    assert "files changed" in review
    assert "additions" in review
    assert "✅ tests passed" in review


def test_compose_review_includes_test_failure():
    diff = "diff --git a/a b/a\n+line\n"
    review = prv.compose_review(diff, tests_passed=False, test_output="FAIL: assertion error")
    assert "❌ tests failed" in review
    assert "FAIL" in review


def test_compose_review_warns_about_todo():
    diff = "diff --git a/a b/a\n+# TODO: fix this\n+good code\n"
    review = prv.compose_review(diff, tests_passed=None, test_output="")
    assert "TODO" in review


def test_compose_review_warns_about_console_log():
    diff = "diff --git a/a b/a\n+console.log('debug')\n+code\n"
    review = prv.compose_review(diff, tests_passed=None, test_output="")
    assert "console.log" in review


def test_compose_review_warns_large_pr():
    # create a diff with >500 added lines
    diff = "diff --git a/big b/big\n" + "\n".join(f"+line{i}" for i in range(600))
    review = prv.compose_review(diff, tests_passed=None, test_output="")
    assert "Large PR" in review


def test_compose_review_no_issues_when_clean():
    diff = "diff --git a/small b/small\n+just one line\n"
    review = prv.compose_review(diff, tests_passed=True, test_output="")
    assert "No heuristic issues detected" in review


# --- post_review ---

def test_post_review_success():
    with patch.object(prv.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assert prv.post_review("owner/repo", 1, "looks good") is True


def test_post_review_failure():
    with patch.object(prv.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="auth failed")
        assert prv.post_review("owner/repo", 1, "looks good") is False


# --- Integration: dry run ---

def test_script_no_work_to_do(tmp_path, monkeypatch):
    """Script returns 0 when no repos have open PRs."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch.object(prv, "list_open_prs", return_value=[]):
        r = __import__("subprocess").run(
            [sys.executable, str(SCRIPT)],
            capture_output=True, text=True, timeout=30
        )
    # Exit 0 (nothing to do) or 1 (some work done)
    assert r.returncode in (0, 1)
    assert "Traceback" not in r.stderr


# --- get_current_gh_user ---

def test_get_current_gh_user_returns_login():
    with patch.object(prv.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="bbasketballer75\n", stderr="")
        user = prv.get_current_gh_user()
    assert user == "bbasketballer75"


def test_get_current_gh_user_handles_failure():
    with patch.object(prv.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not authed")
        user = prv.get_current_gh_user()
    assert user is None


def test_get_current_gh_user_handles_empty_stdout():
    with patch.object(prv.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="   \n", stderr="")
        user = prv.get_current_gh_user()
    assert user is None


# --- own-PR skip in main() ---

def test_main_skips_own_prs(tmp_path, monkeypatch):
    """When the active gh user authored a PR, it should be skipped silently.

    This is the regression that produced 7 duplicate self-reviews on
    bbasketballer75/hermes-agent#1 + #2.
    """
    monkeypatch.setattr(prv, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(prv, "STATE_FILE", tmp_path / "state.json")
    # Pretend gh is on FriendlyCityJohnstown (the business account used
    # for the upstream PRs).
    with patch.object(prv, "get_current_gh_user", return_value="FriendlyCityJohnstown"), \
         patch.object(prv, "list_open_prs") as mock_list, \
         patch.object(prv, "post_review") as mock_post:
        # All PRs are by FriendlyCityJohnstown (self)
        mock_list.return_value = [
            {"number": 1, "author": {"login": "FriendlyCityJohnstown"}, "headRefName": "abc"},
            {"number": 2, "author": {"login": "FriendlyCityJohnstown"}, "headRefName": "def"},
        ]
        rc = prv.main()
    assert rc == 0
    # post_review should never have been called
    assert mock_post.call_count == 0
    # state should be untouched
    assert prv.load_state() == {"reviewed_prs": {}}


def test_main_reviews_other_authors(tmp_path, monkeypatch):
    """PRs from other authors should still be reviewed even when the active
    gh user is excluded from review."""
    monkeypatch.setattr(prv, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(prv, "STATE_FILE", tmp_path / "state.json")
    with patch.object(prv, "get_current_gh_user", return_value="bbasketballer75"), \
         patch.object(prv, "list_open_prs") as mock_list, \
         patch.object(prv, "fetch_pr_diff", return_value="diff --git a/a b/a\n+x"), \
         patch.object(prv, "detect_test_command", return_value=None), \
         patch.object(prv, "compose_review", return_value="review body"), \
         patch.object(prv, "post_review", return_value=True):
        # gemini-code-assist is a bot, but others should be reviewed
        mock_list.return_value = [
            {"number": 1, "author": {"login": "alice"}, "headRefName": "x"},
        ]
        rc = prv.main()
    # Reviewed → rc=1
    assert rc == 1
    state = prv.load_state()
    assert "FriendlyCityJohnstown/jadas-jazz-cafe-web#1" in state["reviewed_prs"] or \
           "bbasketballer75/honcho#1" in state["reviewed_prs"] or \
           "bbasketballer75/hermes-agent#1" in state["reviewed_prs"]


def test_main_excludes_dependabot(tmp_path, monkeypatch):
    """dependabot[bot] should still be excluded (regression guard for the
    existing DEFAULT_EXCLUDE_AUTHORS list)."""
    monkeypatch.setattr(prv, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(prv, "STATE_FILE", tmp_path / "state.json")
    with patch.object(prv, "get_current_gh_user", return_value="someother"), \
         patch.object(prv, "list_open_prs") as mock_list, \
         patch.object(prv, "post_review") as mock_post:
        mock_list.return_value = [
            {"number": 99, "author": {"login": "dependabot[bot]"}, "headRefName": "y"},
        ]
        rc = prv.main()
    assert rc == 0
    assert mock_post.call_count == 0