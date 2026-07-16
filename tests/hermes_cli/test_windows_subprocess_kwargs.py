"""Coverage for windows_subprocess_kwargs() and its wiring into banner git calls.

The helper forces UTF-8 decoding for text-mode subprocess output on Windows
(where the default cp1252 codepage raises UnicodeDecodeError on UTF-8 bytes)
and is a no-op elsewhere. These tests assert both helper branches and that
every text=True git subprocess in banner.py forwards the encoding kwargs,
while the binary-mode ``git fetch`` does not.
"""
from unittest.mock import MagicMock, patch

from hermes_cli import _subprocess_compat


# --------------------------------------------------------------------------
# Helper result: Windows vs POSIX
# --------------------------------------------------------------------------
def test_windows_subprocess_kwargs_on_windows():
    with patch.object(_subprocess_compat, "IS_WINDOWS", True):
        assert _subprocess_compat.windows_subprocess_kwargs() == {
            "encoding": "utf-8",
            "errors": "replace",
        }


def test_windows_subprocess_kwargs_on_posix():
    with patch.object(_subprocess_compat, "IS_WINDOWS", False):
        assert _subprocess_compat.windows_subprocess_kwargs() == {}


# --------------------------------------------------------------------------
# Banner git calls forward the encoding kwargs on Windows
# --------------------------------------------------------------------------
def _has_utf8_kwargs(kwargs):
    return kwargs.get("encoding") == "utf-8" and kwargs.get("errors") == "replace"


def test_banner_git_state_calls_pass_encoding_kwargs_on_windows(tmp_path):
    """Every text-mode git call in get_git_banner_state() gets UTF-8 kwargs."""
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    results = {
        ("git", "rev-parse", "--short=8", "origin/main"): MagicMock(returncode=0, stdout="b2f477a3\n"),
        ("git", "rev-parse", "--short=8", "HEAD"): MagicMock(returncode=0, stdout="af8aad31\n"),
        ("git", "rev-list", "--count", "origin/main..HEAD"): MagicMock(returncode=0, stdout="3\n"),
    }
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append((tuple(cmd), kwargs))
        return results[tuple(cmd)]

    with patch.object(_subprocess_compat, "IS_WINDOWS", True), \
         patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        state = banner.get_git_banner_state(repo_dir)

    assert state == {"upstream": "b2f477a3", "local": "af8aad31", "ahead": 3}
    assert len(seen) == 3
    for cmd, kwargs in seen:
        assert _has_utf8_kwargs(kwargs), f"missing UTF-8 kwargs on {cmd}"


def test_banner_git_state_calls_omit_encoding_kwargs_on_posix(tmp_path):
    """On POSIX the helper is a no-op — no encoding kwargs are injected."""
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    results = {
        ("git", "rev-parse", "--short=8", "origin/main"): MagicMock(returncode=0, stdout="b2f477a3\n"),
        ("git", "rev-parse", "--short=8", "HEAD"): MagicMock(returncode=0, stdout="af8aad31\n"),
        ("git", "rev-list", "--count", "origin/main..HEAD"): MagicMock(returncode=0, stdout="3\n"),
    }
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append((tuple(cmd), kwargs))
        return results[tuple(cmd)]

    with patch.object(_subprocess_compat, "IS_WINDOWS", False), \
         patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        banner.get_git_banner_state(repo_dir)

    assert len(seen) == 3
    for cmd, kwargs in seen:
        assert "encoding" not in kwargs and "errors" not in kwargs, f"unexpected kwargs on {cmd}"


def test_local_git_check_text_calls_get_kwargs_but_binary_fetch_does_not(tmp_path):
    """_check_via_local_git: text-mode git calls get UTF-8 kwargs; the binary
    ``git fetch`` (no text=True) must NOT receive them."""
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="https://github.com/someone/fork.git\n")
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0, stdout="")
        if cmd == ["git", "rev-list", "--count", "HEAD..origin/main"]:
            return MagicMock(returncode=0, stdout="4\n")
        raise AssertionError(f"unexpected git command: {cmd!r}")

    seen = []

    def recording_run(cmd, **kwargs):
        seen.append((tuple(cmd), kwargs))
        return fake_run(cmd, **kwargs)

    with patch.object(_subprocess_compat, "IS_WINDOWS", True), \
         patch("hermes_cli.banner.subprocess.run", side_effect=recording_run):
        behind = banner._check_via_local_git(repo_dir)

    assert behind == 4
    for cmd, kwargs in seen:
        if cmd[:2] == ("git", "fetch"):
            assert not _has_utf8_kwargs(kwargs), "binary git fetch must not force text decoding"
        else:
            assert _has_utf8_kwargs(kwargs), f"missing UTF-8 kwargs on {cmd}"


def test_check_via_rev_and_release_tag_get_kwargs_on_windows(tmp_path):
    """_check_via_rev (ls-remote) and get_latest_release_tag (describe) both
    forward the encoding kwargs on Windows."""
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    seen = []

    def recording_run(cmd, **kwargs):
        seen.append((tuple(cmd), kwargs))
        if cmd[:2] == ["git", "ls-remote"]:
            return MagicMock(returncode=0, stdout="local-sha\trefs/heads/main\n")
        if cmd[:2] == ["git", "describe"]:
            return MagicMock(returncode=0, stdout="v0.13.0\n")
        raise AssertionError(f"unexpected git command: {cmd!r}")

    banner._latest_release_cache = None
    with patch.object(_subprocess_compat, "IS_WINDOWS", True), \
         patch("hermes_cli.banner.subprocess.run", side_effect=recording_run):
        banner._check_via_rev("local-sha")
        banner.get_latest_release_tag(repo_dir)

    assert len(seen) == 2
    for cmd, kwargs in seen:
        assert _has_utf8_kwargs(kwargs), f"missing UTF-8 kwargs on {cmd}"
