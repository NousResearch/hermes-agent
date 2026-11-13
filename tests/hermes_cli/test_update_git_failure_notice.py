"""A ⚠ notice for a git step that failed names git's cause, not its trailing advice (#154940).

git prints the cause first and sets its follow-up advice apart with a blank line. A stale
``.git/shallow.lock`` is the measured shape: the ``fatal:`` line carries the lock's path, the advice
ends in "remove the file manually to continue." — so a notice built from the last line told users to
remove a file it never identified.
"""

from __future__ import annotations

import subprocess

import pytest

LOCK_PATH = "/Users/me/.hermes/hermes-agent/.git/shallow.lock"

# git 2.x on macOS, as the reporter saw it: cause, blank line, five lines of advice.
STALE_LOCK_STDERR = (
    f"fatal: Unable to create '{LOCK_PATH}': File exists.\n"
    "\n"
    "Another git process seems to be running in this repository, e.g.\n"
    "an editor opened by 'git commit'. Please make sure all processes\n"
    "are terminated then try again. If it still fails, a git process\n"
    "may have crashed in this repository earlier:\n"
    "remove the file manually to continue.\n"
)

# Captured from a localized git (LANG=zh_CN) on this host: both the ``fatal:`` prefix and the advice
# are translated, so the stanza break — not an English word — has to be what the helper keys on.
LOCALIZED_LOCK_STDERR = (
    "\u81f4\u547d\u9519\u8bef\uff1a\u65e0\u6cd5\u521b\u5efa "
    "'/tmp/work/.git/shallow.lock'\uff1a\u6587\u4ef6\u5df2\u5b58\u5728\u3002\n"
    "\n"
    "\u53e6\u4e00 git \u8fdb\u7a0b\u4f3c\u4e4e\u6b63\u5728\u6b64\u4ed3\u5e93\u4e2d\u8fd0\u884c\uff0c"
    "\u6216\u8005\u9501\u6587\u4ef6\u53ef\u80fd\u5df2\u8fc7\u671f\n"
)


def _git_failure_line():
    """Imported inside the test: on a tree without the helper this is the assertion, not a collection
    error that hides which behaviour is missing."""
    from hermes_cli._subprocess_compat import git_failure_line

    return git_failure_line


def _failed_git(stderr: str) -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(128, ["git", "fetch"], output="", stderr=stderr)


def test_the_cause_above_gits_advice_is_the_line_that_names_the_file():
    git_failure_line = _git_failure_line()

    detail = git_failure_line(_failed_git(STALE_LOCK_STDERR))

    assert LOCK_PATH in detail
    assert "File exists" in detail
    assert "remove the file manually to continue." not in detail


def test_a_localized_git_reports_its_cause_too():
    git_failure_line = _git_failure_line()

    detail = git_failure_line(_failed_git(LOCALIZED_LOCK_STDERR))

    assert "shallow.lock" in detail
    # The translated advice paragraph is the half that names nothing.
    assert detail == LOCALIZED_LOCK_STDERR.splitlines()[0].strip()


def test_stderr_without_a_stanza_break_still_reports_its_last_line():
    # A fetch failure prints progress and then one fatal line; nothing here changes that answer.
    git_failure_line = _git_failure_line()
    stderr = (
        "remote: Enumerating objects: 12, done.\n"
        "remote: Total 12 (delta 0), reused 0 (delta 0)\n"
        "fatal: unable to access 'https://example.invalid/r.git/': Could not resolve host\n"
    )

    assert git_failure_line(_failed_git(stderr)) == (
        "fatal: unable to access 'https://example.invalid/r.git/': Could not resolve host")


def test_an_exception_without_stderr_still_names_something():
    git_failure_line = _git_failure_line()

    assert git_failure_line(OSError("fork failed")) == "fork failed"
    assert git_failure_line(OSError()) == "OSError"  # nothing to quote: the type is the answer


def test_the_release_history_notice_names_the_lock_the_refresh_could_not_take(tmp_path, monkeypatch, capsys):
    from hermes_cli import update_cmd, update_cmd_maint as maint
    from hermes_cli import update_receipt
    from hermes_cli import update_windows_acl

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(update_windows_acl, "repair_windows_home_acl", lambda: None)
    for name in ("_verify_and_restore_state_dbs_post_update", "_invalidate_live_plugin_catalog_caches",
                 "_report_catalog_plugin_updates", "_print_bundled_skills_sync_report",
                 "_sync_profiles_after_update", "_print_post_update_notices_and_self_heals"):
        monkeypatch.setattr(maint, name, lambda name=name: None)
    monkeypatch.setattr(maint, "_print_verified_update_completion", lambda message: True)
    monkeypatch.setattr("hermes_cli.model_catalog.seed_cache_from_checkout", lambda root: False)
    monkeypatch.setattr("hermes_cli.git_commit_graph.spawn_commit_graph_refresh", lambda root: None)
    monkeypatch.setattr(update_cmd, "_check_and_apply_config_migration", lambda **kwargs: None)

    def refused_refresh(*args, **kwargs):
        raise _failed_git(STALE_LOCK_STDERR)

    monkeypatch.setattr("hermes_cli.gitlock.fetch_full_commit_graph", refused_refresh)
    update_receipt.begin_update_receipt()
    try:
        maint._run_post_update_maintenance(
            assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
            had_desktop_app_before_update=False, pre_update_version=None,
        )
    finally:
        update_receipt.finalize_update_receipt("success")

    out = capsys.readouterr().out
    notice = next(line for line in out.splitlines() if "Could not refresh release history" in line)
    assert LOCK_PATH in notice
    assert "remove the file manually to continue." not in notice


def test_the_release_tag_notice_names_the_lock_too(tmp_path, monkeypatch, capsys):
    from hermes_cli import update_cmd_check

    calls = []

    def fake_git(git_cmd, root, args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, stdout="", stderr=STALE_LOCK_STDERR)

    monkeypatch.setattr(update_cmd_check, "_git", fake_git)
    update_cmd_check.fetch_release_tag(["git"], tmp_path, {"release_tag": "v0.23.1"})

    out = capsys.readouterr().out
    assert calls[0][0] == "rev-parse"
    assert f"Could not fetch the v0.23.1 tag (fatal: Unable to create '{LOCK_PATH}': File exists.)" in out


def test_remote_stderr_paragraph_cannot_hide_the_local_lock_refusal():
    """A remote banner can precede git's own refusal and subsequent advice."""
    stderr = "remote: Server-side notice before fetch failure\n\n" + STALE_LOCK_STDERR
    detail = _git_failure_line()(_failed_git(stderr))
    assert detail == f"fatal: Unable to create '{LOCK_PATH}': File exists."
