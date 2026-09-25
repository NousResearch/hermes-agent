"""Update-path git probes must decode as UTF-8 regardless of locale (#122470).

On machines whose locale encoding is not UTF-8 (e.g. cp936),
subprocess.run(..., text=True) without encoding= decodes git output with
the locale codec and raises UnicodeDecodeError on bytes outside it (CJK
branch names, commit messages, ...). Every git probe on the update path must
pass encoding="utf-8", errors="replace", matching
hermes_cli.update_cmd._git_run.

Uses only stdlib + pytest + monkeypatch. No live git, no network.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

CJK_BRANCH = "機能/日本語ブランチ"
HEAD_SHA = "f83a9e9" + "0" * 33


def _install_cp936_git(monkeypatch, respond):
    """Fake subprocess.run that simulates a cp936 locale.

    Any call without explicit encoding="utf-8", errors="replace" raises
    UnicodeDecodeError exactly as a real cp936-locale decode of non-ASCII
    git output would. Returns the list of recorded (cmd, kwargs).
    """
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        if kwargs.get("encoding") != "utf-8" or kwargs.get("errors") != "replace":
            raise UnicodeDecodeError(
                "cp936", b"\x96\x97\x8b\x9f", 0, 1,
                "simulated cp936-locale decode failure",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout=respond(list(cmd)), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _assert_utf8(calls):
    assert calls, "expected at least one git probe"
    for _, kwargs in calls:
        assert kwargs.get("encoding") == "utf-8", kwargs
        assert kwargs.get("errors") == "replace", kwargs


def test_gitlock_merge_base_probe_survives_cp936_locale(monkeypatch, tmp_path):
    from hermes_cli.gitlock import is_ancestor_of_head

    calls = _install_cp936_git(monkeypatch, lambda argv: "")
    assert is_ancestor_of_head(tmp_path, HEAD_SHA) is True
    _assert_utf8(calls)


def test_gitlock_stdout_lines_survives_cp936_locale(monkeypatch, tmp_path):
    from hermes_cli.gitlock import _git_stdout_lines

    calls = _install_cp936_git(monkeypatch, lambda argv: f"main\n{CJK_BRANCH}\n")
    assert _git_stdout_lines(tmp_path, ["branch", "--format=%(refname:short)"]) == [
        "main",
        CJK_BRANCH,
    ]
    _assert_utf8(calls)


def test_boot_read_git_head_survives_cp936_locale(monkeypatch, tmp_path):
    import hermes_cli.boot_bootstrap as boot_bootstrap

    monkeypatch.setattr(boot_bootstrap, "_git_binary", lambda: "git")
    calls = _install_cp936_git(monkeypatch, lambda argv: HEAD_SHA + "\n")
    assert boot_bootstrap.read_git_head(tmp_path) == HEAD_SHA
    _assert_utf8(calls)


def test_fleet_checkout_contains_survives_cp936_locale(monkeypatch, tmp_path):
    import hermes_cli.build_info
    import hermes_cli.update_cmd
    from hermes_cli.update_cmd_fleet_checkout import checkout_contains

    monkeypatch.setattr(hermes_cli.build_info, "get_code_identity", lambda: {})
    monkeypatch.setattr(
        hermes_cli.update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=str(tmp_path))
    )
    calls = _install_cp936_git(monkeypatch, lambda argv: "")
    assert checkout_contains(HEAD_SHA) is True
    _assert_utf8(calls)
