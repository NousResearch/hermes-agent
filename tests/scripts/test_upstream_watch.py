"""Upstream watch: the parts that decide what a maintainer is told to read first."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "upstream_watch", Path(__file__).resolve().parents[2] / "scripts" / "upstream_watch.py")
watch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(watch)


def _commit(sha, title, *files):
    return {"sha": sha * 40, "title": title, "url": f"https://x.test/{sha}", "files": list(files)}


def test_conflict_list_is_only_files_both_sides_changed():
    both = watch.conflict_files(["tools/a.py", "agent/b.py", "docs/c.md"], {"tools/a.py", "apps/desktop/x.ts"})

    assert both == ["tools/a.py"]


def test_commits_group_under_every_area_they_touch_and_unknown_paths_go_to_other():
    groups = watch.group_by_area([
        _commit("a", "both", "tools/a.py", "gateway/g.py"),
        _commit("b", "odd", "flake.nix"),
        _commit("c", "bare"),
    ])

    assert list(groups)[:2] == ["tools", "gateway"]  # maintainer-priority order, not alphabetical
    assert [c["title"] for c in groups["other"]] == ["odd", "bare"]


def test_report_flags_shared_files_first_and_marks_the_commits_that_touch_them():
    config = {"upstream": "Up/stream", "reviewed_sha": None, "since": "2026-09-26T00:00:00Z"}
    report = watch.render_report(
        config=config,
        commits=[_commit("a", "risky", "tools/a.py"), _commit("b", "safe", "docs/c.md")],
        fork_files={"tools/a.py"}, truncated=False, head_sha="f" * 40)

    assert report.index("read these first") < report.index("## tools")
    assert "risky ⚠️" in report and "safe ⚠️" not in report
    assert "--mark-reviewed " + "f" * 40 in report


def test_no_changes_says_so_and_a_reviewed_sha_replaces_the_date_as_the_starting_point():
    config = {"upstream": "Up/stream", "reviewed_sha": "1234567890abcdef", "since": "2026-09-26T00:00:00Z"}
    report = watch.render_report(config=config, commits=[], fork_files=set(), truncated=False, head_sha=None)

    assert "since commit `1234567890`" in report and "Nothing new to review." in report


def test_fetch_uses_compare_once_a_sha_is_recorded():
    calls = []

    def fake_get(url, token):
        calls.append(url)
        if url.endswith("/commits/main"):
            return {"sha": "9" * 40}
        return {"commits": [{"sha": "a" * 40, "commit": {"message": "t\nbody"}, "html_url": "u"}],
                "files": [{"filename": "tools/a.py"}]}

    commits, truncated, tip = watch.fetch_commits(
        {"upstream": "Up/stream", "reviewed_sha": "abc"}, None, get=fake_get)

    assert any("/compare/abc...main" in c for c in calls) and tip == "9" * 40 and not truncated
    assert commits[0]["title"] == "t" and commits[0]["files"] == ["tools/a.py"]
