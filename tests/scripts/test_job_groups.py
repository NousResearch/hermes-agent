"""Behavior contract for the `jobs` input parser (no workflow YAML is read)."""
import pytest

from scripts.releases.job_groups import ALL_JOBS, JOB_GROUPS, parse_jobs, phase_jobs, selects_all


def test_the_default_selects_every_group():
    for raw in (None, ALL_JOBS):
        selected = parse_jobs(raw)
        assert set(selected) == set(JOB_GROUPS)
        assert all(selected.values())


def test_one_group_selects_only_itself():
    selected = parse_jobs("termux")
    assert selected["termux"] is True
    assert all(value is False for group, value in selected.items() if group != "termux")
    selected = parse_jobs("darwin-arm64, win32-bundle")
    assert selected == {**{group: False for group in JOB_GROUPS},
                        "darwin-arm64": True, "win32-bundle": True}


def test_duplicates_unknown_names_and_empty_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        parse_jobs("darwin-arm64,darwin-arm64")
    with pytest.raises(ValueError, match="unknown job group: mac"):
        parse_jobs("mac")
    with pytest.raises(ValueError, match="at least one group"):
        parse_jobs("")
    with pytest.raises(ValueError, match="empty group name"):
        parse_jobs("termux,")
    with pytest.raises(ValueError, match="unknown job group"):
        parse_jobs("termux,smoke-win32")


def test_selects_all_refuses_partial_selections():
    assert selects_all(None) is True
    assert selects_all(ALL_JOBS) is True
    assert selects_all("termux") is False
    assert selects_all("darwin-arm64,darwin-x64,win32-arm64,win32-x64") is False
    with pytest.raises(ValueError):
        selects_all("bogus")


def test_phase_jobs_judge_only_the_selected_groups():
    every = {group: True for group in JOB_GROUPS}
    candidate = phase_jobs(every, "candidate")
    assert candidate == ["validate", "build-darwin-arm64", "smoke-darwin-arm64",
                         "build-darwin-x64", "smoke-darwin-x64",
                         "build-win32-arm64", "smoke-win32-arm64",
                         "build-win32-x64", "smoke-win32-x64",
                         "assemble-win32-bundle", "termux-deb"]
    termux_only = {**{group: False for group in JOB_GROUPS}, "termux": True}
    assert phase_jobs(termux_only, "candidate") == ["validate", "termux-deb"]
    # B4 moved candidate-manifest into stable-release.yml; the desktop phase
    # result judges only the group jobs themselves.
    assert "candidate-manifest" not in candidate
    partial = {**{group: False for group in JOB_GROUPS}, "darwin-arm64": True}
    assert phase_jobs(partial, "candidate") == ["validate", "build-darwin-arm64", "smoke-darwin-arm64"]
    # A claim that skipped tests still judges every build, and no smoke.
    untested = phase_jobs(every, "candidate", skip_tests=True)
    assert untested == [job for job in candidate if not job.startswith("smoke-")]
    # A claim without the signed Windows lane builds the Store packages but smokes only the other groups.
    unsigned = phase_jobs(every, "candidate", skip_signed=True)
    assert unsigned == [job for job in candidate if not job.startswith("smoke-win32-")]
    assert "build-win32-x64" in unsigned and "assemble-win32-bundle" in unsigned
    assert phase_jobs(every, "publish") == ["validate", "stable-publish"]
    with pytest.raises(ValueError, match="Unknown release phase"):
        phase_jobs(every, "promote")


def test_windows_smokes_with_bundle_move_the_msix_smokes_to_the_bundle_call():
    """Stable's per-arch calls then only build, and its bundle call fails on either MSIX smoke."""
    none = {group: False for group in JOB_GROUPS}
    per_arch = phase_jobs({**none, "win32-x64": True}, "candidate", windows_smokes_with_bundle=True)
    assert per_arch == ["validate", "build-win32-x64"]
    bundle = {**none, "win32-bundle": True}
    assert phase_jobs(bundle, "candidate", windows_smokes_with_bundle=True) == [
        "validate", "assemble-win32-bundle", "smoke-win32-arm64", "smoke-win32-x64"]
    # Without the signed lane, or with tests off, the bundle call smokes nothing.
    for skip in ({"skip_signed": True}, {"skip_tests": True}):
        assert phase_jobs(bundle, "candidate", windows_smokes_with_bundle=True, **skip) == [
            "validate", "assemble-win32-bundle"], skip
    # Callers that build everything in one call keep the smokes on the per-arch groups.
    assert phase_jobs(bundle, "candidate") == ["validate", "assemble-win32-bundle"]


@pytest.mark.parametrize("jobs, expected", [(None, "true"), ("termux", "false"),
                                            ("darwin-arm64,darwin-x64", "false")])
def test_admission_emits_all_jobs_only_for_a_full_selection(monkeypatch, capsys, jobs, expected):
    from scripts.releases import job_groups

    if jobs is None:
        monkeypatch.delenv("JOBS", raising=False)
    else:
        monkeypatch.setenv("JOBS", jobs)
    job_groups.main()
    lines = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    assert lines["all-jobs"] == expected
    assert set(lines) == {*job_groups.JOB_GROUPS, "all-jobs"}
