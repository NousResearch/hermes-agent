"""The stable release lanes: what a lane set builds, tests, and records in its claim."""
import pytest

from scripts.releases import lanes as lane_set


def test_the_default_builds_everything_but_termux():
    assert lane_set.DEFAULT == set(lane_set.NAMES) - {"termux"}
    assert lane_set.select() == lane_set.DEFAULT


@pytest.mark.parametrize("kwargs,expected", [
    # The request that motivated lanes: only the Windows bundles and the image, no tests.
    ({"only": ["windows", "docker"]}, {"windows", "docker"}),
    ({"with_": ["termux"]}, set(lane_set.NAMES)),
    ({"without": ["macos-arm64", "upgrade"]}, set(lane_set.DEFAULT) - {"macos-arm64", "upgrade"}),
    ({"only": ["windows"], "with_": ["smoke"], "without": ["windows"]}, {"smoke"}),
    ({"only": []}, set()),
])
def test_select_edits_the_default_or_replaces_it(kwargs, expected):
    assert lane_set.select(**kwargs) == expected


def test_select_refuses_unknown_and_contradictory_lanes():
    with pytest.raises(ValueError, match="unknown lane intel"):
        lane_set.select(with_=["intel"])
    # macOS Intel is not built, so it is not a lane a release can opt into.
    with pytest.raises(ValueError, match="unknown lane macos-intel"):
        lane_set.select(with_=["macos-intel"])
    with pytest.raises(ValueError, match="both name ci"):
        lane_set.select(with_=["ci"], without=["ci"])


def test_options_flatten_repeats_and_commas():
    assert lane_set.parse_list(["windows, docker", "ci"]) == ["windows", "docker", "ci"]
    assert lane_set.parse_list(None) is None


def test_every_desktop_target_has_a_build_lane():
    from scripts.releases.desktop_targets import MANIFEST_TARGETS
    desktop = {target for target in lane_set.TARGET_LANE if not target.startswith("termux/")}
    assert desktop == {f"{platform}/{arch}" for platform, arch in MANIFEST_TARGETS}


def test_every_job_names_real_lanes():
    for job, needs in lane_set.JOB_LANES.items():
        assert needs <= {*lane_set.NAMES, *lane_set.DERIVED}, job
    for target, lane in lane_set.TARGET_LANE.items():
        assert lane in lane_set.NAMES, target
    for handoff, lane in lane_set.HANDOFF_LANE.items():
        assert lane in {*lane_set.NAMES, *lane_set.DERIVED}, handoff


def test_a_job_runs_only_when_every_lane_it_needs_is_on():
    windows_only = lane_set.jobs_to_run(["windows", "docker"])
    assert {job for job, runs in windows_only.items() if runs} == {
        "docker", "publish-docker", "candidates-win32-arm64", "candidates-win32-x64",
        "candidates-win32-bundle", "candidate-manifest", "publish-bundles"}
    # A test lane for something that is not built is inert.
    assert not lane_set.jobs_to_run(["upgrade", "windows"])["transitions-darwin-arm64"]
    assert lane_set.jobs_to_run(["upgrade", "macos-arm64"])["transitions-darwin-arm64"]
    # termux-checks needs both the check and the package.
    assert not lane_set.jobs_to_run(["termux-checks"])["termux-checks"]
    assert lane_set.jobs_to_run(["termux-checks", "termux"])["termux-checks"]
    assert not any(lane_set.jobs_to_run([]).values())


def test_each_windows_lane_alone_builds_its_own_package_with_the_shared_jobs():
    shared = {"candidates-win32-arm64", "candidates-win32-x64", "candidates-win32-bundle",
              "candidate-manifest", "publish-bundles"}
    for lanes in (["windows"], ["windows-store"]):
        assert {job for job, runs in lane_set.jobs_to_run(lanes).items() if runs} == shared
    # The signed pipeline's test arms need the signed lane, never the Store one.
    store_only = lane_set.jobs_to_run(["windows-store", "smoke", "upgrade"])
    assert not store_only["transitions-win32"] and not store_only["windows-packaged"]
    assert lane_set.jobs_to_run(["windows", "upgrade"])["transitions-win32"]
    # A Store-only release has a manifest and smokes nothing: its bundle is not an install target.
    assert lane_set.has_bundles(["windows-store"])
    assert lane_set.targets(["windows-store"]) == frozenset()
    assert not lane_set.runs_smokes(["windows-store", "smoke"])
    assert not lane_set.advances_desktop(["windows-store", "macos-arm64"])
    assert lane_set.handoffs(["windows-store"], ("win32-x64", "windows-universal", "termux")) == \
        ("win32-x64", "windows-universal")


def test_flags_round_trip_and_refuse_anything_loose():
    flags = lane_set.flags(["windows", "docker"])
    assert flags["bundles"] is True and flags["macos-arm64"] is False
    assert flags["desktop-channel"] is False
    assert lane_set.flags(lane_set.DEFAULT)["desktop-channel"] is True
    assert lane_set.from_flags(flags) == {"windows", "docker"}
    assert lane_set.flags(["docker"])["bundles"] is False
    for bad in (None, {}, {**flags, "extra": True}, {**flags, "ci": "true"},
                {name: flags[name] for name in flags if name != "ci"}):
        with pytest.raises(ValueError, match="boolean for every lane"):
            lane_set.from_flags(bad)


def test_a_claim_without_a_lanes_revision_keeps_the_store_bundle_it_was_cut_with():
    """rc.2-v0.22.1 recorded ["docker","windows"] when `windows` also built and submitted the Store bundle."""
    legacy = {"lanes": ["docker", "windows"]}
    assert lane_set.claim_lanes(legacy) == {"docker", "windows", "windows-store"}
    current = {**legacy, "lanesRevision": lane_set.LANES_REVISION}
    assert lane_set.claim_lanes(current) == {"docker", "windows"}
    assert lane_set.claim_lanes({"lanes": ["docker"]}) == {"docker"}
    with pytest.raises(ValueError, match="invalid"):
        lane_set.claim_lanes({**legacy, "lanesRevision": 3})


def test_a_pre_lanes_claim_reads_only_while_it_left_macos_intel_out():
    """Published pre-lanes claims carry includeMacosIntel:false; no build can honour true any more."""
    legacy = {"skipBundles": False, "skipTests": False}
    assert lane_set.claim_lanes({**legacy, "includeMacosIntel": False}) == lane_set.DEFAULT
    with pytest.raises(ValueError, match="invalid: it opts into macOS Intel.*no longer built"):
        lane_set.claim_lanes({**legacy, "includeMacosIntel": True})


def test_targets_and_handoffs_follow_the_build_lanes():
    assert lane_set.targets(["windows", "termux"]) == {"windows/x64", "windows/arm64", "termux/aarch64"}
    assert lane_set.handoffs(["windows"], ("win32-x64", "darwin-arm64", "windows-universal", "termux")) == \
        ("win32-x64", "windows-universal")


def test_smokes_run_only_for_a_built_desktop_package():
    assert lane_set.runs_smokes(lane_set.DEFAULT)
    assert not lane_set.runs_smokes(["smoke", "docker", "termux"])
    assert not lane_set.runs_smokes(["windows"])


def test_the_desktop_channel_moves_only_for_the_full_desktop_set():
    assert lane_set.advances_desktop(lane_set.DEFAULT)
    assert not lane_set.advances_desktop(["windows", "docker"])
    assert not lane_set.advances_desktop(["macos-arm64"])
    # A release that moves no channel is finished once public.
    assert lane_set.moves_channels(["docker"]) and lane_set.moves_channels(["windows", "macos-arm64"])
    assert not lane_set.moves_channels(["windows"]) and not lane_set.moves_channels([])
