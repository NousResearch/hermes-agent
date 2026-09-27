"""A ref names a version, and the next version is derived, never read from the tree.

Derivation reads the published stable head, or the seed ``0.21.4`` before one
exists. Attempt refs number attempts within a version and never move the line.
"""
import json

import pytest

from scripts.releases.versioning import derive_next_version, next_attempt, version_from_tag

SEED = "0.21.4"


def test_final_tag_is_its_version():
    assert version_from_tag("v0.21.5") == "0.21.5"


@pytest.mark.parametrize("ref", [
    "v0.21.5-rc",
    "v0.21.4+canary.20260922T001400Z",
    "v2026.9.21",
    "canary-0.21.4+canary.20260922T001400Z",
    "v0.0.7+channel.20260922T001400Z.98765",
    "v0.0.0+commit.20260922T001400Z.98766",
])
def test_non_final_refs_are_not_versions(ref):
    assert version_from_tag(ref) is None


def test_attempt_ref_carries_version_and_attempt():
    from scripts.releases.versioning import parse_attempt_ref

    assert parse_attempt_ref("rc.1-v0.21.5") == ("0.21.5", 1)
    assert parse_attempt_ref("rc.12-v1.0.0") == ("1.0.0", 12)


def test_marker_ref_names_the_attempt_it_clears():
    from scripts.releases.versioning import parse_marker_ref

    assert parse_marker_ref("abandoned-rc.1-v0.21.5") == ("0.21.5", 1)
    assert parse_marker_ref("rc.1-v0.21.5") is None


@pytest.mark.parametrize("ref", [
    "v0.21.5-rc", "v0.21.5-rc.1", "rc.01-v0.21.5", "rc.0-v0.21.5",
    "rc.1-v2026.9.21", "v0.21.5", "abandoned-rc.1-v0.21.5",
    "rc.1-v0.21.5+canary.20260922T001400Z", "rc.1-v0.21", "rc.-v0.21.5",
])
def test_attempt_ref_rejects_other_shapes(ref):
    from scripts.releases.versioning import parse_attempt_ref

    assert parse_attempt_ref(ref) is None


def test_attempt_and_marker_refs_round_trip():
    from scripts.releases.versioning import attempt_ref, marker_ref, parse_attempt_ref, parse_marker_ref

    assert parse_attempt_ref(attempt_ref("0.21.5", 3)) == ("0.21.5", 3)
    assert parse_marker_ref(marker_ref("0.21.5", 3)) == ("0.21.5", 3)


@pytest.mark.parametrize("version, attempt", [("2026.9.21", 1), ("0.21.5", 0), ("0.21", 1)])
def test_attempt_ref_refuses_what_it_could_not_parse(version, attempt):
    from scripts.releases.versioning import attempt_ref

    with pytest.raises(ValueError):
        attempt_ref(version, attempt)


def test_next_attempt_counts_cleared_attempts_and_skips_other_shapes():
    refs = ["rc.1-v0.21.5", "abandoned-rc.1-v0.21.5", "rc.2-v0.21.6",
            "v0.21.5-rc", "v2026.9.21", "abandoned-rc.4-v0.21.5"]
    assert next_attempt("0.21.5", refs) == 2
    assert next_attempt("0.21.6", refs) == 3
    assert next_attempt("0.21.7", refs) == 1


def test_canary_compares_equal_to_its_stable():
    from scripts.releases.semver import compare, is_canary_version
    assert is_canary_version("0.21.4+canary.20260922T001400Z")
    assert compare("0.21.4+canary.20260922T001400Z", "0.21.4") == 0


def test_empty_head_seeds_the_line():
    assert derive_next_version(published=None, bump="patch") == "0.21.5"
    assert derive_next_version(published=None, bump="minor") == "0.22.0"
    assert derive_next_version(published=None, bump="major") == "1.0.0"


def test_published_head_spends_its_version():
    assert derive_next_version(published="0.21.5", bump="patch") == "0.21.6"


def test_canary_base_comes_from_the_validated_protected_stable_head():
    from scripts.releases.versioning import published_stable_version

    class Reader:
        def __init__(self, base, repository):
            assert base == "https://assets.example"
            assert repository == "example/hermes-agent"

        def resolve(self, name):
            assert name == "stable"
            return type("Resolution", (), {
                "terminal": {"policy": "stable-release"},
                "manifest": {"request": {"version": "0.21.7", "commit": "a" * 40}},
            })()

    assert published_stable_version(
        "example/hermes-agent", base_url="https://assets.example", reader_type=Reader,
        run=lambda argv: "[[]]",
    ) == "0.21.7"


def test_a_newer_published_release_outranks_the_protected_head():
    """A release that skipped bundles never moves the R2 head, but it still spends its version."""
    from scripts.releases.versioning import published_stable_identity

    class Reader:
        def __init__(self, base, repository):
            pass

        def resolve(self, name):
            return type("Resolution", (), {
                "terminal": {"policy": "stable-release"},
                "manifest": {"request": {"version": "0.21.7", "commit": "a" * 40}},
            })()

    releases = [
        {"tag_name": "v0.21.8", "draft": False, "prerelease": False},
        # Drafts, prereleases and CalVer labels are not published stable releases.
        {"tag_name": "v0.21.9", "draft": True, "prerelease": False},
        {"tag_name": "v0.21.8+canary.20260924T000000Z", "draft": False, "prerelease": True},
        {"tag_name": "v2026.9.24", "draft": False, "prerelease": False},
    ]

    def run(argv):
        if argv[:4] == ["gh", "api", "--paginate", "--slurp"]:
            return json.dumps([releases])
        assert argv[:2] == ["gh", "api"] and argv[3:] == ["--jq", ".sha"]
        return {"repos/example/hermes-agent/commits/v0.21.8": "b" * 40,
                "repos/example/hermes-agent/commits/v0.21.6": "c" * 40}[argv[2]]

    assert published_stable_identity(
        "example/hermes-agent", base_url="https://assets.example", reader_type=Reader, run=run,
    ) == ("0.21.8", "b" * 40)
    releases[0]["tag_name"] = "v0.21.6"
    assert published_stable_identity(
        "example/hermes-agent", base_url="https://assets.example", reader_type=Reader, run=run,
    ) == ("0.21.7", "a" * 40)


def test_outstanding_attempts_is_the_one_shared_predicate():
    from scripts.releases.versioning import outstanding_attempts

    refs = ["rc.1-v0.21.5", "abandoned-rc.1-v0.21.5", "rc.2-v0.21.5",
            "rc.1-v0.21.6", "v0.21.5", "rc.1-v0.21.7"]
    published = {"0.21.6", "0.21.7"}

    def is_published(version):
        return version in published

    assert outstanding_attempts(refs, is_published) == [("0.21.5", 2, "rc.2-v0.21.5")]
