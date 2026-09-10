from datetime import UTC, datetime

import pytest

from github_pr_feedback.stack import StackEntry, StackManifest, StackStore


def entry(branch: str, base: str) -> StackEntry:
    return StackEntry(branch, base, f"Title {branch}", "Body")


def test_valid_linear_stack_round_trips(tmp_path):
    manifest = StackManifest(
        "acme/widgets",
        "release-1",
        "stable",
        (entry("codex/one", "stable"), entry("codex/two", "codex/one")),
        datetime(2026, 9, 2, tzinfo=UTC),
    )
    path = StackStore(tmp_path).save(manifest)
    assert StackStore(tmp_path).load("acme/widgets", "release-1") == manifest
    assert path.is_file()


@pytest.mark.parametrize(
    "entries",
    [
        (entry("codex/one", "stable"), entry("codex/one", "stable")),
        (entry("codex/one", "codex/two"), entry("codex/two", "codex/one")),
        (entry("codex/one", "missing"),),
    ],
)
def test_invalid_stack_relationships_are_rejected(entries):
    with pytest.raises(ValueError):
        StackManifest("acme/widgets", "stack", "stable", entries, datetime.now(UTC))


def test_unsafe_stack_identifiers_are_rejected():
    with pytest.raises(ValueError):
        StackManifest("acme/widgets", "../stack", "stable", (entry("codex/one", "stable"),), datetime.now(UTC))
