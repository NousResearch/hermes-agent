"""Real tests for the approval-flow metadata stamping helper.

This helper lives in its own module (no hermes-core imports, no
`hermes_cli.config`, no agent.replay_cleanup) precisely so the test
runner can exercise it WITHOUT tripping the conftest home-guard.

The bug these tests pin down:

    The gateway runner used to pass
    ``metadata=ctx._status_thread_metadata`` straight to
    ``adapter.send_exec_approval`` without copying in
    ``approval_data["request_id"]``. The mattermost adapter's
    entry-registration read ``request_id`` from ``prompt.metadata``
    (then from ``prompt.request_id``); with no ``request_id`` stamped,
    every pending card landed in the registry with ``request_id == ""``
    and the tap handler fell through to ``queue.pop(0)`` (FIFO head).
    Two pending approvals in one session meant tapping ✅ on card B
    resolved card A's command.

    AI review follow-up #5863117294 (Finding 3 part B).

These tests use the helper directly — real Python calls, real assertions.
No stubs, no MagicMock of the helper itself. They are pure-functional
tests: the helper has no I/O, no side effects, no hermes-core imports.
"""
from __future__ import annotations

import pytest

from gateway.run_turn_runner_approval_metadata import stamp_request_id_into_metadata


# Bug-class: stamps request_id iff approval_data carries a truthy one.
# These tests would have FAILED before the helper was created — that's
# the whole point of extracting it (the same logic was inline in the
# runner, untestable here).

class TestStampsRequestId:
    """The basic contract: when approval_data has a truthy request_id,
    it ends up in the returned metadata."""

    def test_truthy_request_id_is_stamped(self):
        result = stamp_request_id_into_metadata(
            {"thread_id": "t-1"},
            {"request_id": "req-abc", "command": "ls"},
        )
        assert result == {"thread_id": "t-1", "request_id": "req-abc"}, (
            f"expected request_id stamped into thread_id, got {result!r}"
        )

    def test_existing_request_id_in_metadata_is_overwritten(self):
        """If ctx._status_thread_metadata already carries a stale
        request_id, the approval's id wins (it's the source of truth)."""
        result = stamp_request_id_into_metadata(
            {"thread_id": "t-1", "request_id": "stale-id"},
            {"request_id": "req-current", "command": "ls"},
        )
        assert result["request_id"] == "req-current", (
            "approval_data should overwrite a stale request_id in metadata"
        )
        assert result["thread_id"] == "t-1", "unrelated metadata keys preserved"

    def test_none_metadata_yields_dict_with_only_request_id(self):
        result = stamp_request_id_into_metadata(
            None,
            {"request_id": "req-abc", "command": "ls"},
        )
        assert result == {"request_id": "req-abc"}, (
            f"None metadata must produce a fresh dict with the stamped id, got {result!r}"
        )


class TestDoesNotStampsWhenIdMissing:
    """The honesty contract: when there's no request_id, don't fabricate one.
    The existing adapter already treats request_id="" as "no id"."""

    def test_empty_string_request_id_leaves_metadata_unchanged(self):
        """The original inline guard was `if request_id:` — an empty string
        must NOT overwrite a real value in metadata with ""."""
        input_metadata = {"thread_id": "t-1", "request_id": "preexisting-id"}
        result = stamp_request_id_into_metadata(
            input_metadata,
            {"request_id": "", "command": "ls"},
        )
        assert result == input_metadata, (
            f"empty-string id must not clobber existing metadata; got {result!r}"
        )

    def test_none_request_id_leaves_metadata_unchanged(self):
        input_metadata = {"thread_id": "t-1", "request_id": "preexisting-id"}
        result = stamp_request_id_into_metadata(
            input_metadata,
            {"request_id": None, "command": "ls"},
        )
        assert result == input_metadata, (
            f"None request_id must not clobber; got {result!r}"
        )

    def test_missing_request_id_key_leaves_metadata_unchanged(self):
        input_metadata = {"thread_id": "t-1", "request_id": "preexisting-id"}
        result = stamp_request_id_into_metadata(
            input_metadata,
            {"command": "ls"},  # no request_id key at all
        )
        assert result == input_metadata, (
            f"missing request_id key must not clobber; got {result!r}"
        )

    def test_none_metadata_with_empty_id_returns_empty_dict(self):
        """None metadata + falsy id = empty dict, not {} being skipped silently
        somewhere. Verifies the helper always returns a usable dict shape."""
        result = stamp_request_id_into_metadata(None, {"request_id": ""})
        assert result == {}, (
            f"None metadata + empty id should yield empty dict, got {result!r}"
        )


class TestMutationSafety:
    """The helper MUST NOT mutate the caller's metadata dict.
    The runner passes `ctx._status_thread_metadata` (a cached property
    keyed by chat); mutating it would corrupt that cache across turns."""

    def test_input_metadata_is_not_mutated(self):
        input_metadata = {"thread_id": "t-1"}
        original = dict(input_metadata)  # snapshot for comparison
        stamp_request_id_into_metadata(
            input_metadata,
            {"request_id": "req-abc"},
        )
        assert input_metadata == original, (
            f"helper must not mutate caller's dict; "
            f"expected {original!r}, input now {input_metadata!r}"
        )

    def test_returns_new_dict_object(self):
        """The return value must be a separate dict object. If callers
        retain both, one mutation doesn't bleed into the other."""
        input_metadata = {"thread_id": "t-1"}
        result = stamp_request_id_into_metadata(
            input_metadata,
            {"request_id": "req-abc"},
        )
        assert result is not input_metadata, "helper must return a new dict"

    def test_returns_new_dict_even_when_no_changes(self):
        """Empty-id path also must return a new dict (caller may rely on this)."""
        input_metadata = {"thread_id": "t-1"}
        result = stamp_request_id_into_metadata(
            input_metadata,
            {"request_id": ""},
        )
        assert result is not input_metadata, (
            "helper must always return a new dict, even when nothing was stamped"
        )


class TestKeyIsolation:
    """request_id is the only key this helper stamps. The card-registry
    is key-sensitive (request_id is the routing key); other approval_data
    keys must NOT leak into metadata."""

    def test_other_approval_data_keys_do_not_leak(self):
        """approval_data carries command, description, pattern_key,
        pattern_keys, flags — none of these belong in metadata (the
        adapter is responsible for those, not this helper)."""
        result = stamp_request_id_into_metadata(
            {"thread_id": "t-1"},
            {
                "request_id": "req-abc",
                "command": "ls",
                "description": "list files",
                "pattern_key": "safe_read",
                "pattern_keys": ["safe_read"],
                "flags": {"allow_permanent": True},
            },
        )
        assert result == {"thread_id": "t-1", "request_id": "req-abc"}, (
            f"only request_id should be stamped; got {result!r}"
        )


class TestIntegrationContracts:
    """Cross-checks the helper's contract against the inputs the runner
    actually passes. These pin down the call-site shape so a future
    refactor in run_turn_runner.py can't quietly break the helper
    contract by changing the data the runner feeds it."""

    def test_helper_accepts_typical_runner_input(self):
        """What _approval_notify_sync passes: ctx._status_thread_metadata
        (a dict keyed by thread metadata) and approval_data (the dict
        built in _ApprovalEntry)."""
        # Mimic _ApprovalEntry's shape: a real "approval_data" dict from
        # tools/approval.py::add_pending. We're testing the helper's tolerance,
        # not the approval code — but the values here match what the runner
        # actually builds (search tools/approval.py for `"request_id"`).
        approval_data = {
            "request_id": "550e8400-e29b-41d4-a716-446655440000",
            "command": "rm -rf /tmp/old-backup",
            "description": "delete old backup",
            "pattern_key": "deny_pattern",
            "pattern_keys": ["deny_pattern", "dangerous"],
            "user_id": "u-1",
            "session_key": "agent:main:dm:t-1",
        }
        metadata = {
            "thread_root_id": "post-root-1",
            "thread_position": "queued",
        }
        result = stamp_request_id_into_metadata(metadata, approval_data)
        assert result["request_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert result["thread_root_id"] == "post-root-1"
        assert result["thread_position"] == "queued"
        assert len(result) == 3, (
            f"expected exactly 3 keys, got {sorted(result)}: {result!r}"
        )

    def test_helper_accepts_unicode_request_id(self):
        """request_ids in the wild are UUIDs but real-world systems
        sometimes stuff non-ASCII into them. Helper should not assume."""
        result = stamp_request_id_into_metadata(
            {"thread_id": "t-1"},
            {"request_id": "żółć-⚡-1234"},
        )
        assert result["request_id"] == "żółć-⚡-1234"


@pytest.mark.parametrize(
    "approval_data",
    [
        {"request_id": ""},
        {"request_id": None},
        {"command": "ls"},           # no key
        {"request_id": False},       # explicit falsy
        {"request_id": 0},
    ],
    ids=["empty_str", "none", "missing_key", "false", "zero"],
)
def test_falsy_request_id_is_a_no_op(approval_data):
    """Truthiness check: any falsy request_id is the "no id" signal.
    This is the entire safety contract of `if request_id:` —
    replacing it with `if approval_data.get("request_id") is not None:`
    would BREAK the contract (a future approval_data with explicit
    request_id=None would silently mint an entry with id=None, breaking
    the mattermost card-registry dict-key type)."""
    input_metadata = {"thread_id": "t-1"}
    result = stamp_request_id_into_metadata(input_metadata, approval_data)
    assert result == input_metadata, (
        f"falsy request_id should be a no-op; "
        f"input {input_metadata!r}, got {result!r}, "
        f"approval_data={approval_data!r}"
    )
