from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    _SUMMARY_END_MARKER,
)
from agent.conversation_compression import (
    _COMPRESSION_CHECKPOINT_END,
    _COMPRESSION_CHECKPOINT_META_PREFIX,
    _COMPRESSION_CHECKPOINT_START,
    _inject_compression_checkpoint,
    build_compression_checkpoint,
    compress_context,
)


SUMMARY = """## Goal
Ship the feature.

## Blocked
- Blocker: CI cannot authenticate | Evidence: HTTP 401 | Failed attempts: retried twice | Artifact state: commit abc123 | Required input: refreshed token | Resume: pytest -q

## Key Decisions
- Decision: use SessionDB state_meta | Rationale: reuse the existing transactional store | Rejected: standalone YAML files | Scope: context compression

## Resolved Questions
None.
"""


def test_build_checkpoint_extracts_structured_current_state():
    checkpoint = build_compression_checkpoint(SUMMARY, session_id="s-1")

    assert checkpoint == {
        "version": 1,
        "session_id": "s-1",
        "decisions": [
            {
                "decision": "use SessionDB state_meta",
                "rationale": "reuse the existing transactional store",
                "rejected_alternatives": "standalone YAML files",
                "scope": "context compression",
            }
        ],
        "blockers": [
            {
                "blocker": "CI cannot authenticate",
                "evidence": "HTTP 401",
                "failed_attempts": "retried twice",
                "artifact_state": "commit abc123",
                "required_input": "refreshed token",
                "resume_action": "pytest -q",
                "status": "unresolved",
            }
        ],
    }


def test_build_checkpoint_drops_placeholders_and_superseded_sections():
    checkpoint = build_compression_checkpoint(
        """## Blocked
None.

## Key Decisions
None recoverable from deterministic fallback.

## Historical Remaining Work
- stale work
""",
        session_id="s-2",
    )

    assert checkpoint["decisions"] == []
    assert checkpoint["blockers"] == []


def test_blocker_pipeline_is_preserved_and_resolved_entries_are_dropped():
    checkpoint = build_compression_checkpoint(
        """## Blocked
- Blocker: CI count is unknown | Evidence: command not run | Failed attempts: none yet | Artifact state: worktree unchanged | Required input: none | Resume: git status --porcelain | wc -l | Status: unresolved
- Blocker: old token failure | Status: resolved | Resume: refresh-token
- [closed] superseded network outage

## Key Decisions
None.
""",
        session_id="s-pipeline",
    )

    assert checkpoint["blockers"] == [
        {
            "blocker": "CI count is unknown",
            "evidence": "command not run",
            "failed_attempts": "none yet",
            "artifact_state": "worktree unchanged",
            "required_input": "none",
            "resume_action": "git status --porcelain | wc -l",
            "status": "unresolved",
        }
    ]


def test_resume_action_preserves_internal_whitespace_exactly():
    command = "printf 'a  b' | sed 's/  / /' | wc -c"
    checkpoint = build_compression_checkpoint(
        "## Blocked\n- Blocker: verify whitespace | Evidence: pending | "
        "Failed attempts: none | Artifact state: unchanged | Required input: none | Resume: "
        + command
        + " | Status: unresolved",
        session_id="s-exact-whitespace",
    )

    assert checkpoint["blockers"][0]["resume_action"] == command


def test_resume_action_preserves_boundary_and_multiline_whitespace_exactly():
    command = "  printf %s value  \n  continuation-arg  "
    checkpoint = build_compression_checkpoint(
        "## Blocked\n- Blocker: verify whitespace | Evidence: pending | "
        "Failed attempts: none | Artifact state: unchanged | Required input: none | Resume: "
        + command
        + "\n\n## Key Decisions\nNone.",
        session_id="s-exact-boundary-whitespace",
    )

    assert checkpoint["blockers"][0]["resume_action"] == command


def test_resume_action_preserves_trailing_whitespace_before_status():
    command = "printf %s value  "
    checkpoint = build_compression_checkpoint(
        "## Blocked\n- Blocker: verify whitespace | Evidence: pending | "
        "Failed attempts: none | Artifact state: unchanged | Required input: none | Resume: "
        + command
        + " | Status: unresolved\n\n## Key Decisions\nNone.",
        session_id="s-exact-status-boundary-whitespace",
    )

    assert checkpoint["blockers"][0]["resume_action"] == command


def test_resume_action_preserves_quoted_label_like_pipeline_text():
    command = "printf '%s\\n' 'x | Status: unresolved | wc -c'"
    checkpoint = build_compression_checkpoint(
        "## Blocked\n- Blocker: verify quoted pipeline | Evidence: pending | "
        "Failed attempts: none | Artifact state: unchanged | Required input: none | Resume: "
        + command,
        session_id="s-exact-label",
    )

    assert checkpoint["blockers"][0]["resume_action"] == command


def test_checkpoint_bounds_fail_closed_without_truncating_state():
    too_many = "\n".join(f"- Decision: current {index}" for index in range(9))
    with pytest.raises(ValueError, match="entry limit"):
        build_compression_checkpoint(
            f"## Key Decisions\n{too_many}\n\n## Blocked\nNone.",
            session_id="bounded",
        )

    with pytest.raises(ValueError, match="size limit"):
        build_compression_checkpoint(
            "## Key Decisions\n- Decision: " + ("x" * 2001),
            session_id="bounded",
        )


def test_free_form_colons_fail_closed_instead_of_creating_partial_schema():
    with pytest.raises(ValueError, match="is not structured"):
        build_compression_checkpoint(
            """## Blocked
- CI remains blocked: token refresh is pending

## Key Decisions
- Keep the API shape: callers already depend on it
""",
            session_id="s-free-form",
        )


@pytest.mark.parametrize(
    "summary, missing",
    [
        (
            "## Key Decisions\n- Decision: ship | Rationale:  | "
            "Rejected: later | Scope: parser",
            "rationale",
        ),
        (
            "## Blocked\n- Blocker: CI | Evidence:  | Failed attempts: none | "
            "Artifact state: clean | Required input: token | Resume: pytest -q",
            "evidence",
        ),
    ],
)
def test_required_semantic_fields_cannot_be_empty(summary, missing):
    with pytest.raises(ValueError, match=rf"missing required {missing}"):
        build_compression_checkpoint(summary, session_id="s-required")


def test_injection_is_inside_boundary_and_replaces_older_checkpoint():
    old = {
        "role": "user",
        "content": (
            f"summary\n\n{_COMPRESSION_CHECKPOINT_START}\nold\n"
            f"{_COMPRESSION_CHECKPOINT_END}\n\n{_SUMMARY_END_MARKER}"
        ),
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }
    checkpoint = build_compression_checkpoint(SUMMARY, session_id="s-3")

    assert _inject_compression_checkpoint([old], checkpoint) is True
    content = old["content"]
    assert content.count(_COMPRESSION_CHECKPOINT_START) == 1
    assert content.count(_COMPRESSION_CHECKPOINT_END) == 1
    assert content.index(_COMPRESSION_CHECKPOINT_END) < content.index(_SUMMARY_END_MARKER)
    assert json.dumps(checkpoint, ensure_ascii=False, sort_keys=True, separators=(",", ":")) in content
    assert "\nold\n" not in content


def test_reinjection_removes_persisted_stale_checkpoint_and_state_sections():
    old_checkpoint = build_compression_checkpoint(
        "## Key Decisions\n- Decision: obsolete choice | Rationale: old | "
        "Rejected: none | Scope: old scope\n\n## Blocked\nNone.",
        session_id="old",
    )
    old_summary = {
        "role": "user",
        "content": (
            "[CONTEXT COMPACTION — REFERENCE ONLY]\n"
            "## Historical Task Snapshot\nkeep this history\n\n"
            "## Blocked\n- Blocker: old blocker\n\n"
            "## Key Decisions\n- Decision: obsolete choice\n\n"
            f"{_COMPRESSION_CHECKPOINT_START}\n```json\n"
            f"{json.dumps(old_checkpoint)}\n```\n{_COMPRESSION_CHECKPOINT_END}\n\n"
            f"{_SUMMARY_END_MARKER}"
        ),
    }
    new_summary = {
        "role": "assistant",
        "content": "## Key Decisions\n- Decision: current choice\n\n" + _SUMMARY_END_MARKER,
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }
    current = build_compression_checkpoint(
        "## Key Decisions\n- Decision: current choice | Rationale: current | "
        "Rejected: old choice | Scope: current scope",
        session_id="current",
    )

    assert _inject_compression_checkpoint([old_summary, new_summary], current) is True
    rendered = "\n".join(str(message["content"]) for message in (old_summary, new_summary))
    assert rendered.count(_COMPRESSION_CHECKPOINT_START) == 1
    assert "obsolete choice" not in rendered
    assert "old blocker" not in rendered
    assert "keep this history" in rendered
    assert "current choice" in rendered
    assert _COMPRESSION_CHECKPOINT_START not in old_summary["content"]
    assert _COMPRESSION_CHECKPOINT_START in new_summary["content"]


def test_reinjection_sanitizes_stale_state_across_multimodal_text_blocks():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    checkpoint_start_cut = len(_COMPRESSION_CHECKPOINT_START) // 2
    checkpoint_end_cut = len(_COMPRESSION_CHECKPOINT_END) // 2
    boundary_cut = len(_SUMMARY_END_MARKER) // 2
    old_summary = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "[CONTEXT COMPACTION — REFERENCE ONLY]\nkeep history\n## Key Deci",
            },
            {"type": "text", "text": "sions\n"},
            {"type": "text", "text": "- Decision: obsolete choice\n"},
            image,
            {
                "type": "text",
                "text": "## Other History\nkeep other\n"
                + _COMPRESSION_CHECKPOINT_START[:checkpoint_start_cut],
            },
            {
                "type": "text",
                "text": _COMPRESSION_CHECKPOINT_START[checkpoint_start_cut:]
                + "stale checkpoint payload"
                + _COMPRESSION_CHECKPOINT_END[:checkpoint_end_cut],
            },
            {
                "type": "text",
                "text": _COMPRESSION_CHECKPOINT_END[checkpoint_end_cut:]
                + "\n"
                + _SUMMARY_END_MARKER,
            },
        ],
    }
    new_summary = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "## Key Decisions\n- Decision: current choice\n\n"
                + _SUMMARY_END_MARKER[:boundary_cut],
            },
            {"type": "text", "text": _SUMMARY_END_MARKER[boundary_cut:]},
        ],
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }
    current = build_compression_checkpoint(
        "## Key Decisions\n- Decision: current choice | Rationale: current | "
        "Rejected: superseded option | Scope: multimodal",
        session_id="current-multimodal",
    )

    assert _inject_compression_checkpoint([old_summary, new_summary], current) is True

    def _render(message):
        content = message["content"]
        blocks = content if isinstance(content, list) else [content]
        return "".join(
            block if isinstance(block, str) else str(block.get("text") or "")
            for block in blocks
        )

    old_rendered = _render(old_summary)
    new_rendered = _render(new_summary)
    rendered = old_rendered + "\n" + new_rendered
    assert rendered.count(_COMPRESSION_CHECKPOINT_START) == 1
    assert "obsolete choice" not in rendered
    assert "stale checkpoint payload" not in rendered
    assert "keep history" in rendered
    assert "keep other" in rendered
    assert "current choice" in rendered
    assert old_summary["content"][3] is image
    assert new_rendered.index(_COMPRESSION_CHECKPOINT_END) < new_rendered.index(
        _SUMMARY_END_MARKER
    )


def test_injection_preserves_multimodal_blocks_and_stays_inside_boundary():
    original_text_block = {"type": "text", "text": "preserved tail"}
    image_block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    summary_block = {
        "type": "text",
        "text": (
            "summary body\n\n"
            f"{_COMPRESSION_CHECKPOINT_START}\nold\n"
            f"{_COMPRESSION_CHECKPOINT_END}\n\n{_SUMMARY_END_MARKER}"
        ),
    }
    message = {
        "role": "user",
        "content": [original_text_block, image_block, summary_block],
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }
    checkpoint = build_compression_checkpoint(SUMMARY, session_id="s-multimodal")

    assert _inject_compression_checkpoint([message], checkpoint) is True
    assert isinstance(message["content"], list)
    assert message["content"][0] == original_text_block
    assert message["content"][1] is image_block
    assert message["content"][2] is not summary_block
    rendered = "\n".join(
        block if isinstance(block, str) else str(block.get("text") or "")
        for block in message["content"]
    )
    assert rendered.count(_COMPRESSION_CHECKPOINT_START) == 1
    assert rendered.count(_COMPRESSION_CHECKPOINT_END) == 1
    assert rendered.index(_COMPRESSION_CHECKPOINT_END) < rendered.index(_SUMMARY_END_MARKER)
    assert "\nold\n" not in rendered

    assert _inject_compression_checkpoint([message], checkpoint) is True
    rendered_again = "\n".join(
        block if isinstance(block, str) else str(block.get("text") or "")
        for block in message["content"]
    )
    assert rendered_again.count(_COMPRESSION_CHECKPOINT_START) == 1


class _FailingCheckpointDB:
    def __init__(self):
        self.calls: list[str] = []

    def try_acquire_compression_lock(self, session_id, holder, *, ttl_seconds):
        self.calls.append("lock")
        return True

    def refresh_compression_lock(self, session_id, holder, *, ttl_seconds):
        return True

    def release_compression_lock(self, session_id, holder):
        self.calls.append("release")

    def get_compression_lock_holder(self, session_id):
        return None

    def archive_and_compact(self, _session_id, _messages, *, state_meta):
        self.calls.append("archive")
        assert set(state_meta) == {_COMPRESSION_CHECKPOINT_META_PREFIX + "s-fail"}
        raise OSError("disk full")


class _BuiltinLikeCompressor:
    compression_count = 4
    _previous_summary = "older summary"
    _last_compress_aborted = False
    _last_compression_made_progress = False
    _last_summary_fallback_used = False
    _last_summary_error = None

    def __init__(self):
        self._anti_thrash = {"attempts": [1]}
        self._nested_tuple = (["before"],)
        self._last_compression_savings_pct = 10.0

    def compress(self, messages, **kwargs):
        self.compression_count += 1
        self._previous_summary = SUMMARY
        self._last_compression_made_progress = True
        self._anti_thrash["attempts"].append(2)
        self._nested_tuple[0].append("mutated")
        self._last_compression_savings_pct = 99.0
        return [
            {
                "role": "user",
                "content": SUMMARY + "\n\n" + _SUMMARY_END_MARKER,
                COMPRESSED_SUMMARY_METADATA_KEY: True,
            }
        ]


def test_checkpoint_write_failure_aborts_before_boundary_and_rolls_back_compressor():
    db = _FailingCheckpointDB()
    compressor = _BuiltinLikeCompressor()
    warnings: list[str] = []
    messages = [
        {"role": "user", "content": "original one"},
        {"role": "assistant", "content": "original two"},
    ]
    agent = SimpleNamespace(
        api_mode="chat_completions",
        context_compressor=compressor,
        _compression_feasibility_checked=True,
        compression_in_place=True,
        session_id="s-fail",
        _session_db=db,
        _memory_manager=None,
        model="test-model",
        platform="cli",
        _cached_system_prompt="existing system",
        _todo_store=SimpleNamespace(format_for_injection=lambda: ""),
        _emit_status=lambda _message: None,
        _emit_warning=warnings.append,
        _invalidate_system_prompt=lambda: None,
        _build_system_prompt=lambda _message: "rebuilt system",
        commit_memory_session=lambda _messages: None,
        _compression_lock_refresh_interval=999.0,
    )

    with patch("agent.context_compressor.ContextCompressor", _BuiltinLikeCompressor):
        returned, system_prompt = compress_context(agent, messages, "system")

    assert returned is messages
    assert system_prompt == "existing system"
    assert compressor.compression_count == 4
    assert compressor._previous_summary == "older summary"
    assert compressor._anti_thrash == {"attempts": [1]}
    assert compressor._nested_tuple == (["before"],)
    assert compressor._last_compression_savings_pct == 10.0
    assert compressor._last_compress_aborted is True
    assert compressor._last_compression_made_progress is False
    assert compressor._last_summary_error is not None
    assert "disk full" in compressor._last_summary_error
    assert db.calls == ["lock", "archive", "release"]
    assert warnings and "No messages were dropped" in warnings[-1]


def test_plugin_compressor_state_is_not_snapshotted_or_deepcopied():
    class ExplosiveState(dict):
        def __deepcopy__(self, memo):
            raise OSError("plugin state must not be copied")

    class PluginCompressor:
        def __init__(self):
            self.state = ExplosiveState(value=1)
            self.called = False

        def compress(self, messages, *, current_tokens):
            self.called = True
            return messages

    compressor = PluginCompressor()
    messages = [{"role": "user", "content": "unchanged"}]
    agent = SimpleNamespace(
        api_mode="chat_completions",
        context_compressor=compressor,
        _compression_feasibility_checked=True,
        compression_in_place=True,
        session_id="s-plugin",
        _session_db=None,
        _memory_manager=None,
        model="test-model",
        platform="cli",
        _cached_system_prompt="existing system",
        _emit_status=lambda _message: None,
        _emit_warning=lambda _message: None,
        _build_system_prompt=lambda _message: "rebuilt system",
    )

    returned, system_prompt = compress_context(agent, messages, "system")

    assert compressor.called is True
    assert returned is messages
    assert system_prompt == "existing system"
    assert compressor.state == {"value": 1}
