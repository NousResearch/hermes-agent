"""Tests for stale codex_reasoning_items pruning during compaction (#71058).

Salvaged from PR #71077 (@webtecnica) with two correctness fixes:
the prune boundary is the last USER message (a Codex turn spans multiple
assistant messages whose reasoning items must replay together), and the
newest native compaction checkpoint (type="compaction") is exempt because
it carries already-pruned history, not per-turn reasoning.  Checkpoints a
newer carrier shadows are pruned: the wire builder discards them anyway.
"""

from agent.context_compressor import (
    _STALE_REPLAY_PRUNE_KEYS,
    _prune_stale_reasoning_replay,
)


def _reasoning(item_id="rs_1"):
    return {"type": "reasoning", "encrypted_content": "blob-" + item_id, "id": item_id}


def _compaction():
    return {"type": "compaction", "encrypted_content": "checkpoint-blob"}


def test_prior_turn_reasoning_items_are_pruned():
    messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "a1", "codex_reasoning_items": [_reasoning("rs_a")]},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "a2", "codex_reasoning_items": [_reasoning("rs_b")]},
    ]
    pruned = _prune_stale_reasoning_replay(messages)
    assert pruned == 1
    assert "codex_reasoning_items" not in messages[1]
    # Active turn (after last user message) keeps its items.
    assert messages[3]["codex_reasoning_items"] == [_reasoning("rs_b")]


def test_multi_message_active_turn_chain_is_never_pruned():
    """A Codex turn spans assistant+tool_calls -> tool -> assistant; ALL of
    the active turn's reasoning items must survive (the #71077 review gap)."""
    messages = [
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old", "codex_reasoning_items": [_reasoning("rs_old")]},
        {"role": "user", "content": "active turn"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
            "codex_reasoning_items": [_reasoning("rs_chain1")],
        },
        {"role": "tool", "content": "result", "tool_call_id": "c1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
            "codex_reasoning_items": [_reasoning("rs_chain2")],
        },
        {"role": "tool", "content": "result", "tool_call_id": "c2"},
        {"role": "assistant", "content": "done", "codex_reasoning_items": [_reasoning("rs_final")]},
    ]
    pruned = _prune_stale_reasoning_replay(messages)
    assert pruned == 1  # only the old turn
    assert "codex_reasoning_items" not in messages[1]
    for idx in (3, 5, 7):
        assert messages[idx].get("codex_reasoning_items"), f"active-chain msg {idx} lost its items"


def test_native_compaction_checkpoints_survive_pruning():
    """type="compaction" items are cumulative context carriers — they must
    survive on stale messages even when reasoning items are stripped."""
    messages = [
        {"role": "user", "content": "turn 1"},
        {
            "role": "assistant",
            "content": "a1",
            "codex_reasoning_items": [_compaction(), _reasoning("rs_a")],
        },
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "a2"},
    ]
    pruned = _prune_stale_reasoning_replay(messages)
    assert pruned == 1
    # Reasoning stripped, checkpoint kept.
    assert messages[1]["codex_reasoning_items"] == [_compaction()]


def test_checkpoint_only_sidecar_untouched_and_uncounted():
    messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "a1", "codex_reasoning_items": [_compaction()]},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "a2"},
    ]
    pruned = _prune_stale_reasoning_replay(messages)
    assert pruned == 0
    assert messages[1]["codex_reasoning_items"] == [_compaction()]


def test_no_user_boundary_prunes_nothing():
    messages = [
        {"role": "assistant", "content": "a1", "codex_reasoning_items": [_reasoning("rs_a")]},
        {"role": "assistant", "content": "a2", "codex_reasoning_items": [_reasoning("rs_b")]},
    ]
    assert _prune_stale_reasoning_replay(messages) == 0
    assert messages[0]["codex_reasoning_items"]
    assert messages[1]["codex_reasoning_items"]


def test_non_codex_messages_untouched():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "plain"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "plain2"},
    ]
    assert _prune_stale_reasoning_replay(messages) == 0
    assert messages == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "plain"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "plain2"},
    ]


def test_prune_keys_contract():
    """codex_message_items are replayed for prefix-cache continuity and must
    NOT be in the prune set; the prune targets reasoning blobs only."""
    assert "codex_reasoning_items" in _STALE_REPLAY_PRUNE_KEYS
    assert "codex_message_items" not in _STALE_REPLAY_PRUNE_KEYS


class TestInterimMergePreservesCheckpoints:
    """Sibling site: the Codex incomplete-continuation dedup path must not
    drop checkpoints when overwriting a visually-duplicate interim message."""

    def test_prior_checkpoint_survives_overwrite(self):
        from agent.native_compaction import merge_interim_reasoning_items

        prior = [_compaction(), _reasoning("rs_old")]
        newer = [_reasoning("rs_new")]
        merged = merge_interim_reasoning_items(prior, newer)
        assert _compaction() in merged
        assert _reasoning("rs_new") in merged
        assert _reasoning("rs_old") not in merged  # newer reasoning wins

    def test_newer_checkpoint_wins_outright(self):
        from agent.native_compaction import merge_interim_reasoning_items

        prior = [{"type": "compaction", "encrypted_content": "old-ckpt"}]
        newer = [{"type": "compaction", "encrypted_content": "new-ckpt"}, _reasoning("rs_new")]
        merged = merge_interim_reasoning_items(prior, newer)
        assert merged == newer

    def test_no_prior_checkpoint_is_plain_overwrite(self):
        from agent.native_compaction import merge_interim_reasoning_items

        assert merge_interim_reasoning_items(
            [_reasoning("rs_old")], [_reasoning("rs_new")]
        ) == [_reasoning("rs_new")]

    def test_non_list_inputs_are_safe(self):
        from agent.native_compaction import merge_interim_reasoning_items

        assert merge_interim_reasoning_items(None, None) == []
        assert merge_interim_reasoning_items(None, [_reasoning("r")]) == [_reasoning("r")]
        assert merge_interim_reasoning_items([_compaction()], None) == [_compaction()]


class TestShadowedCheckpointsArePruned:
    """A checkpoint that a newer carrier shadows can never reach a request:
    ``native_compaction.prune_pre_checkpoint_items`` rebuilds every wire
    around the newest checkpoint run and drops each earlier one.  Retaining
    the shadowed copies charged them in full against the tail budget and
    persisted ~120 KB of unreachable ciphertext per assistant row.
    """

    @staticmethod
    def _checkpoint(tag):
        return {"type": "compaction", "encrypted_content": f"ckpt-{tag}"}

    def _transcript(self, turns):
        messages = []
        for t in range(turns):
            messages.append({"role": "user", "content": f"u{t}"})
            messages.append({
                "role": "assistant",
                "content": f"a{t}",
                "codex_reasoning_items": [
                    _reasoning(f"rs_{t}"),
                    self._checkpoint(t),
                ],
            })
        messages.append({"role": "user", "content": "now"})
        return messages

    def test_only_the_newest_carrier_keeps_its_checkpoint(self):
        messages = self._transcript(4)
        _prune_stale_reasoning_replay(messages)
        surviving = [
            (i, item)
            for i, msg in enumerate(messages)
            for item in (msg.get("codex_reasoning_items") or [])
            if item.get("type") == "compaction"
        ]
        assert surviving == [(7, self._checkpoint(3))]

    def test_newest_carrier_inside_the_active_turn_shadows_every_stale_one(self):
        messages = self._transcript(2)
        # Active turn (after the last user message) mints its own checkpoint.
        messages.append({
            "role": "assistant",
            "content": "live",
            "codex_reasoning_items": [self._checkpoint("live")],
        })
        _prune_stale_reasoning_replay(messages)
        assert "codex_reasoning_items" not in messages[1]
        assert "codex_reasoning_items" not in messages[3]
        assert messages[-1]["codex_reasoning_items"] == [self._checkpoint("live")]

    def test_retained_checkpoints_match_what_the_wire_builder_keeps(self):
        """The invariant behind the prune: after it runs, the transcript holds
        exactly the checkpoints ``prune_pre_checkpoint_items`` would have kept."""
        from agent.native_compaction import prune_pre_checkpoint_items

        messages = self._transcript(5)
        _prune_stale_reasoning_replay(messages)

        items = []
        for msg in messages:
            items.extend(dict(i) for i in (msg.get("codex_reasoning_items") or []))
            if msg["role"] == "user":
                items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": msg["content"]}],
                })
        wire = [i for i in prune_pre_checkpoint_items(items) if i.get("type") == "compaction"]
        retained = [
            item
            for msg in messages
            for item in (msg.get("codex_reasoning_items") or [])
            if item.get("type") == "compaction"
        ]
        assert retained == wire
