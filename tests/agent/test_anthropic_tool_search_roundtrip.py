"""Round-trip tests for Anthropic server-side tool_search blocks.

The tool_search server-side tool produces blocks whose response shape
diverges from the input shape. The API will 400 on resubmit if any
response-only field (``text``, ``citations``, etc.) leaks back, or if
the type discriminator carries a wire-form variant suffix (e.g.
``tool_search_tool_regex_tool_result``) — the input validator only
accepts the bare canonical ``tool_search_tool_result``.

The wire OUTPUT carries the variant suffix; the SDK Pydantic models
strip it to the canonical bare form during parse. Our pipeline
canonicalizes any leaked variant suffix at capture time and again at
request-build time so persisted sessions and SDK-bypass paths all
emit the bare canonical type Anthropic expects.

This module validates every code path that touches these blocks against
the Anthropic SDK's documented input TypedDicts so we don't have to
discover the schema one rejected field at a time.
"""

from __future__ import annotations

import json
from typing import Any, Dict, get_type_hints

import pytest

from agent.anthropic_adapter import (
    _normalize_tool_reference_for_input,
    _canonicalize_tool_search_result_types,
    _normalize_tool_search_result_for_input,
    _normalize_tool_search_result_inner,
    _relocate_orphaned_tool_search_results,
    convert_messages_to_anthropic,
)


def _typed_dict_keys(td_cls) -> set[str]:
    """Return the set of field names declared on a TypedDict class."""
    return set(get_type_hints(td_cls).keys())


# ---------------------------------------------------------------------------
# Schema-derived expected key sets (from the Anthropic SDK TypedDicts)
# ---------------------------------------------------------------------------
try:
    from anthropic.types.beta import (
        beta_tool_reference_block_param,
        beta_tool_search_tool_result_block_param,
        beta_tool_search_tool_result_error_param,
        beta_tool_search_tool_search_result_block_param,
    )

    OUTER_KEYS = _typed_dict_keys(
        beta_tool_search_tool_result_block_param.BetaToolSearchToolResultBlockParam
    )
    INNER_RESULT_KEYS = _typed_dict_keys(
        beta_tool_search_tool_search_result_block_param.BetaToolSearchToolSearchResultBlockParam
    )
    INNER_ERROR_KEYS = _typed_dict_keys(
        beta_tool_search_tool_result_error_param.BetaToolSearchToolResultErrorParam
    )
    REF_KEYS = _typed_dict_keys(
        beta_tool_reference_block_param.BetaToolReferenceBlockParam
    )
    SDK_AVAILABLE = True
except ImportError:  # pragma: no cover — skip if SDK not installed in env
    SDK_AVAILABLE = False
    OUTER_KEYS = INNER_RESULT_KEYS = INNER_ERROR_KEYS = REF_KEYS = set()


pytestmark = pytest.mark.skipif(
    not SDK_AVAILABLE, reason="anthropic SDK not installed"
)


# ---------------------------------------------------------------------------
# Sample response payloads — what Anthropic actually returns
# ---------------------------------------------------------------------------
def _sample_outer_response(
    *, variant: str = "regex", with_text: bool = True, with_citations: bool = True
) -> Dict[str, Any]:
    """Build a synthetic outer block as it appears in a streamed response."""
    block: Dict[str, Any] = {
        # The first error from the live API referenced
        # ``tool_search_tool_regex_tool_result`` — exercise the
        # variant-suffixed form even though the SDK 0.86.0 declares the
        # canonical type. This is what our normalizer must rewrite.
        "type": f"tool_search_tool_{variant}_tool_result",
        "tool_use_id": f"srvtoolu_{variant}_abc123",
        "content": {
            "type": "tool_search_tool_search_result",
            "tool_references": [
                {
                    "type": "tool_reference",
                    "tool_name": "mcp__example__do_thing",
                    # Response carries arbitrary extras; allowlist must drop them.
                    "description": "RESPONSE-ONLY",
                    "input_schema": {"type": "object"},
                },
                {
                    "type": "tool_reference",
                    "tool_name": "mcp__example__other",
                    "extra_meta": "RESPONSE-ONLY",
                },
            ],
        },
    }
    if with_text:
        block["text"] = "Found 2 tools matching the query"
    if with_citations:
        block["citations"] = [{"type": "char_location", "start_char": 0, "end_char": 5}]
    return block


def _sample_outer_response_error() -> Dict[str, Any]:
    return {
        "type": "tool_search_tool_regex_tool_result",
        "tool_use_id": "srvtoolu_err",
        "content": {
            "type": "tool_search_tool_result_error",
            "error_code": "execution_time_exceeded",
            "message": "RESPONSE-ONLY",
        },
        "text": "RESPONSE-ONLY",
    }


# ---------------------------------------------------------------------------
# tool_reference normalization
# ---------------------------------------------------------------------------
class TestNormalizeToolReference:
    def test_strips_response_only_fields(self):
        ref = {
            "type": "tool_reference",
            "tool_name": "mcp__x__y",
            "description": "BAD",
            "input_schema": {"x": 1},
            "rank": 0.95,
        }
        out = _normalize_tool_reference_for_input(ref)
        assert set(out.keys()).issubset(REF_KEYS)
        assert out == {"type": "tool_reference", "tool_name": "mcp__x__y"}

    def test_preserves_cache_control(self):
        ref = {
            "type": "tool_reference",
            "tool_name": "x",
            "cache_control": {"type": "ephemeral"},
        }
        out = _normalize_tool_reference_for_input(ref)
        assert out["cache_control"] == {"type": "ephemeral"}
        assert set(out.keys()).issubset(REF_KEYS)

    def test_handles_string_input_defensively(self):
        out = _normalize_tool_reference_for_input("foo")
        assert out == {"type": "tool_reference", "tool_name": "foo"}

    def test_drops_non_dict_cache_control(self):
        ref = {"type": "tool_reference", "tool_name": "x", "cache_control": "bad"}
        out = _normalize_tool_reference_for_input(ref)
        assert "cache_control" not in out


# ---------------------------------------------------------------------------
# Inner content normalization
# ---------------------------------------------------------------------------
class TestNormalizeInnerSearchResult:
    def test_search_result_strips_extras_and_normalizes_refs(self):
        inner = {
            "type": "tool_search_tool_search_result",
            "text": "BAD",
            "citations": ["BAD"],
            "tool_references": [
                {"type": "tool_reference", "tool_name": "a", "description": "BAD"},
            ],
        }
        out = _normalize_tool_search_result_inner(inner)
        assert set(out.keys()).issubset(INNER_RESULT_KEYS)
        assert out["type"] == "tool_search_tool_search_result"
        assert out["tool_references"] == [{"type": "tool_reference", "tool_name": "a"}]

    def test_error_variant_strips_extras(self):
        inner = {
            "type": "tool_search_tool_result_error",
            "error_code": "unavailable",
            "message": "BAD",
            "details": {"x": 1},
        }
        out = _normalize_tool_search_result_inner(inner)
        assert set(out.keys()).issubset(INNER_ERROR_KEYS)
        assert out == {
            "type": "tool_search_tool_result_error",
            "error_code": "unavailable",
        }

    def test_unknown_inner_type_passes_through(self):
        inner = {"type": "future_unknown_type", "data": 1}
        assert _normalize_tool_search_result_inner(inner) == inner

    def test_non_dict_passes_through(self):
        assert _normalize_tool_search_result_inner("foo") == "foo"


# ---------------------------------------------------------------------------
# Outer block normalization
# ---------------------------------------------------------------------------
class TestNormalizeOuterToolSearchResult:
    @pytest.mark.parametrize("variant", ["regex", "bm25"])
    def test_collapses_variant_suffix_to_canonical_type(self, variant):
        """Anthropic's INPUT validator (verified live 2026-05-07,
        request_id ``req_011Cap2RUgsJp1CVsGAR6LTa``) accepts only the
        bare canonical ``tool_search_tool_result`` type. Variant-suffixed
        forms — which appear on the wire OUTPUT — are rejected with
        "Input tag '<variant>_tool_result' ... does not match any of
        the expected tags". Inversion of an earlier wrong invariant
        held by this codebase: the variant-suffix-must-be-preserved
        claim was a misdiagnosis of an earlier 400 with a different
        cause."""
        sb = _sample_outer_response(variant=variant)
        out = _normalize_tool_search_result_for_input(sb)
        assert out["type"] == "tool_search_tool_result"

    def test_strips_response_only_fields_at_outer_level(self):
        sb = _sample_outer_response(with_text=True, with_citations=True)
        out = _normalize_tool_search_result_for_input(sb)
        assert "text" not in out
        assert "citations" not in out
        # Outer keys must be a subset of the SDK TypedDict's declared keys
        # — type is collapsed to the bare canonical so the result is a
        # legal BetaToolSearchToolResultBlockParam.
        assert set(out.keys()).issubset(OUTER_KEYS)

    def test_preserves_required_fields(self):
        sb = _sample_outer_response()
        out = _normalize_tool_search_result_for_input(sb)
        assert out["tool_use_id"] == sb["tool_use_id"]
        assert "content" in out

    def test_inner_content_is_recursively_normalized(self):
        sb = _sample_outer_response()
        out = _normalize_tool_search_result_for_input(sb)
        inner = out["content"]
        assert set(inner.keys()).issubset(INNER_RESULT_KEYS)
        for ref in inner["tool_references"]:
            assert set(ref.keys()).issubset(REF_KEYS)
            assert "description" not in ref
            assert "input_schema" not in ref

    def test_handles_error_variant_inner_content(self):
        sb = _sample_outer_response_error()
        out = _normalize_tool_search_result_for_input(sb)
        assert set(out.keys()).issubset(OUTER_KEYS)
        assert "text" not in out
        inner = out["content"]
        assert inner == {
            "type": "tool_search_tool_result_error",
            "error_code": "execution_time_exceeded",
        }

    def test_handles_list_wrapped_content(self):
        """Some response shapes wrap inner content in a list — handle either form."""
        sb = {
            "type": "tool_search_tool_regex_tool_result",
            "tool_use_id": "srvtoolu_list",
            "content": [
                {
                    "type": "tool_search_tool_search_result",
                    "tool_references": [
                        {"type": "tool_reference", "tool_name": "a", "description": "BAD"},
                    ],
                }
            ],
        }
        out = _normalize_tool_search_result_for_input(sb)
        assert isinstance(out["content"], list)
        assert out["content"][0]["tool_references"] == [
            {"type": "tool_reference", "tool_name": "a"}
        ]

    def test_preserves_outer_cache_control(self):
        sb = _sample_outer_response()
        sb["cache_control"] = {"type": "ephemeral"}
        out = _normalize_tool_search_result_for_input(sb)
        assert out["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Full message round-trip — convert_messages_to_anthropic
# ---------------------------------------------------------------------------
class TestConvertMessagesRoundTrip:
    def _build_assistant_msg(self, server_tool_blocks):
        return {
            "role": "assistant",
            "content": "Looking that up for you.",
            "server_tool_blocks": server_tool_blocks,
            "tool_calls": [],
        }

    def _walk(self, obj):
        """Yield every dict found anywhere in obj (recursive)."""
        if isinstance(obj, dict):
            yield obj
            for v in obj.values():
                yield from self._walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from self._walk(v)

    def test_full_message_collapses_variant_suffix_to_canonical(self):
        """Inverted from an earlier (wrong) version that asserted the
        variant suffix was preserved. Anthropic accepts only the bare
        canonical ``tool_search_tool_result`` on input."""
        sb = _sample_outer_response(variant="regex")
        msg = self._build_assistant_msg([sb])
        _, out_msgs = convert_messages_to_anthropic(
            [{"role": "user", "content": "hi"}, msg]
        )
        # Find the canonical result block in the output.
        ts_canonical = [
            d for d in self._walk(out_msgs)
            if isinstance(d, dict) and d.get("type") == "tool_search_tool_result"
        ]
        assert len(ts_canonical) == 1
        # And no variant-suffixed forms remain.
        ts_variant = [
            d for d in self._walk(out_msgs)
            if isinstance(d, dict)
            and isinstance(d.get("type"), str)
            and d["type"].startswith("tool_search_tool_")
            and d["type"].endswith("_tool_result")
            and d["type"] != "tool_search_tool_result"
        ]
        assert ts_variant == []

    def test_full_message_strips_all_response_only_fields(self):
        sb = _sample_outer_response(with_text=True, with_citations=True)
        msg = self._build_assistant_msg([sb])
        _, out_msgs = convert_messages_to_anthropic(
            [{"role": "user", "content": "hi"}, msg]
        )
        # No dict in the output should have a response-only field.
        forbidden = {"citations", "input_schema", "rank", "description"}
        for d in self._walk(out_msgs):
            for fld in forbidden:
                assert fld not in d, f"forbidden field {fld!r} in {d}"

    def test_text_field_does_not_leak_onto_tool_search_result(self):
        sb = _sample_outer_response(with_text=True, with_citations=True)
        msg = self._build_assistant_msg([sb])
        _, out_msgs = convert_messages_to_anthropic(
            [{"role": "user", "content": "hi"}, msg]
        )
        for d in self._walk(out_msgs):
            t = d.get("type")
            if isinstance(t, str) and t.startswith("tool_search_tool_") and t.endswith("_tool_result"):
                assert "text" not in d
                assert "citations" not in d
            if d.get("type") == "tool_search_tool_search_result":
                assert "text" not in d
                assert "citations" not in d

    @pytest.mark.parametrize("variant", ["regex", "bm25"])
    def test_variant_suffix_is_collapsed_through_round_trip(self, variant):
        """Inverted: any variant_suffix on the inbound block must be
        gone from the outbound payload — Anthropic only accepts the
        bare canonical type on input. The ``variant`` parameter
        verifies the collapse works regardless of which variant the
        wire delivered."""
        sb = _sample_outer_response(variant=variant)
        msg = self._build_assistant_msg([sb])
        _, out_msgs = convert_messages_to_anthropic(
            [{"role": "user", "content": "hi"}, msg]
        )
        unwanted_type = f"tool_search_tool_{variant}_tool_result"
        types_seen = [
            d.get("type") for d in self._walk(out_msgs)
            if isinstance(d.get("type"), str)
            and d.get("type").startswith("tool_search_tool_")
            and d.get("type").endswith("_tool_result")
        ]
        # No variant suffix anywhere.
        assert unwanted_type not in types_seen
        # The bare canonical IS present.
        assert "tool_search_tool_result" in types_seen

    def test_full_message_outputs_only_sdk_declared_keys(self):
        """Strict allowlist for inner blocks: every emitted block (except
        the outer one whose ``type`` carries a variant suffix not in the
        SDK enum) must have only keys declared by the corresponding
        TypedDict."""
        sb = _sample_outer_response()
        msg = self._build_assistant_msg([sb])
        _, out_msgs = convert_messages_to_anthropic(
            [{"role": "user", "content": "hi"}, msg]
        )
        for d in self._walk(out_msgs):
            t = d.get("type")
            if isinstance(t, str) and t.startswith("tool_search_tool_") and t.endswith("_tool_result"):
                # Outer block: same field set as the SDK declares, just
                # with a variant-suffixed type.
                assert set(d.keys()) - {"type"} <= OUTER_KEYS - {"type"} | {"tool_use_id", "content", "cache_control"}
            elif t == "tool_search_tool_search_result":
                assert set(d.keys()).issubset(INNER_RESULT_KEYS)
            elif t == "tool_search_tool_result_error":
                assert set(d.keys()).issubset(INNER_ERROR_KEYS)
            elif t == "tool_reference":
                assert set(d.keys()).issubset(REF_KEYS)

    def test_error_variant_round_trip_is_clean(self):
        sb = _sample_outer_response_error()
        msg = self._build_assistant_msg([sb])
        _, out_msgs = convert_messages_to_anthropic(
            [{"role": "user", "content": "hi"}, msg]
        )
        for d in self._walk(out_msgs):
            t = d.get("type")
            if isinstance(t, str) and t.startswith("tool_search_tool_") and t.endswith("_tool_result"):
                assert "text" not in d
                assert "citations" not in d
            if t == "tool_search_tool_result_error":
                assert set(d.keys()).issubset(INNER_ERROR_KEYS)
                assert "message" not in d

    def test_assistant_content_remains_text_block(self):
        """When the assistant message has plain text content, the conversion
        should still emit a text block alongside the rebuilt tool_search
        block — not lose the user-facing reply."""
        sb = _sample_outer_response()
        msg = self._build_assistant_msg([sb])
        _, out_msgs = convert_messages_to_anthropic(
            [{"role": "user", "content": "hi"}, msg]
        )
        assistant_msg = out_msgs[-1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        text_blocks = [b for b in assistant_msg["content"] if b.get("type") == "text"]
        assert any("Looking that up" in b.get("text", "") for b in text_blocks)


# ---------------------------------------------------------------------------
# Relocation pass — same-message pairing for server_tool_use ↔ result
# ---------------------------------------------------------------------------
class TestRelocateOrphanedResults:
    def _server_tool_use(self, tu_id: str, name: str = "tool_search_tool_regex"):
        return {
            "type": "server_tool_use",
            "id": tu_id,
            "name": name,
            "input": {"query": "x"},
        }

    def _result(self, tu_id: str, variant: str = "regex"):
        return {
            "type": f"tool_search_tool_{variant}_tool_result",
            "tool_use_id": tu_id,
            "content": {
                "type": "tool_search_tool_search_result",
                "tool_references": [],
            },
        }

    def test_no_orphans_no_change(self):
        """When the tool_use and its result are already in the same message,
        nothing should move."""
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    self._server_tool_use("srvtoolu_a"),
                    self._result("srvtoolu_a"),
                ],
            },
        ]
        before = json.dumps(msgs, sort_keys=True)
        _relocate_orphaned_tool_search_results(msgs)
        assert json.dumps(msgs, sort_keys=True) == before

    def test_relocates_orphan_from_later_message(self):
        """Reproduces the dump: server_tool_use in msg[1], result in msg[3]."""
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "x", "signature": "s"},
                    self._server_tool_use("srvtoolu_X"),
                    {"type": "tool_use", "id": "tu_local", "name": "skills_list", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_local", "content": "ok"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "y", "signature": "s2"},
                    self._result("srvtoolu_X"),
                    {"type": "text", "text": "Done"},
                ],
            },
        ]
        _relocate_orphaned_tool_search_results(msgs)

        # Result should now be in msg[1], right after the server_tool_use.
        msg1_types = [b.get("type") for b in msgs[1]["content"]]
        assert "tool_search_tool_regex_tool_result" in msg1_types
        stu_idx = msg1_types.index("server_tool_use")
        # The block immediately after server_tool_use should be the result.
        assert (
            msgs[1]["content"][stu_idx + 1].get("type")
            == "tool_search_tool_regex_tool_result"
        )
        # And it must NOT remain in msg[3].
        msg3_types = [b.get("type") for b in msgs[3]["content"]]
        assert "tool_search_tool_regex_tool_result" not in msg3_types

    def test_relocation_preserves_block_payload(self):
        result_block = self._result("srvtoolu_K")
        result_block["content"]["tool_references"] = [
            {"type": "tool_reference", "tool_name": "alpha"},
        ]
        msgs = [
            {
                "role": "assistant",
                "content": [self._server_tool_use("srvtoolu_K")],
            },
            {
                "role": "assistant",
                "content": [result_block],
            },
        ]
        _relocate_orphaned_tool_search_results(msgs)
        moved = msgs[0]["content"][1]
        assert moved["tool_use_id"] == "srvtoolu_K"
        assert moved["content"]["tool_references"] == [
            {"type": "tool_reference", "tool_name": "alpha"}
        ]

    def test_handles_multiple_orphans_across_multiple_messages(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    self._server_tool_use("srvtoolu_A"),
                    self._server_tool_use("srvtoolu_B"),
                ],
            },
            {
                "role": "assistant",
                "content": [
                    self._result("srvtoolu_A", variant="regex"),
                ],
            },
            {
                "role": "assistant",
                "content": [
                    self._result("srvtoolu_B", variant="bm25"),
                ],
            },
        ]
        _relocate_orphaned_tool_search_results(msgs)
        msg0_blocks = msgs[0]["content"]
        # Each server_tool_use should be immediately followed by its result.
        types = [b.get("type") for b in msg0_blocks]
        a_idx = next(
            i for i, b in enumerate(msg0_blocks)
            if b.get("type") == "server_tool_use" and b.get("id") == "srvtoolu_A"
        )
        b_idx = next(
            i for i, b in enumerate(msg0_blocks)
            if b.get("type") == "server_tool_use" and b.get("id") == "srvtoolu_B"
        )
        assert msg0_blocks[a_idx + 1]["type"] == "tool_search_tool_regex_tool_result"
        assert msg0_blocks[a_idx + 1]["tool_use_id"] == "srvtoolu_A"
        assert msg0_blocks[b_idx + 1]["type"] == "tool_search_tool_bm25_tool_result"
        assert msg0_blocks[b_idx + 1]["tool_use_id"] == "srvtoolu_B"
        # Source messages should no longer carry the result blocks.
        for src in (msgs[1]["content"], msgs[2]["content"]):
            for b in src:
                t = b.get("type", "")
                assert not (t.startswith("tool_search_tool_") and t.endswith("_tool_result"))

    def test_no_matching_tool_use_leaves_orphan_in_place(self):
        """If the result has no matching server_tool_use in any message,
        leave it where it is rather than dropping it on the floor."""
        msgs = [
            {
                "role": "assistant",
                "content": [self._result("srvtoolu_does_not_exist")],
            },
        ]
        _relocate_orphaned_tool_search_results(msgs)
        assert msgs[0]["content"][0]["type"] == "tool_search_tool_regex_tool_result"

    def test_relocation_runs_inside_convert_messages_to_anthropic(self):
        """End-to-end: assistant messages whose persisted server_tool_blocks
        carry the orphan pattern should come out paired after conversion."""
        # First assistant turn issued the tool_search.
        msg_turn1 = {
            "role": "assistant",
            "content": "",
            "server_tool_blocks": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_Z",
                    "name": "tool_search_tool_regex",
                    "input": {"query": "x"},
                },
            ],
            "tool_calls": [
                {
                    "id": "tu_local",
                    "function": {"name": "skills_list", "arguments": "{}"},
                }
            ],
        }
        msg_tool_result = {
            "role": "tool",
            "tool_call_id": "tu_local",
            "content": "skills...",
        }
        # Third assistant turn — Anthropic delivered the search result here,
        # but its tool_use_id pairs with msg_turn1's server_tool_use.
        msg_turn3 = {
            "role": "assistant",
            "content": "Got it",
            "server_tool_blocks": [
                {
                    "type": "tool_search_tool_regex_tool_result",
                    "tool_use_id": "srvtoolu_Z",
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": [],
                    },
                    "text": "RESPONSE-ONLY",
                    "citations": ["RESPONSE-ONLY"],
                },
            ],
            "tool_calls": [],
        }
        _, out_msgs = convert_messages_to_anthropic(
            [
                {"role": "user", "content": "hi"},
                msg_turn1,
                msg_tool_result,
                msg_turn3,
            ]
        )
        # Find the assistant messages in output (by role).
        assistants = [m for m in out_msgs if m["role"] == "assistant"]
        # The first assistant message must contain BOTH the server_tool_use
        # AND the tool_search result (collapsed to canonical type) in same
        # content list.
        first = assistants[0]["content"]
        types = [b.get("type") for b in first]
        assert "server_tool_use" in types
        assert "tool_search_tool_result" in types
        # The variant-suffixed wire type must NOT survive — Anthropic
        # rejects it on input.
        assert "tool_search_tool_regex_tool_result" not in types
        stu_idx = types.index("server_tool_use")
        assert (
            first[stu_idx + 1]["type"] == "tool_search_tool_result"
        )
        # And response-only fields are stripped on the relocated block.
        assert "text" not in first[stu_idx + 1]
        assert "citations" not in first[stu_idx + 1]
        # The later assistant message no longer carries the result
        # (in any form — bare canonical OR variant-suffixed).
        last_types = [b.get("type") for b in assistants[-1]["content"]]
        assert "tool_search_tool_result" not in last_types
        assert "tool_search_tool_regex_tool_result" not in last_types


# ---------------------------------------------------------------------------
# Variant-suffix canonicalization (regression for HTTP 400 "Input tag
# '<variant>_tool_result' ... does not match any of the expected tags")
# ---------------------------------------------------------------------------

class TestCanonicalizeToolSearchResultTypes:
    """Anthropic's INPUT validator (live verification 2026-05-07,
    request_id ``req_011Cap2RUgsJp1CVsGAR6LTa``) accepts ONLY the bare
    canonical ``tool_search_tool_result`` type. Variant-suffixed types
    (``tool_search_tool_regex_tool_result``, etc.) — which appear on
    the wire OUTPUT — are rejected with:

      "Input tag '<variant>_tool_result' found using 'type' does not
       match any of the expected tags: ..., 'tool_search_tool_result',
       'tool_use', ..."

    The Pydantic SDK model strips the variant during response parse, so
    fresh responses already carry the canonical type. But persisted
    sessions captured before the fix and any path that bypasses the
    SDK's parser can still leak the wire variant. The
    ``_canonicalize_tool_search_result_types`` helper normalizes them.

    These tests guard against regression by exercising the helper at
    every code path: capture-time (transports/anthropic.py), request-
    build-time (convert_messages_to_anthropic), and the helper in
    isolation.
    """

    # ------------------------------------------------------------------
    # Helper-level unit tests
    # ------------------------------------------------------------------
    def test_variant_suffixed_type_is_collapsed_to_bare_canonical(self):
        content = [
            {"type": "tool_use", "id": "toolu_x", "name": "terminal", "input": {}},
            {
                "type": "server_tool_use",
                "id": "srvtoolu_a",
                "name": "tool_search_tool_regex",
                "input": {"pattern": ".*"},
            },
            {
                "type": "tool_search_tool_regex_tool_result",  # wire variant
                "tool_use_id": "srvtoolu_a",
                "content": {"type": "tool_search_tool_search_result", "tool_references": []},
            },
        ]
        _canonicalize_tool_search_result_types(content)
        assert content[2]["type"] == "tool_search_tool_result"
        # Other blocks unchanged
        assert content[0]["type"] == "tool_use"
        assert content[1]["type"] == "server_tool_use"

    def test_idempotent_on_already_canonical(self):
        """Bare canonical is the fixed point — repeated passes don't
        mutate it."""
        content = [
            {
                "type": "tool_search_tool_result",
                "tool_use_id": "srvtoolu_b",
                "content": {"type": "tool_search_tool_search_result", "tool_references": []},
            },
        ]
        _canonicalize_tool_search_result_types(content)
        assert content[0]["type"] == "tool_search_tool_result"
        # And again
        _canonicalize_tool_search_result_types(content)
        assert content[0]["type"] == "tool_search_tool_result"

    def test_handles_multiple_variants(self):
        """Any ``tool_search_tool_<variant>_tool_result`` collapses,
        regardless of which variant — bm25, regex, substring, future ones."""
        for variant in ("regex", "bm25", "substring", "future_variant_v2"):
            content = [
                {
                    "type": f"tool_search_tool_{variant}_tool_result",
                    "tool_use_id": "srvtoolu_x",
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": [],
                    },
                },
            ]
            _canonicalize_tool_search_result_types(content)
            assert content[0]["type"] == "tool_search_tool_result", variant

    def test_non_tool_search_blocks_unchanged(self):
        """web_search and other server-side tool blocks must not be
        touched by this normalizer."""
        content = [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_web",
                "name": "web_search_20250305",
                "input": {"query": "x"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_web",
                "content": [],
            },
            {
                "type": "code_execution_tool_result",
                "tool_use_id": "srvtoolu_code",
                "content": {},
            },
        ]
        _canonicalize_tool_search_result_types(content)
        # All blocks unchanged
        assert content[0]["name"] == "web_search_20250305"
        assert content[1]["type"] == "web_search_tool_result"
        assert content[2]["type"] == "code_execution_tool_result"

    def test_message_list_dispatch_per_message(self):
        """Helper auto-detects message-list shape and dispatches per-message."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_m",
                        "name": "tool_search_tool_regex",
                        "input": {},
                    },
                    {
                        "type": "tool_search_tool_regex_tool_result",
                        "tool_use_id": "srvtoolu_m",
                        "content": {
                            "type": "tool_search_tool_search_result",
                            "tool_references": [],
                        },
                    },
                ],
            },
        ]
        _canonicalize_tool_search_result_types(messages)
        assert messages[1]["content"][1]["type"] == "tool_search_tool_result"

    def test_empty_and_non_list_inputs_are_safe(self):
        """No exceptions on edge-case inputs."""
        _canonicalize_tool_search_result_types([])
        _canonicalize_tool_search_result_types(None)
        _canonicalize_tool_search_result_types("not a list")
        _canonicalize_tool_search_result_types([{"type": "text", "text": "x"}])

    # ------------------------------------------------------------------
    # End-to-end via convert_messages_to_anthropic
    # ------------------------------------------------------------------
    def test_variant_suffixed_type_is_collapsed_on_send(self):
        """Reproduces the HTTP 400 scenario observed live on
        ``request_id req_011Cap2RUgsJp1CVsGAR6LTa``:

          msg[0] user
          msg[1] assistant content = [
            tool_use(toolu_x) name=tanium_developer_whoami,
            server_tool_use(srvtoolu_y) name=tool_search_tool_regex,
            tool_search_tool_regex_tool_result(srvtoolu_y)  ← VARIANT (BUG)
          ]
          msg[2] user content = [tool_result(toolu_x)]

        Anthropic rejected the variant-suffixed type with:
          "Input tag 'tool_search_tool_regex_tool_result' found using
           'type' does not match any of the expected tags: ...,
           'tool_search_tool_result', 'tool_use', ..."

        The fix collapses the type to the bare canonical
        ``tool_search_tool_result`` before the request is serialized.
        """
        anthropic_blocks = [
            {
                "type": "tool_use",
                "id": "toolu_x",
                "name": "tanium_developer_whoami",
                "input": {},
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_y",
                "name": "tool_search_tool_regex",
                "input": {"pattern": "tanium"},
            },
            {
                "type": "tool_search_tool_regex_tool_result",  # the bug
                "tool_use_id": "srvtoolu_y",
                "content": {
                    "type": "tool_search_tool_search_result",
                    "tool_references": [
                        {"type": "tool_reference", "tool_name": "x"},
                    ],
                },
            },
        ]
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "anthropic_content_blocks": anthropic_blocks,
                "tool_calls": [],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_x",
                "content": "ok",
            },
        ]
        _, out_msgs = convert_messages_to_anthropic(messages)

        # Find the assistant message and verify the search result block's
        # type was collapsed to the bare canonical form.
        asst = next(m for m in out_msgs if m["role"] == "assistant")
        types = [b.get("type") for b in asst["content"]]
        assert "tool_search_tool_result" in types, types
        # And the variant-suffixed form is GONE — that's the bug that
        # caused the 400.
        assert "tool_search_tool_regex_tool_result" not in types, types

    def test_capture_time_canonicalization_via_normalize_response(self):
        """The canonicalization also fires at capture time inside the
        Anthropic transport. We can't easily exercise the transport
        without mocking the SDK response object, so this test invokes
        the helper directly on a list shaped like ``server_tool_blocks``
        — which is what the transport stores under that key on the
        provider_data dict. The transport calls
        ``_canonicalize_tool_search_result_types(server_tool_blocks)``
        before persisting; this verifies the helper handles that exact
        list shape correctly."""
        server_tool_blocks = [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_capture",
                "name": "tool_search_tool_regex",
                "input": {"pattern": ".*"},
            },
            {
                "type": "tool_search_tool_regex_tool_result",
                "tool_use_id": "srvtoolu_capture",
                "content": {"type": "tool_search_tool_search_result", "tool_references": []},
            },
        ]
        _canonicalize_tool_search_result_types(server_tool_blocks)
        assert server_tool_blocks[1]["type"] == "tool_search_tool_result"
