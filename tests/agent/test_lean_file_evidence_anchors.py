"""Unit tests for everyday/enterprise file-provenance anchors and distilled read_file extracts
in lean context compaction (addressing ShellBench arXiv:2611.32946 information-omission gaps
while preserving the single-aux-call contract #96603 and strict secret redaction).
"""

import json
from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    ContextCompressor,
    _LEAN_ANCHOR_HEADING,
    _LEAN_FILE_EVIDENCE_HEADING,
    _build_anchor_index,
    _build_file_evidence_ledger,
)


def _mk_lean_compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
            tail_mode="lean",
        )
        _ = c.context_length
    c._session_id = "sess-file-evidence"
    return c


def test_anchor_index_harvests_tool_call_office_files_ids_and_amounts():
    turns = [
        {"role": "user", "content": "Audit westside_pediatrics_slots.eml and freight ledger."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "accounting/q4_freight_register.csv"}),
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "contracts/cedar_lodge_agreement.pdf"}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "INV-8841-REV2,TRK-99204,$4,185.50,2026-11-30",
        },
    ]
    idx = _build_anchor_index(turns)
    assert _LEAN_ANCHOR_HEADING in idx
    assert "accounting/q4_freight_register.csv" in idx
    assert "contracts/cedar_lodge_agreement.pdf" in idx
    assert "westside_pediatrics_slots.eml" in idx
    assert "INV-8841-REV2" in idx
    assert "TRK-99204" in idx
    assert "$4,185.50" in idx
    assert "2026-11-30" in idx


def test_file_evidence_ledger_strips_mime_and_repeated_boilerplate_and_redacts_secrets():
    boilerplate = "Standard legal confidentiality footer sentence repeated verbatim. " * 6
    turns = [
        {
            "role": "assistant",
            "content": "Reading email",
            "tool_calls": [
                {
                    "id": "c_eml",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "mail/inbox/ruling.eml"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c_eml",
            "content": (
                "Return-Path: <a@b.com>\n"
                "Received: from mx1 by mx2\n"
                "Message-ID: <123@b.com>\n"
                "Subject: Void INV-8841\n\n"
                "Confirmed: pallet #4 was double-counted. Token sk-proj-1234567890abcdefghijklmnop.\n"
                + boilerplate
            ),
        },
    ]
    ledger = _build_file_evidence_ledger(turns)
    assert _LEAN_FILE_EVIDENCE_HEADING in ledger
    assert "`mail/inbox/ruling.eml`" in ledger
    assert "pallet #4 was double-counted" in ledger
    assert "Return-Path:" not in ledger
    assert "Standard legal confidentiality footer" not in ledger
    assert "sk-proj-1234567890abcdefghijklmnop" not in ledger


def test_lean_summary_includes_file_evidence_with_single_llm_call():
    c = _mk_lean_compressor()
    turns = [
        {"role": "user", "content": "Check cedar_lodge_agreement.pdf"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "contracts/cedar_lodge_agreement.pdf"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "Negotiated Rate: $189.00/night\nPenalty-Free Attrition Cutoff: 2026-11-30 at 17:00 PST",
        },
    ]
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = MagicMock()
    resp.choices[0].message.content = "## Historical Task Snapshot\nUser asked.\n\n## Goal\nCheck contract."

    with patch("agent.context_compressor.call_llm", return_value=resp) as mock_call:
        summary = c._generate_summary(turns)

    assert mock_call.call_count == 1
    assert _LEAN_FILE_EVIDENCE_HEADING in summary
    assert "2026-11-30 at 17:00 PST" in summary
    assert "`contracts/cedar_lodge_agreement.pdf`" in summary


def test_compress_rescues_head_pruned_read_file_into_lean_evidence_ledger():
    """Regression test: when `protect_first_n` places the opening `read_file` group inside the
    head (`< compress_start`), Phase 1 prunes its body to a 1-line stub while `turns_to_summarize`
    starts at `compress_start`. Verify `compress()` rescues its distilled extract into the handoff."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=3,
            quiet_mode=True,
            tail_mode="lean",
        )
        _ = c.context_length
    c._session_id = "sess-head-rescue"

    msgs = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Audit freight ledger and run checks."},
        {
            "role": "assistant",
            "content": "Reading ledger",
            "tool_calls": [
                {
                    "id": "head_tc_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": "accounting/freight.csv"})},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "head_tc_1",
            "content": "invoice_id,status\nINV-8841,DRAFT_SUPERSEDED\n" + ("[Archive Boilerplate Row] " * 400),
        },
    ]
    for i in range(12):
        msgs.append(
            {
                "role": "assistant",
                "content": f"Step {i}",
                "tool_calls": [
                    {
                        "id": f"m_{i}",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": json.dumps({"command": f"echo {i}"})},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"m_{i}", "content": ("output line " * 400)})
    msgs.append({"role": "user", "content": "Continue."})
    msgs.append({"role": "assistant", "content": "Done."})

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = MagicMock()
    resp.choices[0].message.content = "## Historical Task Snapshot\nAudit.\n\n## Goal\nAudit."

    with patch("agent.context_compressor.call_llm", return_value=resp):
        compressed = c.compress(msgs, current_tokens=60_000, force=True)

    joined = "\n".join(str(m.get("content") or "") for m in compressed)
    assert "`accounting/freight.csv`" in joined
    assert "DRAFT_SUPERSEDED" in joined


def test_head_pruned_rescue_excludes_system_prompt_and_unpruned_head_siblings():
    """Regression test for PR #160445 review: when a head tool output is pruned in Phase 1,
    only the pruned tool exchange may feed `_augment_summary_lean` — identifiers in the
    protected `system` prompt, head `user` prompt, or unpruned sibling head tool calls must
    never appear in the summary's Anchor Index or File Evidence Extracts."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=4,
            protect_last_n=3,
            quiet_mode=True,
            tail_mode="lean",
        )
        _ = c.context_length
    c._session_id = "sess-head-isolation"

    msgs = [
        {
            "role": "system",
            "content": "Runbook https://internal.example/runbook at ops/private_runbook.md (#99999 @ops-oncall).",
        },
        {"role": "user", "content": "Check docs/unrelated_head_note.md and audit freight."},
        {
            "role": "assistant",
            "content": "Checking files (see docs/assistant_head_prose.md)",
            "tool_calls": [
                {
                    "id": "head_pruned_tc",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": "accounting/freight.csv"})},
                },
                {
                    "id": "head_kept_tc",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": "notes/tiny_keep.txt"})},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "head_pruned_tc",
            "content": "invoice_id,status\nINV-8841,DRAFT_SUPERSEDED\n" + ("[Archive Boilerplate Row] " * 400),
        },
        {
            "role": "tool",
            "tool_call_id": "head_kept_tc",
            "content": "TINY-1234 short",
        },
    ]
    for i in range(12):
        msgs.append(
            {
                "role": "assistant",
                "content": f"Step {i}",
                "tool_calls": [
                    {
                        "id": f"m_{i}",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": json.dumps({"command": f"echo {i}"})},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"m_{i}", "content": ("output line " * 400)})
    msgs.append({"role": "user", "content": "Continue."})
    msgs.append({"role": "assistant", "content": "Done."})

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = MagicMock()
    resp.choices[0].message.content = "## Historical Task Snapshot\nAudit.\n\n## Goal\nAudit."

    with patch("agent.context_compressor.call_llm", return_value=resp):
        compressed = c.compress(msgs, current_tokens=60_000, force=True)

    summary_rows = [
        str(m.get("content") or "")
        for m in compressed
        if m.get("role") != "system" and _LEAN_ANCHOR_HEADING in str(m.get("content") or "")
    ]
    assert len(summary_rows) == 1
    handoff = summary_rows[0]

    # Rescued pruned head read IS preserved:
    assert "accounting/freight.csv" in handoff
    assert "INV-8841" in handoff
    assert "DRAFT_SUPERSEDED" in handoff

    # Protected system prompt, head user/assistant prose, and unpruned sibling head tool call MUST NOT leak:
    for forbidden in (
        "https://internal.example/runbook",
        "ops/private_runbook.md",
        "#99999",
        "@ops-oncall",
        "docs/unrelated_head_note.md",
        "docs/assistant_head_prose.md",
        "notes/tiny_keep.txt",
        "TINY-1234",
    ):
        assert forbidden not in handoff, f"Unpruned head content {forbidden!r} leaked into handoff summary"

