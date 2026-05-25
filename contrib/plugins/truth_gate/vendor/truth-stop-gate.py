#!/usr/bin/env python3
"""Phase-1 Stop hook truth gate. Mechanical block on rule violation.

Reads Stop payload. If stop_hook_active true: emit {}, exit 0.
Else evaluates last_assistant_message against rules and emits a block
payload + exit 2 if violations found.

Footer-required scope is CURRENT TURN ONLY:
  Read transcript_path JSONL, find the most recent two assistant
  messages. If any tool_use/tool_result event sits BETWEEN them,
  the current turn used tools -> footer required.
  Fallback when transcript read fails: footer required only if
  risky language is present in the final assistant text.
  No session-wide footer lock.

Block payload format (Phase 0B confirmed):
  {"decision":"block","reason":"...","continue":false,"stopReason":"..."}
+ exit 2.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

GATE_LOG = Path(os.path.expanduser("~")) / ".claude" / "truth" / "stop-gate.log.jsonl"
MAX_LOG_BYTES = 512 * 1024
KEEP_LOG_BYTES = 256 * 1024

# Pack 1 truth-gate-hook-repair: discover-required flag for fillable-GAP path.
DISCOVER_FLAG = Path(os.path.expanduser("~")) / ".claude" / "truth" / "discover-required.flag"
DISCOVER_FLAG_TTL_SEC = 1800

# Pack 1J-Y: metrics-gate-failed flag. Written when schema-v2 Metrics Gate
# violations fire in a session; read by truth-prompt-inject to append a
# METRICS GATE FAILED reminder block on the next turn. Cleared when the
# next assistant answer carries a valid Metrics Gate PASS table.
METRICS_GATE_FAILED_FLAG = Path(os.path.expanduser("~")) / ".claude" / "truth" / "metrics-gate-failed.flag"
METRICS_GATE_FAILED_TTL_SEC = 1800

# Footer detection strip: ONLY fenced code, NOT TRUTH-footer-and-below
# (otherwise footer detection would always strip its own target).
_FENCED_ONLY_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)

# Pack 1 GAP classification.
_UNFILLABLE_REASON_RE = re.compile(r"(?i)\bunfillable\s*:\s*(\S.{3,})")
_UNFILLABLE_BARE_RE = re.compile(r"(?i)\bunfillable\b")
_FILLABLE_KEYWORDS_RE = re.compile(
    r"(?i)\b(fillable|tbd|todo|recheck|rerun|need\s+to\s+confirm"
    r"|need\s+check|to\s+verify|to\s+investigate|need\s+to\s+look"
    r"|need\s+to\s+read|next\s+pack|next\s+session|defer)\b"
)

# Owner-decision / approval-required GAP marker. Strict: must start the GAP
# text (leading whitespace only), uppercase-literal, mandatory single
# separator between words, and a meaningful reason after the colon. Keeps
# the "Al must decide" escape hatch hard to trip accidentally.
_NEEDS_APPROVAL_RE = re.compile(
    r"\A\s*(?:NEEDS[ _-]APPROVAL|AWAITING[ _-]OWNER[ _-]DECISION"
    r"|OWNER[ _-]DECISION[ _-]REQUIRED)\s*:\s*(.+)"
)
_NEEDS_APPROVAL_BAD_REASONS = frozenset({"todo", "tbd", "none", "n/a"})

# Pack 1B truth-gate-rewrite-loop-fallback: rewrite-required flag.
# Survives user-prompt arrival so a Stop-block always carries a durable
# rewrite directive into the next assistant turn.
REWRITE_FLAG = Path(os.path.expanduser("~")) / ".claude" / "truth" / "rewrite-required.flag"
REWRITE_FLAG_TTL_SEC = 1800

# Pack 1J-AI: per-session rewrite flag dir. Eliminates the global-file
# contention that allowed sibling Stop hooks to archive each other's
# rewrite pressure (Pack 1J-AG forensics).
REWRITE_FLAG_DIR = Path(os.path.expanduser("~")) / ".claude" / "truth" / "rewrite-required-flags"
MIN_REWRITE_BYTES = 200
REWRITE_HEADER_LITERAL = "REWRITE CORRECTION -- prior answer failed Truth Gate"
_REWRITE_RULE_IDS_LINE_RE = re.compile(r"^\s*rule_ids\s*:\s*(.+?)\s*$")

# Pack 1C: strip exactly ONE optional Claude Code UI bullet/marker glyph
# (followed by whitespace) from the start of a rendered line, before the
# strict header / rule_ids parse. Allows 'â— REWRITE CORRECTION ...' to
# match the literal header without permitting arbitrary words before it.
_UI_BULLET_PREFIX_RE = re.compile(
    r"^[\sÂ ]*"
    r"(?:[â—â€¢â€£â—†â—‡â—¦â–¸â–¶â– â€“â€”\-\*â†’âž”â®ž]\s+)?"
)

PREAMBLE_RE = re.compile(
    r"(?im)^(?:\s*)("
    r"let me"
    r"|i'll"
    r"|i will"
    r"|i'm going to"
    r"|sure[,!]"
    r"|here's what"
    r"|here is what"
    r")\b"
)

RECAP_RE = re.compile(
    r"(?i)\b("
    r"to summarize"
    r"|in summary"
    r"|in short"
    r"|just to confirm"
    r"|to recap"
    r"|recap:"
    r")\b"
)

MOTIVE_RE = re.compile(
    r"(?i)\b("
    r"i lied because"
    r"|the reason i (?:did|said|wrote)"
    r"|my intent was"
    r"|on purpose"
    r")\b"
)

CLOSURE_RE = re.compile(
    r"(?i)\b("
    r"done"
    r"|complete[d]?"
    r"|fixed"
    r"|resolved"
    r"|shipped"
    r"|implemented"
    r"|landed"
    r")\b"
)

BEHAVIOR_RE = re.compile(
    r"(?i)\b("
    r"works"
    r"|working"
    r"|functional"
    r"|operational"
    r"|healthy"
    r"|stable"
    r")\b"
)

VERIFICATION_RE = re.compile(
    r"(?i)\b("
    r"verified"
    r"|confirmed"
    r"|proven"
    r"|validated"
    r"|tested"
    r"|end-to-end"
    r"|e2e"
    r")\b"
)

CLEANLINESS_RE = re.compile(
    r"(?i)\b("
    r"clean"
    r"|green"
    r"|passed"
    r"|no failures"
    r"|no errors"
    r"|zero failures"
    r")\b"
)

UNIVERSAL_RE = re.compile(
    r"(?i)\b("
    r"all\s+(?:nodes?|boxes?|machines?|tests?|hooks?|phases?|repos?|files?|builds?|jobs?|sessions?)"
    r"|zero\s+(?:failures?|errors?|gaps?|drift|destructive\s+ops?)"
    r"|no\s+(?:failures?|errors?|gaps?|drift|destructive\s+ops?)"
    r"|every\s+(?:node|box|machine|test|hook|phase|repo|file|build|job|session)"
    r"|none\s+(?:failed|missing)"
    r"|fully\s+(?:converged|synced|deployed)"
    r")\b"
)

CONFIDENCE_RE = re.compile(
    r"(?i)("
    r"\b100\s*%"
    r"|\b95\s*%"
    r"|\bconfidence\b"
    r"|\bcertain(?:ty)?\b"
    r"|\bguaranteed\b"
    r"|\bdefinitely\b"
    r"|\bfor sure\b"
    r")"
)

# Body-to-footer hard link: production blockers in claim-bearing prose must
# not be laundered into a green footer. Keep this intentionally narrow and
# deterministic; strip_non_claim_regions() removes code fences and footer text.
BODY_BLOCKING_FINDING_RE = re.compile(
    r"(?i)("
    r"unsafe\s+to\s+leave\s+live"
    r"|not\s+safe\s+to\s+leave\s+live"
    r"|fix\s+or\s+revert"
    r"|failed\s+tests?"
    r"|\b\d+\s+failed\s*/\s*\d+\s+passed\b"
    r"|auto-rewrite-stuck"
    r"|auto-rewrite-actuator-failed"
    r"|metrics-gate-failed"
    r"|transcript-user-row-mismatch"
    r")"
)

DANGER_CATEGORIES = (
    ("closure",      CLOSURE_RE),
    ("behavior",     BEHAVIOR_RE),
    ("verification", VERIFICATION_RE),
    ("cleanliness",  CLEANLINESS_RE),
    ("universal",    UNIVERSAL_RE),
    ("confidence",   CONFIDENCE_RE),
)

SPECIFIC_RES = [
    (re.compile(r"(?i)\bregistered\b.*\bfiring\b|\bfiring\b.*\bregistered\b"),
     "'registered' != 'firing' -- need PID/listen evidence"),
    (re.compile(r"(?i)--help\b.*\bworks?\b"),
     "'--help exit 0' != 'works' -- need real-call evidence"),
    (re.compile(r"(?i)\bdeployed\b.*\b(verified|runtime)\b"),
     "'deployed' != 'runtime verified' -- need runtime probe"),
    (re.compile(r"(?i)\b(FAIL|WARN)=\d+"),
     "'FAIL/WARN counters present' != 'clean'"),
    (re.compile(r"(?i)\ball\s+(?:nodes?|boxes?|machines?)\b"),
     "'all nodes/boxes/machines' requires per-node evidence in footer body"),
]

FOOTER_HEADER_RE = re.compile(r"(?im)^\s*TRUTH:\s*$")
FOOTER_BULLET_RE = re.compile(r"(?im)^\s*-\s*(PROVEN|PARTIAL|GAP)\s*:")

# Pack 1J-W schema v2: extended section set + Metrics Gate.
SCHEMA_V2_SECTIONS = ("PROVEN", "PARTIAL", "GAP", "CURRENT_STATE", "NEXT", "BEHAVIOR_FAIL")
FOOTER_BULLET_V2_RE = re.compile(
    r"(?im)^\s*-\s*(PROVEN|PARTIAL|GAP|CURRENT_STATE|NEXT|BEHAVIOR_FAIL)\s*:"
)
METRICS_GATE_HEADER_RE = re.compile(r"(?im)^\s*(?:BUILD\s+)?METRICS\s+GATE\s*:?\s*$")
METRICS_ROW_RE = re.compile(
    r"(?im)^\s*\|\s*(GAPS_FILLED|DISCOVERY|BUILD_CONFIDENCE|METRICS_GATE)\s*\|"
    r"[^|]*\|\s*([^|]+?)\s*\|\s*(PASS|FAIL|__)\s*\|"
)
METRICS_ROW_ANY_RE = re.compile(
    r"(?im)^\s*\|\s*(GAPS_FILLED|DISCOVERY|BUILD_CONFIDENCE|METRICS_GATE)\s*\|"
    r"[^|]*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|"
)
METRIC_BLANK_RE = re.compile(r"\|\s*__\s*(?:%\s*)?\|")
METRICS_GATE_WIDE_REQUIRED_RE = re.compile(
    r"(?im)^\s*\|\s*METRICS_GATE\s*\|\s*PASS only if all above pass and no blocking "
    r"GAP/PARTIAL/BEHAVIOR_FAIL remains\s*\|"
)
BLANK_SEPARATED_PIPE_TABLE_RE = re.compile(
    r"(?im)^[ \t]*\|[^|\n]*[A-Za-z][^\n]*\|[ \t]*\r?\n[ \t]*\r?\n[ \t]*\|[ \t:\-|]+\|[ \t]*$"
)
E2E_FORCE_SHORT_TOKEN = "TG_E2E_FORCE_SHORT_FAIL_7B3E9C"
E2E_FORCE_FULL_TOKEN = "TG_E2E_FORCE_FULL_FAIL_7B3E9C"
METRICS_REQUIRED_GAPS_FILLED = 100
METRICS_REQUIRED_DISCOVERY = 100
# Pack 1J-BN: canonical strict-grid regexes. Legacy TRUTH bullets are no
# longer accepted as a final footer. Each row is a literal Markdown pipe row.
TRUTH_PROVEN_HEADER_RE = re.compile(r"(?im)^\s*(?:TRUTH|TRUTH_PROVEN)\s*:?\s*$")
TRUTH_PARTIAL_HEADER_RE = re.compile(r"(?im)^\s*TRUTH_PARTIAL\s*:?\s*$")
TRUTH_GAP_HEADER_RE = re.compile(r"(?im)^\s*(?:GAP|TRUTH_GAP)\s*:?\s*$")
CURRENT_STATE_HEADER_RE = re.compile(r"(?im)^\s*CURRENT_STATE\s*:?\s*$")
NEXT_HEADER_RE = re.compile(r"(?im)^\s*NEXT\s*:?\s*$")
STATE_NEXT_HEADER_RE = re.compile(r"(?im)^\s*STATE_NEXT\s*:?\s*$")
BEHAVIOR_FAIL_HEADER_RE = re.compile(r"(?im)^\s*BEHAVIOR_FAIL\s*:?\s*$")
TRUTH_PROVEN_ROW_RE = re.compile(
    r"(?im)^\s*\|\s*([TP]\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(YES|NO)\s*\|\s*$"
)
SHORT_TRUTH_ROW_RE = re.compile(
    r"(?im)^\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(YES|NO)\s*\|\s*$"
)
TRUTH_PARTIAL_ROW_RE = re.compile(
    r"(?im)^\s*\|\s*(PT\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(YES|NO)\s*\|\s*$"
)
TRUTH_GAP_ROW_RE = re.compile(
    r"(?im)^\s*\|\s*(G\d+)\s*\|\s*([^|]+?)\s*\|\s*(YES|NO)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(YES|NO)\s*\|\s*$"
)
CURRENT_STATE_ROW_RE = re.compile(
    r"(?im)^\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$"
)
NEXT_ROW_RE = re.compile(
    r"(?im)^\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$"
)
STATE_NEXT_ROW_RE = re.compile(
    r"(?im)^\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$"
)
BEHAVIOR_FAIL_ROW_RE = re.compile(
    r"(?im)^\s*\|\s*(BF\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(YES|NO)\s*\|\s*$"
)
SHORT_GAP_ROW_RE = re.compile(r"(?im)^\s*\|\s*([^|]+?)\s*\|\s*$")
SHORT_STATE_NEXT_ROW_RE = re.compile(r"(?im)^\s*\|\s*([^|]+?)\s*\|\s*$")
SHORT_METRICS_ROW_RE = re.compile(r"(?im)^\s*\|\s*(PASS|FAIL)\s*\|\s*$")
SHORT_BEHAVIOR_FAIL_ROW_RE = re.compile(
    r"(?im)^\s*\|\s*([^|]+?)\s*\|\s*(YES|NO)\s*\|\s*$"
)
COMPACT_TRUTH_RE = re.compile(r"(?im)^\s*TRUTH:\s*(.+?)\s*\|\s*VERIFIED:\s*YES\s*$")
COMPACT_GAP_RE = re.compile(r"(?im)^\s*GAP:\s*none\s*$")
COMPACT_STATE_NEXT_RE = re.compile(r"(?im)^\s*STATE_NEXT:\s*idle\s*/\s*await instruction\s*$")
COMPACT_BUILD_METRICS_RE = re.compile(r"(?im)^\s*BUILD\s+METRICS\s+GATE:\s*PASS\s*$")
COMPACT_BEHAVIOR_FAIL_RE = re.compile(
    r"(?im)^\s*BEHAVIOR_FAIL:\s*none\s*\|\s*REWRITE_REQUIRED:\s*NO\s*$"
)
COMPACT_DISALLOWED_RE = re.compile(
    r"(?i)("
    r"https?://|[A-Za-z]:[\\/]|[\\/][\w.-]+[\\/]|sha256|hash|exit[_ -]?code|"
    r"\b(git|drive|runtime|e2e|test(?:ed|s)?|passed|failed|fixed|changed|installed|uploaded|deployed|implemented|resolved|shipped|verified|validated|confirmed)\b|"
    r"\b(compare|review|analy[sz]e|risk|risks|recommendation|recommend|option|options|plan|launch|deadline|rollback|monitoring|tradeoff|tradeoffs|dimension|dimensions)\b|"
    r"\b\d+\s+(?:passed|failed|tests?)\b"
    r")"
)
INVALID_CELL_VALUES = ("HOLD", "DEFERRED", "N/A", "TBD", "UNKNOWN", "PARTIAL")
INVALID_CELL_RE = re.compile(
    r"\|\s*(HOLD|DEFERRED|N/A|TBD|UNKNOWN|PARTIAL)\s*\|"
)
# Pack 1J-AP: detect key/value fake-grid form. After TRUTH_PROVEN: header,
# look for non-pipe-table lines like "ID: P1" / "Claim:" / "Ledger anchor:"
# / "Verified:" / "What proven:" / "Gap:" / "Fillable:" -- these are key/value
# blocks masquerading as grids and must be rejected when v3 required.
KV_FAKE_GRID_KEY_RE = re.compile(
    r"(?im)^\s*(ID|Claim|Ledger anchor|Proof|Verified|What proven|What not proven|"
    r"Why partial|What closes it|Objective-critical|Gap|Fillable|"
    r"Missing proof|Next read-test-action|Blocks PASS|State|Next action|Owner|"
    r"Metric|Required|Actual|Pass/Fail|Failure)\s*:"
)
# Pack 1J-AS: detect box-drawing fake-grid form. After any canonical section
# header, any Unicode U+2500-U+257F box-drawing
# glyph indicates a non-pipe-table grid (e.g. â”Œâ”€â” â”‚ â•”â•â•—) that must be
# rejected. Pipe-grid uses ASCII | and -, so box-drawing chars never appear
# in a valid canonical grid row.
BOX_DRAWING_RE = re.compile(r"[\u2500-\u257F]")
METRICS_REQUIRED_BUILD_CONFIDENCE = 95

FIX_FOOTER = "add SHORT or FULL Truth Gate box-table-rendering Markdown pipe-grid footer with BUILD METRICS GATE, OR remove the claim"
FIX_TOOL_FOOTER = "tools were used in THIS answer; end reply with canonical Truth Gate pipe-grid footer plus BUILD METRICS GATE"

# --- Non-claim region stripping (hotfix + hotfix2) ---
_SECTION_HEADINGS = (
    # Approval-packet sections (Hotfix 1).
    "TEST PLAN",
    "TEST PLAN AFTER APPROVAL",
    "EXPECTED OBSERVATIONS",
    "GAPS",
    "ROLLBACK",
    "ROLLBACK COMMANDS",
    "VALIDATION COMMANDS",
    "BACKUP COMMANDS",
    "TEST PROCEDURE AFTER APPROVAL",
    "UNINSTALL COMMANDS",
    "PROOF CONTRACTS FILE",
    "PROBE FILE",
    "BLOCK PROBE FILE",
    "INJECT PROBE FILE",
    "PROMPT INJECT FILE",
    "EVIDENCE LEDGER FILE",
    "STOP GATE FILE",
    "REGEX CONSTANTS ONLY",
    "PROPOSED PATCH",
    "CURRENT ISSUE",
    "HOTFIX PURPOSE",
    "HOTFIX 2 PURPOSE",
    "FILES TO CHANGE",
    "PROPOSED USERPROMPTSUBMIT HOOKS",
    "PROPOSED STOP HOOKS",
    "PROPOSED SETTINGS",
    "SETTINGS DIFF",
    "SETTINGS PROPOSED HOOKS",
    "SETTINGS CURRENT HOOKS AFFECTED",
    "EXACT PROPOSED HOOKS AFTER 0A",
    "EXACT CURRENT HOOKS",
    "STOP PAYLOAD RAW SHAPE",
    "USERPROMPTSUBMIT PAYLOAD RAW SHAPE",
    "PARSER IMPLICATIONS",
    "BLOCK RESULT FILE",
    "INJECT RESULT FILE",
    "PATCH",
    "SMOKE TESTS",
    # Report-format sections (Hotfix 2).
    "LIVE TEST SURFACE",
    "LIVE TEST 1 SURFACE",
    "LIVE TEST 2 SURFACE",
    "LIVE TEST 3 SURFACE",
    "LIVE TEST 4 SURFACE",
    "LIVE TEST 5 SURFACE",
    "LIVE TEST 6 SURFACE",
    "LIVE TEST 7 SURFACE",
    "LIVE TEST 8 SURFACE",
    "EVIDENCE LEDGER ROW",
    "EVIDENCE LEDGER ROWS",
    "STOP GATE LOG ROW",
    "STOP GATE LOG ROWS",
    "RESULT",
    "VERDICT",
    "WAITING FOR NEXT INSTRUCTION",
    "WAITING FOR APPROVAL",
    "WAITING FOR ACTIVE INJECTION TEST",
    "WAITING FOR SENTINEL TEST",
    "BACKUP HASH",
    "POST-EDIT JSON VALIDATION",
    "USERPROMPTSUBMIT CHAIN COUNT",
    "STOP CHAIN COUNT",
    "INERT TEST RESULT",
    "0A USERPROMPTSUBMIT STILL CAPTURING",
    "0A STOP PROBE STILL CAPTURING",
    "INERT RESULT",
    "BLOCK RESULT FILE EXISTS",
    "BLOCK RESULT FILE MTIME/HASH",
    "INJECT RESULT FILE EXISTS",
    "INJECT RESULT FILE MTIME/HASH",
    "LATEST 0A USERPROMPTSUBMIT PAYLOAD",
    "LATEST 0A STOP PAYLOAD",
    "INERT VERDICT",
    "0D VERDICT",
    "INJECT RESULT JSON",
    "0A PAYLOAD AROUND TEST",
    "0D UNINSTALL RESULT",
    "0B UNINSTALL RESULT",
    "SETTINGS HASH AFTER RESTORE",
    "JSON VALIDATION",
    "USERPROMPTSUBMIT CHAIN",
    "STOP CHAIN",
    "0A STILL INSTALLED",
    "0B RESULT FILE KEPT",
    "0D RESULT FILE KEPT",
    "0D HOOK REMOVED",
    "TRUTH-BLOCK-PROBE.PY REMOVED",
    "INSTALL RESULT",
    "VALIDATION RESULT",
    "HOOK CHAIN RESULT",
    "TEST 1 RESULT",
    "TEST 2 RESULT",
    "TEST 3 RESULT",
    "TEST 4 RESULT",
    "TEST 5 RESULT",
    "TEST 6 RESULT",
    "TEST 7 RESULT",
    "TEST 8 RESULT",
    "0A CLEANUP STATUS",
    "LEDGER FILE",
    "STOP GATE LOG",
    "HOTFIX RESULT",
    "HOTFIX 2 RESULT",
    "SETTINGS HASH",
    "HOOK BACKUP",
    "PY_COMPILE",
    "IMPORT",
    "SMOKE CASE A",
    "SMOKE CASE B",
    "SMOKE CASE C",
    "SMOKE CASE D",
    "SMOKE CASE E",
    "SMOKE 1",
    "SMOKE 2",
    "SMOKE 3",
    "SMOKE 4",
    "STOP GATE HASH",
    "LIVE TESTS STILL NEEDED",
    # Audit / self-audit containers (Hotfix 4b).
    "SELF-AUDIT",
    "CLAIM",
    "REALITY",
    "WHAT I SHOULD HAVE SAID",
    "HONEST ANSWER",
    "PRIOR CLAIM",
    "CORRECTION",
    "DOWNGRADE",
    "DOWNGRADED",
)

# --- Hotfix 4b: line-prefix dropping for inline audit annotations ---
_LINE_PREFIX_RE = re.compile(
    r"(?im)^\s*(?:"
    r"Claim"
    r"|Reality"
    r"|What I should have said"
    r"|Prior claim"
    r"|Correction"
    r"|Downgraded"
    r"|Honest answer"
    r")\s*:.*$"
)

# Markdown audit-table detection: header row containing audit column terms.
_AUDIT_TABLE_HEADER_RE = re.compile(
    r"(?im)^\s*\|.*\b(?:"
    r"Claim"
    r"|Reality"
    r"|Prior claim"
    r"|Correction"
    r"|Downgraded"
    r"|What I should have said"
    r"|Honest answer"
    r")\b.*\|.*$"
)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")

_FENCED_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_TRUTH_FOOTER_AND_BELOW_RE = re.compile(r"(?ims)^\s*TRUTH:\s*$.*\Z")

_HEADING_LINE_RE = re.compile(
    r"(?im)^\s*(?:" + "|".join(re.escape(h) for h in _SECTION_HEADINGS) + r")\s*:\s*$"
)

# Parametric heading regex (Hotfix 2). Catches numbered/structural variants
# without enumerating every integer.
_HEADING_PATTERN_RE = re.compile(
    r"(?im)^\s*(?:"
    r"LIVE\s+TEST\s+\d+\s+SURFACE"
    r"|TEST\s+\d+\s+RESULT"
    r"|SMOKE\s+CASE\s+[A-Z]"
    r"|SMOKE\s+\d+"
    r"|STOP\s+GATE\s+LOG\s+ROWS?"
    r"|EVIDENCE\s+LEDGER\s+ROWS?"
    r"|WAITING\s+FOR\s+[A-Z][A-Z _]+"
    r"|PHASE\s+\d[A-Z]?"
    r"|HOTFIX\s+\d+\s+PURPOSE"
    r"|HOTFIX\s+\d+\s+RESULT"
    r")\s*:\s*$"
)


def _is_heading(line):
    return bool(_HEADING_LINE_RE.match(line)) or bool(_HEADING_PATTERN_RE.match(line))


def strip_non_claim_regions(text):
    """Return claim-bearing text only.

    Strips:
      1. Fenced code blocks.
      2. TRUTH footer and everything below it.
      3. Markdown audit tables: rows starting from a header line that
         contains audit column names (Claim, Reality, Prior claim, etc.)
         until the table ends (line not starting with '|').
      4. Sections introduced by literal or parametric heading lines.
      5. Lines matching audit line-prefix annotations
         (Claim:, Reality:, etc.).

    has_footer() runs on the ORIGINAL full text so footer detection is
    unaffected by stripping.
    """
    if not isinstance(text, str) or not text:
        return ""
    out = _FENCED_CODE_RE.sub(" ", text)
    out = _TRUTH_FOOTER_AND_BELOW_RE.sub("", out)
    lines = out.split("\n")
    keep = []
    in_section = False
    in_audit_table = False
    for line in lines:
        if in_audit_table:
            if _TABLE_ROW_RE.match(line):
                continue
            in_audit_table = False
            # fall through to other handling for this line
        if _AUDIT_TABLE_HEADER_RE.match(line):
            in_audit_table = True
            continue
        if _is_heading(line):
            in_section = True
            continue
        if in_section:
            if line.strip() == "":
                in_section = False
            continue
        if _LINE_PREFIX_RE.match(line):
            continue
        keep.append(line)
    return "\n".join(keep)


# --- Phase-2-lite: evidence-anchor check ---
LEDGER = Path(os.path.expanduser("~")) / ".claude" / "truth" / "evidence-ledger.jsonl"
LEDGER_LOOKBACK = 30
ANCHOR_MIN_LEN = 4

_PROVEN_BULLET_RE = re.compile(r"(?im)^\s*-\s*(PROVEN|PARTIAL|GAP)\s*:\s*(.*)$")


def _extract_proven_lines(text):
    if not isinstance(text, str):
        return []
    proven = []
    # Pack 1J-AP: also walk v3 TRUTH_PROVEN pipe grid rows; append synthetic
    # "<claim> <ledger_anchor>" string per row so existing _has_ledger_anchor
    # substring match works against v3 grid bodies. Forward-ref to
    # _parse_truth_proven_grid (defined later in file) resolves at call time.
    try:
        for row in _parse_truth_proven_grid(text):
            claim = row.get("claim", "") or ""
            anchor = row.get("ledger_anchor", "") or ""
            combined = (claim + " " + anchor).strip()
            if combined:
                proven.append(combined)
    except NameError:
        pass
    try:
        for row in _parse_short_truth_grid(text):
            claim = row.get("claim", "") or ""
            proof = row.get("proof", "") or ""
            combined = (claim + " " + proof).strip()
            if combined:
                proven.append(combined)
    except NameError:
        pass
    m = FOOTER_HEADER_RE.search(text)
    if not m:
        return proven
    footer = text[m.end():]
    lines = footer.split("\n")
    in_proven = False
    for ln in lines:
        bm = _PROVEN_BULLET_RE.match(ln)
        if bm:
            section = bm.group(1).upper()
            if section == "PROVEN":
                in_proven = True
                tail = bm.group(2).strip()
                if tail:
                    proven.append(tail)
            else:
                in_proven = False
        elif in_proven:
            stripped = ln.strip()
            if not stripped:
                continue
            if stripped.startswith("-"):
                in_proven = False
                continue
            proven.append(stripped)
    return proven


def _ledger_anchors_for_session(session_id, lookback=LEDGER_LOOKBACK):
    if not session_id or not LEDGER.exists():
        return []
    try:
        with LEDGER.open("r", encoding="utf-8") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200 * 1024))
            tail = f.read()
        rows = [r for r in tail.split("\n") if r.strip()]
    except Exception:
        return []
    anchors = []
    for line in rows[-lookback:][::-1]:
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("session_id") != session_id:
            continue
        ti = row.get("tool_input") or {}
        for k in ("command", "file_path", "path", "query", "pattern", "symbol", "description"):
            v = ti.get(k)
            if isinstance(v, str) and len(v) >= ANCHOR_MIN_LEN:
                anchors.append(v)
        for k in ("stdout_tail", "stderr_tail", "error_tail"):
            v = row.get(k)
            if isinstance(v, str) and len(v) >= ANCHOR_MIN_LEN:
                anchors.append(v)
        iec = row.get("inferred_exit_code")
        if isinstance(iec, int):
            anchors.append(f"exit code {iec}")
            anchors.append(f"Exit code {iec}")
        elif isinstance(iec, str) and len(iec) >= ANCHOR_MIN_LEN:
            anchors.append(iec)
    return anchors


def _has_ledger_anchor(proven_lines, anchors):
    """Substring match with slash-direction normalization.

    A path stored as 'C:\\Users\\X' in the ledger should still match a
    PROVEN line that quotes 'C:/Users/X' (forward slashes), and vice
    versa. Normalize both sides by replacing backslashes with forward
    slashes before comparing.
    """
    if not proven_lines or not anchors:
        return False
    for plain in proven_lines:
        if not plain:
            continue
        plain_norm = plain.replace("\\", "/")
        for anchor in anchors:
            if not anchor:
                continue
            anchor_norm = anchor.replace("\\", "/")
            if anchor in plain or anchor_norm in plain_norm:
                return True
            if len(anchor) >= 12:
                head = anchor[:12]
                head_norm = anchor_norm[:12]
                if (head and head in plain) or (head_norm and head_norm in plain_norm):
                    return True
    return False


def _strip_for_footer_detection(text):
    """Pack 1: strip ONLY fenced code so a fenced TRUTH footer cannot
    masquerade as a satisfying footer. Does NOT strip the TRUTH footer
    region itself (unlike strip_non_claim_regions which removes it for
    risky-language scanning)."""
    if not isinstance(text, str) or not text:
        return ""
    return _FENCED_ONLY_RE.sub(" ", text)


def _has_blank_separated_pipe_table(text):
    if not isinstance(text, str):
        return False
    return bool(BLANK_SEPARATED_PIPE_TABLE_RE.search(text))


def has_footer(text):
    stripped = _strip_for_footer_detection(text)
    return _canonical_truth_footer_present(stripped)


def _compact_truth_footer_present(text):
    if not isinstance(text, str):
        return False
    return (
        bool(COMPACT_TRUTH_RE.search(text))
        and bool(COMPACT_GAP_RE.search(text))
        and bool(COMPACT_STATE_NEXT_RE.search(text))
        and bool(COMPACT_BUILD_METRICS_RE.search(text))
        and bool(COMPACT_BEHAVIOR_FAIL_RE.search(text))
    )


def _compact_behavior_fail_last(text):
    if not isinstance(text, str):
        return False
    bf = COMPACT_BEHAVIOR_FAIL_RE.search(text)
    mg = COMPACT_BUILD_METRICS_RE.search(text)
    return bool(bf and mg and bf.start() > mg.start())


def _compact_footer_allowed(text, used_tools_current_turn=False, rewrite_flag_active=False):
    return False


def _truth_footer_region(text):
    if not isinstance(text, str):
        return ""
    starts = []
    for rx in (
        TRUTH_PROVEN_HEADER_RE,
        TRUTH_GAP_HEADER_RE,
        STATE_NEXT_HEADER_RE,
        METRICS_GATE_HEADER_RE,
        BEHAVIOR_FAIL_HEADER_RE,
    ):
        m = rx.search(text)
        if m:
            starts.append(m.start())
    if not starts:
        return ""
    return text[min(starts):]


def _has_key_value_fake_grid(text):
    if not isinstance(text, str):
        return False
    for match in KV_FAKE_GRID_KEY_RE.finditer(text):
        line = match.group(0).strip()
        # GAP: is a legitimate canonical section header. "Gap:" under that
        # section is the fake key/value row form.
        if line == "GAP:":
            continue
        return True
    return False


def _extract_gap_lines(text):
    """Walk footer GAP bullets. Mirrors _extract_proven_lines shape."""
    if not isinstance(text, str):
        return []
    m = FOOTER_HEADER_RE.search(text)
    if not m:
        return []
    footer = text[m.end():]
    lines = footer.split("\n")
    gaps = []
    in_gap = False
    for ln in lines:
        bm = _PROVEN_BULLET_RE.match(ln)
        if bm:
            section = bm.group(1).upper()
            if section == "GAP":
                in_gap = True
                tail = bm.group(2).strip()
                if tail:
                    gaps.append(tail)
            else:
                in_gap = False
        elif in_gap:
            stripped = ln.strip()
            if not stripped:
                # Pack 1D: blank line terminates GAP continuation so trailing
                # pack-tail markers (e.g. 'AWAITING APPROVAL.', 'End of pack.')
                # following a blank are not absorbed as a new GAP entry.
                in_gap = False
                continue
            if stripped.startswith("-"):
                in_gap = False
                continue
            gaps.append(stripped)
    return gaps


def _classify_gap(gap_text):
    """Return (class, reason_or_none).

    Class âˆˆ {'unfillable_reasoned', 'unfillable_no_reason',
              'fillable', 'unspecified'}.
    """
    if not isinstance(gap_text, str) or not gap_text:
        return ("unspecified", None)
    am = _NEEDS_APPROVAL_RE.search(gap_text)
    if am:
        _na_reason = am.group(1).strip()
        if _na_reason and _na_reason.lower() not in _NEEDS_APPROVAL_BAD_REASONS:
            return ("needs_approval", _na_reason)
    rm = _UNFILLABLE_REASON_RE.search(gap_text)
    if rm:
        return ("unfillable_reasoned", rm.group(1).strip())
    if _UNFILLABLE_BARE_RE.search(gap_text):
        return ("unfillable_no_reason", None)
    if _FILLABLE_KEYWORDS_RE.search(gap_text):
        return ("fillable", None)
    return ("unspecified", None)


def _flag_read():
    try:
        if not DISCOVER_FLAG.exists():
            return None
        raw = DISCOVER_FLAG.read_text(encoding="utf-8")
        return json.loads(raw)
    except Exception:
        return None


def _flag_write(payload):
    try:
        DISCOVER_FLAG.parent.mkdir(parents=True, exist_ok=True)
        tmp = DISCOVER_FLAG.with_suffix(".flag.tmp")
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        os.replace(str(tmp), str(DISCOVER_FLAG))
        return True
    except Exception:
        return False


def _flag_clear():
    try:
        if DISCOVER_FLAG.exists():
            DISCOVER_FLAG.unlink()
        return True
    except Exception:
        return False


def _flag_age_seconds(flag):
    try:
        ts = (flag or {}).get("created_at", "")
        if not ts:
            return None
        ts2 = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts2)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return None


def _ledger_anchors_with_ts(session_id, lookback=LEDGER_LOOKBACK):
    """Return list[(ts_datetime_or_None, anchor_text_str)] for session."""
    if not session_id or not LEDGER.exists():
        return []
    try:
        with LEDGER.open("r", encoding="utf-8") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200 * 1024))
            tail = f.read()
        rows = [r for r in tail.split("\n") if r.strip()]
    except Exception:
        return []
    out = []
    for line in rows[-lookback:]:
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("session_id") != session_id:
            continue
        ts_val = None
        try:
            tsraw = (row.get("ts") or "").replace("Z", "+00:00")
            if tsraw:
                ts_val = datetime.fromisoformat(tsraw)
        except Exception:
            ts_val = None
        ti = row.get("tool_input") or {}
        for k in ("command", "file_path", "path", "query", "pattern", "symbol", "description"):
            v = ti.get(k)
            if isinstance(v, str) and len(v) >= ANCHOR_MIN_LEN:
                out.append((ts_val, v))
        for k in ("stdout_tail", "stderr_tail", "error_tail"):
            v = row.get(k)
            if isinstance(v, str) and len(v) >= ANCHOR_MIN_LEN:
                out.append((ts_val, v))
    return out


def _flag_check_clear(flag, session_id, footer_text, anchors_with_ts):
    """Return (cleared: bool, reason: str)."""
    if not flag:
        return (False, "no-flag")
    if flag.get("session_id") != session_id:
        return (False, "cross-session-ignored")
    age = _flag_age_seconds(flag)
    ttl = flag.get("ttl_seconds", DISCOVER_FLAG_TTL_SEC)
    if age is not None and age > ttl:
        return (True, "ttl-expired")
    flag_gap = (flag.get("gap_text") or "").strip().lower()
    gaps = _extract_gap_lines(footer_text or "")
    for g in gaps:
        cls, _reason = _classify_gap(g)
        if cls == "unfillable_reasoned":
            gl = g.lower()
            if not flag_gap:
                return (True, "unfillable-reason-given")
            if flag_gap[:30] and flag_gap[:30] in gl:
                return (True, "unfillable-reason-matched")
            if gl[:30] and gl[:30] in flag_gap:
                return (True, "unfillable-reason-matched")
    required = (flag.get("required_action") or "").lower()
    if required:
        flag_ts = None
        try:
            tsraw = (flag.get("created_at") or "").replace("Z", "+00:00")
            if tsraw:
                flag_ts = datetime.fromisoformat(tsraw)
        except Exception:
            flag_ts = None
        for a_ts, a_text in (anchors_with_ts or []):
            if flag_ts is None or a_ts is None:
                continue
            if a_ts < flag_ts:
                continue
            if required in a_text.lower():
                return (True, "ledger-anchor-post-flag")
    return (False, "still-pending")


def _build_required_action(gap_text):
    """Derive a short required-action substring from a fillable GAP body."""
    if not isinstance(gap_text, str):
        return ""
    m = _FILLABLE_KEYWORDS_RE.search(gap_text)
    if not m:
        return gap_text[:80].strip()
    start = m.start()
    end = min(len(gap_text), start + 80)
    return gap_text[start:end].strip()


def _assistant_response_key(row):
    msg = row.get("message") or {}
    mid = msg.get("id") if isinstance(msg, dict) else ""
    req = row.get("requestId") or ""
    if mid:
        return ("message_id", mid)
    if req:
        return ("request_id", req)
    return ("uuid", row.get("uuid") or "")


def _current_assistant_span_bounds(rows):
    assistant_idxs = [i for i, r in enumerate(rows) if r.get("type") == "assistant"]
    if not assistant_idxs:
        return None
    last_pos = len(assistant_idxs) - 1
    last_idx = assistant_idxs[last_pos]
    last_key = _assistant_response_key(rows[last_idx])
    first_pos = last_pos
    while first_pos > 0 and _assistant_response_key(rows[assistant_idxs[first_pos - 1]]) == last_key:
        first_pos -= 1
    prev_idx = assistant_idxs[first_pos - 1] if first_pos > 0 else -1
    return (prev_idx, last_idx)


def current_turn_diagnostics(transcript_path):
    """Pack 1: capture transcript-shape facts for stop-gate.log diagnostics."""
    diag = {
        "transcript_path_resolved": transcript_path or "",
        "transcript_total_rows": 0,
        "transcript_assistant_rows": 0,
        "span_len": 0,
        "span_tool_use": 0,
        "span_tool_result": 0,
    }
    if not transcript_path or not os.path.exists(transcript_path):
        return diag
    try:
        rows = []
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        diag["transcript_total_rows"] = len(rows)
        aidxs = []
        for i, r in enumerate(rows):
            if r.get("type") == "assistant":
                aidxs.append(i)
                continue
            msg = r.get("message")
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                aidxs.append(i)
        diag["transcript_assistant_rows"] = len(aidxs)
        bounds = _current_assistant_span_bounds(rows)
        if bounds is not None:
            prev_idx, last_idx = bounds
            span = rows[prev_idx + 1: last_idx + 1]
            diag["span_len"] = len(span)
            tu = tr = 0
            for r in span:
                t = r.get("type")
                if t == "assistant":
                    msg = r.get("message") or {}
                    for c in (msg.get("content") or []):
                        if isinstance(c, dict) and c.get("type") == "tool_use":
                            tu += 1
                if t == "user":
                    msg = r.get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "tool_result":
                                tr += 1
            diag["span_tool_use"] = tu
            diag["span_tool_result"] = tr
    except Exception:
        pass
    return diag


# --- Pack 1B rewrite-required flag helpers ---

def _first_two_nonempty_lines(text):
    if not isinstance(text, str) or not text:
        return ("", "")
    out = []
    for ln in text.split("\n"):
        if ln.strip():
            out.append(ln)
            if len(out) >= 2:
                break
    while len(out) < 2:
        out.append("")
    return (out[0], out[1])


def _strip_one_ui_bullet(line):
    """Pack 1C: strip exactly one optional UI bullet glyph + trailing
    whitespace. Does not allow arbitrary words before the bullet, does
    not allow more than one bullet."""
    if not isinstance(line, str):
        return ""
    return _UI_BULLET_PREFIX_RE.sub("", line, count=1)


def _parse_correction_rule_ids_line(line):
    """Parse 'rule_ids: a, b, c' or 'rule_ids: ["a","b"]' or JSON-like.

    Pack 1C: tolerate one optional UI bullet prefix (e.g. 'â— rule_ids: ...').
    """
    norm = _strip_one_ui_bullet(line or "")
    m = _REWRITE_RULE_IDS_LINE_RE.match(norm)
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    parts = [p.strip().strip("'\"") for p in raw.split(",")]
    return [p for p in parts if p]


def _rewrite_correction_satisfies(text, original_rule_ids):
    """Return (ok, reason_code).

    reason_code in {'ok', 'missing-header-first-line',
                    'missing-rule-ids-second-line',
                    'rule-ids-do-not-cover-original'}.
    """
    line1, line2 = _first_two_nonempty_lines(text)
    # Pack 1C: strip exactly one optional UI bullet prefix before exact compare.
    norm1 = _strip_one_ui_bullet(line1).strip()
    if norm1 != REWRITE_HEADER_LITERAL:
        return (False, "missing-header-first-line")
    parsed = _parse_correction_rule_ids_line(line2)
    if parsed is None:
        return (False, "missing-rule-ids-second-line")
    expected = [str(r).strip() for r in (original_rule_ids or []) if str(r).strip()]
    parsed_set = set(parsed)
    for rid in expected:
        if rid not in parsed_set:
            return (False, "rule-ids-do-not-cover-original")
    return (True, "ok")


def _rewrite_flag_path_for(session_id):
    """Pack 1J-AI: per-session flag path. Empty session_id -> legacy global
    path so older callers / tests that wrote to the global file still find it.
    main() always passes session_id so production reads/writes hit the
    per-session path."""
    if not session_id:
        return REWRITE_FLAG
    return REWRITE_FLAG_DIR / f"{session_id}.flag"


def _rewrite_flag_read(session_id=""):
    """Pack 1J-AI: read per-session flag. Performs one-time legacy migration:
    if per-session file is absent but the global rewrite-required.flag exists
    AND its session_id matches, move it to the per-session path."""
    p = _rewrite_flag_path_for(session_id)
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        # Pack 1J-AI legacy migration: rescue global flag for THIS session only.
        if session_id and REWRITE_FLAG.exists() and p != REWRITE_FLAG:
            try:
                legacy = json.loads(REWRITE_FLAG.read_text(encoding="utf-8"))
                if legacy.get("session_id") == session_id:
                    REWRITE_FLAG_DIR.mkdir(parents=True, exist_ok=True)
                    REWRITE_FLAG.rename(p)
                    return legacy
            except Exception:
                pass
        return None
    except Exception:
        return None


def _rewrite_flag_write(payload, session_id=""):
    sid = session_id or (payload or {}).get("session_id", "")
    p = _rewrite_flag_path_for(sid)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".flag.tmp")
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        os.replace(str(tmp), str(p))
        return True
    except Exception:
        return False


def _rewrite_flag_clear(session_id=""):
    p = _rewrite_flag_path_for(session_id)
    try:
        if p.exists():
            p.unlink()
        return True
    except Exception:
        return False


# ----------------------------------------------------------------
# Pack 1J-A: orphan rewrite-required.flag quarantine.
# ----------------------------------------------------------------
ARCHIVE_DIR = Path(os.path.expanduser("~")) / ".claude" / "truth" / "archive"


def _archive_orphan_rewrite_flag(flag, current_session_id):
    """Move an orphan rewrite-required.flag (cross-session) to archive
    so the active session never re-writes its last_block_ts/current_rule_ids.
    Atomic write to archive then unlink original.
    Returns (archived: bool, archive_path: str)."""
    try:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        old_sid = (flag or {}).get("session_id", "unknown")
        old_created = (flag or {}).get("created_at", "")
        safe_created = "".join(c if c.isalnum() else "_" for c in old_created)
        archive_name = f"rewrite-required-{old_sid}-{safe_created}.flag.archived"
        archive_path = ARCHIVE_DIR / archive_name
        # Atomic temp + os.replace into archive dir.
        import tempfile as _tf
        fd, tmp = _tf.mkstemp(prefix=".tmp-arch-", suffix=".archived",
                              dir=str(ARCHIVE_DIR))
        try:
            payload = dict(flag or {})
            payload["_archived_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            payload["_archived_by_session"] = current_session_id or ""
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, sort_keys=True, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, str(archive_path))
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except Exception:
                pass
            return (False, "")
        # Remove the live flag file last so a crash mid-archive leaves
        # the original flag on disk (failure-safe).
        try:
            if REWRITE_FLAG.exists():
                REWRITE_FLAG.unlink()
        except Exception:
            return (False, str(archive_path))
        return (True, str(archive_path))
    except Exception:
        return (False, "")
# ----------------------------------------------------------------
# end Pack 1J-A archive helper
# ----------------------------------------------------------------


def _rewrite_flag_age_seconds(flag):
    try:
        ts = (flag or {}).get("created_at", "")
        if not ts:
            return None
        ts2 = ts.replace("Z", "+00:00")
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts2)).total_seconds()
    except Exception:
        return None


def _rewrite_required_rule_id(reason_code):
    return {
        "missing-header-first-line": "evidence.rewrite.required.not-first-line",
        "missing-rule-ids-second-line": "evidence.rewrite.required.no-rule-ids",
        "rule-ids-do-not-cover-original": "evidence.rewrite.required.rule-ids-incomplete",
    }.get(reason_code, "evidence.rewrite.required.not-satisfied")


def _rewrite_required_fix(reason_code, original_rule_ids):
    if reason_code == "missing-header-first-line":
        return ("first non-empty line MUST be exactly: "
                f"{REWRITE_HEADER_LITERAL!r}")
    if reason_code == "missing-rule-ids-second-line":
        return ("second non-empty line MUST start with 'rule_ids:' followed "
                "by a comma list of rule ids covering: "
                + ", ".join(original_rule_ids or []))
    if reason_code == "rule-ids-do-not-cover-original":
        return ("rule_ids must include every original rule id: "
                + ", ".join(original_rule_ids or []))
    return ("rewrite correction header and rule_ids required before next "
            "answer can pass")


# Rewrite validation is exact-contract: original failed gates plus explicitly
# named footer/template guardrails only. No broad evidence.* prefix catch-all.
REWRITE_LANE_TEMPLATE_RULES = {
    "evidence.schema.canonical-footer.always-required",
    "evidence.schema.canonical-section-missing",
    "evidence.schema.box-table-rendering.required",
    "evidence.schema.v3.fake-key-value-grid",
    "evidence.schema.v3.fake-box-drawing-grid",
    "evidence.schema.legacy-truth-bullets-disallowed",
    "evidence.schema.section.behavior-fail-not-last",
    "evidence.schema.metrics-gate.missing",
    "evidence.schema.metrics-gate.row-missing",
    "evidence.schema.metrics-gate.blank-placeholder",
    "evidence.schema.metrics-gate.required-cell-too-wide",
    "evidence.schema.metrics-gate.numeric-pass-mismatch",
    "evidence.schema.metrics-gate.fail-with-implementation-claim",
    "evidence.schema.blank-cell",
    "evidence.metrics-gate.invalid-pass-fail-value",
    "evidence.metrics-gate.calculated-mismatch",
    "evidence.metrics-gate.fail-return-disallowed",
    "evidence.metrics-gate.fillable-gap-return-disallowed",
    "evidence.gap.fillable-blocks-pass-forces-fail",
    "evidence.partial.objective-critical-caps-confidence",
    "evidence.behavior-fail.blocks-pass-forces-fail",
    "evidence.body-blocking-claim.requires-behavior-fail",
}


def _rewrite_lane_filter_violations(violations, original_rule_ids):
    """For an active rewrite, keep exact original gates plus named template gates."""
    original = {str(r).strip() for r in (original_rule_ids or []) if str(r).strip()}
    out = []
    for v in violations or []:
        rid = str((v or {}).get("rule") or "")
        if rid in original or rid in REWRITE_LANE_TEMPLATE_RULES:
            out.append(v)
    return out


def current_turn_used_tools(transcript_path):
    if not transcript_path or not os.path.exists(transcript_path):
        return (False, "fallback")
    try:
        rows = []
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        bounds = _current_assistant_span_bounds(rows)
        if bounds is None:
            return (False, "transcript")
        prev_idx, last_idx = bounds
        span = rows[prev_idx + 1: last_idx + 1]
        for r in span:
            t = r.get("type")
            if t == "assistant":
                msg = r.get("message") or {}
                for c in (msg.get("content") or []):
                    if isinstance(c, dict) and c.get("type") == "tool_use":
                        return (True, "transcript")
            if t == "user":
                msg = r.get("message") or {}
                content = msg.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "tool_result":
                            return (True, "transcript")
        return (False, "transcript")
    except Exception:
        return (False, "fallback")


def has_risky_language(text):
    scan_text = strip_non_claim_regions(text)
    for _, rx in DANGER_CATEGORIES:
        if rx.search(scan_text):
            return True
    return False


def log_gate(row):
    try:
        GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with GATE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        if GATE_LOG.stat().st_size > MAX_LOG_BYTES:
            with GATE_LOG.open("rb") as f:
                f.seek(-(KEEP_LOG_BYTES), 2)
                tail_bytes = f.read()
            nl = tail_bytes.find(b"\n")
            if nl >= 0:
                tail_bytes = tail_bytes[nl + 1:]
            with GATE_LOG.open("wb") as f:
                f.write(tail_bytes)
    except Exception:
        pass


# ================================================================
# Pack 1J-W: schema v2 helpers (Truth footer + Metrics Gate).
# ================================================================

def _is_implementation_claim(scan_text):
    """True if answer carries closure/behavior/verification/cleanliness/
    universal/confidence language that triggers schema v2 enforcement."""
    if not isinstance(scan_text, str) or not scan_text:
        return False
    for _, rx in DANGER_CATEGORIES:
        if rx.search(scan_text):
            return True
    return False


def _schema_v2_required(scan_text):
    """Schema v2 mandatory ONLY when env override set. Closure-language-based
    auto-trigger deferred (would break 134 legacy fixtures using closure
    language without v2 schema). Opt-in via TRUTH_SCHEMA_V2_REQUIRED=1."""
    return (not _schema_v3_required()) and os.environ.get("TRUTH_SCHEMA_V2_REQUIRED") == "1"


def _section_present(text, section):
    """True if `- <section>:` bullet present anywhere in TRUTH footer."""
    if not isinstance(text, str):
        return False
    m = FOOTER_HEADER_RE.search(text)
    if not m:
        return False
    footer = text[m.end():]
    pat = re.compile(r"(?im)^\s*-\s*" + re.escape(section) + r"\s*:")
    return bool(pat.search(footer))


def _extract_section_lines(text, section):
    """Return list of body strings for bullets of given section.
    Multi-line bullet bodies (sub-bullets indented under the section header)
    are joined into one string per top-level bullet."""
    if not isinstance(text, str):
        return []
    m = FOOTER_HEADER_RE.search(text)
    if not m:
        return []
    footer = text[m.end():]
    lines = footer.split("\n")
    out = []
    in_target = False
    current = []
    for ln in lines:
        bm = FOOTER_BULLET_V2_RE.match(ln)
        if bm:
            if in_target and current:
                out.append("\n".join(current).strip())
                current = []
            sect = bm.group(1).upper()
            if sect == section.upper():
                in_target = True
                tail = ln.split(":", 1)[1].strip() if ":" in ln else ""
                if tail:
                    current.append(tail)
            else:
                in_target = False
        elif in_target:
            stripped = ln.strip()
            if not stripped:
                continue
            if METRICS_GATE_HEADER_RE.match(ln):
                if current:
                    out.append("\n".join(current).strip())
                    current = []
                in_target = False
                break
            # Strip leading sub-bullet marker so body comparisons (e.g.
            # "none blocking") are not foiled by "- none blocking".
            if stripped.startswith("- "):
                stripped = stripped[2:].strip()
            elif stripped.startswith("-"):
                stripped = stripped[1:].strip()
            current.append(stripped)
    if in_target and current:
        out.append("\n".join(current).strip())
    return out


def _has_metrics_gate(text):
    """True if `METRICS GATE:` header appears after TRUTH_PROVEN."""
    if not isinstance(text, str):
        return False
    m_v3 = TRUTH_PROVEN_HEADER_RE.search(text)
    if not m_v3:
        return False
    return bool(METRICS_GATE_HEADER_RE.search(text[m_v3.end():]))


def _parse_metrics_gate(text):
    """Parse METRICS GATE table rows. Returns dict with 4 keys; missing
    rows = None. Numeric values stripped of '%' and parsed as int when
    possible; non-numeric (e.g. 'PASS', '__') passed through as str.
    Pack 1J-BN: only canonical `TRUTH_PROVEN:` anchors metrics parsing."""
    out = {"GAPS_FILLED": None, "DISCOVERY": None,
           "BUILD_CONFIDENCE": None, "METRICS_GATE": None}
    if not isinstance(text, str):
        return out
    m_v3 = TRUTH_PROVEN_HEADER_RE.search(text)
    if not m_v3:
        return out
    body = text[m_v3.end():]
    mh = METRICS_GATE_HEADER_RE.search(body)
    if not mh:
        return out
    block = body[mh.end():]
    for rm in METRICS_ROW_RE.finditer(block):
        key = rm.group(1).upper()
        actual = rm.group(2).strip()
        if key in out:
            if actual == "__" or actual == "__%":
                out[key] = "__"
            else:
                num_match = re.match(r"^(\d+)\s*%?\s*$", actual)
                if num_match:
                    out[key] = int(num_match.group(1))
                else:
                    out[key] = actual
    return out


def _metrics_gate_internally_consistent(metrics):
    """METRICS_GATE PASS requires numeric thresholds satisfied."""
    if not isinstance(metrics, dict):
        return False
    g = metrics.get("GAPS_FILLED")
    d = metrics.get("DISCOVERY")
    b = metrics.get("BUILD_CONFIDENCE")
    if not isinstance(g, int) or not isinstance(d, int) or not isinstance(b, int):
        return False
    return (g >= METRICS_REQUIRED_GAPS_FILLED
            and d >= METRICS_REQUIRED_DISCOVERY
            and b >= METRICS_REQUIRED_BUILD_CONFIDENCE)


def _partial_bullet_well_formed(body):
    """PARTIAL bullet must address: what proven / what not proven /
    why partial / what closes it. Lenient keyword check."""
    if not isinstance(body, str) or not body.strip():
        return False
    s = body.lower()
    has_proven = ("proven" in s or "tested" in s or "shown" in s)
    has_not_proven = ("not proven" in s or "not tested" in s
                      or "not shown" in s or "not yet" in s
                      or "unproven" in s)
    has_why = ("why" in s or "because" in s or "reason" in s
               or "due to" in s or "since" in s or "partial" in s)
    has_closes = ("closes" in s or "close" in s or "next" in s
                  or "fix" in s or "resolve" in s or "address" in s)
    return has_proven and has_not_proven and has_why and has_closes


def _gap_bullet_well_formed(body):
    """GAP bullet must say: fillable or unfillable + missing proof + next action."""
    if not isinstance(body, str) or not body.strip():
        return False
    s = body.lower()
    has_class = ("fillable" in s or "unfillable" in s)
    has_missing_proof = ("missing proof" in s or "missing" in s
                        or "no proof" in s or "lacks" in s
                        or "not yet" in s or "absent" in s)
    has_next = ("next" in s or "action" in s or "read" in s
                or "test" in s or "run" in s or "check" in s)
    return has_class and has_missing_proof and has_next


def _behavior_fail_is_last_truth_section(text):
    """BEHAVIOR_FAIL must be the LAST schema-v2 section after METRICS GATE
    (or end-of-text). Returns True if BEHAVIOR_FAIL absent (caller handles
    that separately)."""
    if not isinstance(text, str):
        return True
    m = FOOTER_HEADER_RE.search(text)
    if not m:
        return True
    footer = text[m.end():]
    bf_idx = -1
    last_section_idx = -1
    last_section_name = None
    for sm in FOOTER_BULLET_V2_RE.finditer(footer):
        name = sm.group(1).upper()
        if name == "BEHAVIOR_FAIL":
            bf_idx = sm.start()
        last_section_idx = sm.start()
        last_section_name = name
    if bf_idx < 0:
        return True
    if last_section_name != "BEHAVIOR_FAIL":
        return False
    mg = METRICS_GATE_HEADER_RE.search(footer)
    if mg and bf_idx < mg.start():
        return False
    return True


# Pack 1J-Y: metrics-gate-failed.flag lifecycle helpers.

def _metrics_gate_failed_write(session_id, rule_ids):
    """Write or refresh metrics-gate-failed.flag for the given session."""
    try:
        METRICS_GATE_FAILED_FLAG.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "session_id": session_id or "",
            "rule_ids": list(rule_ids or []),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ttl_seconds": METRICS_GATE_FAILED_TTL_SEC,
        }
        METRICS_GATE_FAILED_FLAG.write_text(json.dumps(payload), encoding="utf-8")
        return True
    except Exception:
        return False


def _metrics_gate_failed_clear():
    try:
        if METRICS_GATE_FAILED_FLAG.exists():
            METRICS_GATE_FAILED_FLAG.unlink()
            return True
    except Exception:
        return False
    return False


def _is_schema_v2_rule(rule_id):
    if not isinstance(rule_id, str):
        return False
    return (rule_id.startswith("evidence.schema.")
            or rule_id == "evidence.partial.missing-why-closure"
            or rule_id == "evidence.gap.missing-fillable-or-action")


def _answer_has_valid_metrics_gate_pass(text):
    """True iff the answer's METRICS GATE table is internally consistent
    AND declares METRICS_GATE PASS. Used to clear the failed flag."""
    if _short_truth_footer_present(text):
        return True
    if not _has_metrics_gate(text):
        return False
    metrics = _parse_metrics_gate(text)
    if metrics.get("METRICS_GATE") != "PASS":
        return False
    return _metrics_gate_internally_consistent(metrics)


# ================================================================
# Pack 1J-AM schema v3: strict-grid parsers + machine-calculated metrics.
# ================================================================

def _schema_v3_required():
    """Pack 1J-BN: canonical pipe-grid schema is always required."""
    return True


def _parse_truth_proven_grid(text):
    """Returns list of dicts: {id, claim, ledger_anchor/proof, verified}."""
    if not isinstance(text, str):
        return []
    m = TRUTH_PROVEN_HEADER_RE.search(text)
    if not m:
        return []
    body = text[m.end():]
    end = TRUTH_GAP_HEADER_RE.search(body) or TRUTH_PARTIAL_HEADER_RE.search(body)
    if end:
        body = body[:end.start()]
    rows = []
    for rm in TRUTH_PROVEN_ROW_RE.finditer(body):
        rows.append({
            "id": rm.group(1),
            "claim": rm.group(2).strip(),
            "ledger_anchor": rm.group(3).strip(),
            "verified": rm.group(4).strip().upper(),
        })
    return rows


def _parse_truth_partial_grid(text):
    """Returns list of dicts with PARTIAL row fields."""
    if not isinstance(text, str):
        return []
    m = TRUTH_PARTIAL_HEADER_RE.search(text)
    if not m:
        return []
    body = text[m.end():]
    end = TRUTH_GAP_HEADER_RE.search(body)
    if end:
        body = body[:end.start()]
    rows = []
    for rm in TRUTH_PARTIAL_ROW_RE.finditer(body):
        rows.append({
            "id": rm.group(1),
            "what_proven": rm.group(2).strip(),
            "what_not_proven": rm.group(3).strip(),
            "why_partial": rm.group(4).strip(),
            "what_closes": rm.group(5).strip(),
            "objective_critical": rm.group(6).strip().upper(),
        })
    return rows


def _parse_truth_gap_grid(text):
    """Returns list of dicts with GAP row fields."""
    if not isinstance(text, str):
        return []
    m = TRUTH_GAP_HEADER_RE.search(text)
    if not m:
        return []
    body = text[m.end():]
    rows = []
    for rm in TRUTH_GAP_ROW_RE.finditer(body):
        rows.append({
            "id": rm.group(1),
            "gap": rm.group(2).strip(),
            "fillable": rm.group(3).strip().upper(),
            "missing_proof": rm.group(4).strip(),
            "next_action": rm.group(5).strip(),
            "blocks_pass": rm.group(6).strip().upper(),
        })
    return rows


def _section_body(text, header_re, stop_res):
    if not isinstance(text, str):
        return ""
    m = header_re.search(text)
    if not m:
        return ""
    body = text[m.end():]
    stops = [sr.search(body) for sr in stop_res]
    stops = [s for s in stops if s]
    if stops:
        body = body[:min(s.start() for s in stops)]
    return body


def _simple_pipe_rows(body, row_re, header_names):
    rows = []
    for rm in row_re.finditer(body or ""):
        cells = [g.strip() for g in rm.groups()]
        first = cells[0].strip().lower()
        if first in header_names:
            continue
        if all(set(c.replace(":", "").strip()) <= {"-"} for c in cells):
            continue
        if any(not c.strip() for c in cells):
            continue
        rows.append(cells)
    return rows


def _parse_current_state_grid(text):
    body = _section_body(text, CURRENT_STATE_HEADER_RE,
                         (NEXT_HEADER_RE, BEHAVIOR_FAIL_HEADER_RE, METRICS_GATE_HEADER_RE))
    return _simple_pipe_rows(body, CURRENT_STATE_ROW_RE, {"item"})


def _parse_next_grid(text):
    body = _section_body(text, NEXT_HEADER_RE,
                         (BEHAVIOR_FAIL_HEADER_RE, METRICS_GATE_HEADER_RE))
    return _simple_pipe_rows(body, NEXT_ROW_RE, {"step"})


def _parse_state_next_grid(text):
    body = _section_body(text, STATE_NEXT_HEADER_RE,
                         (BEHAVIOR_FAIL_HEADER_RE, METRICS_GATE_HEADER_RE))
    return _simple_pipe_rows(body, STATE_NEXT_ROW_RE, {"state"})


def _parse_behavior_fail_grid(text):
    body = _section_body(text, BEHAVIOR_FAIL_HEADER_RE, (METRICS_GATE_HEADER_RE,))
    rows = []
    for rm in BEHAVIOR_FAIL_ROW_RE.finditer(body):
        rows.append({
            "id": rm.group(1).strip(),
            "failure": rm.group(2).strip(),
            "proof": rm.group(3).strip(),
            "blocks_pass": rm.group(4).strip().upper(),
        })
    return rows


def _is_separator_cells(cells):
    return all(set(c.replace(":", "").strip()) <= {"-"} for c in cells)


def _parse_short_truth_grid(text):
    body = _section_body(text, TRUTH_PROVEN_HEADER_RE,
                         (TRUTH_GAP_HEADER_RE, TRUTH_PARTIAL_HEADER_RE))
    rows = []
    for rm in SHORT_TRUTH_ROW_RE.finditer(body):
        cells = [rm.group(1).strip(), rm.group(2).strip(), rm.group(3).strip()]
        if cells[0].lower() == "claim" or _is_separator_cells(cells):
            continue
        if cells[0] and cells[1] and cells[2].upper() == "YES":
            rows.append({"claim": cells[0], "proof": cells[1], "verified": "YES"})
    return rows


def _parse_short_gap_grid(text):
    body = _section_body(text, TRUTH_GAP_HEADER_RE,
                         (STATE_NEXT_HEADER_RE, BEHAVIOR_FAIL_HEADER_RE, METRICS_GATE_HEADER_RE))
    rows = []
    for rm in SHORT_GAP_ROW_RE.finditer(body):
        cell = rm.group(1).strip()
        if cell.lower() == "gap" or _is_separator_cells([cell]):
            continue
        if cell.lower() == "none":
            rows.append({"gap": cell})
    return rows


def _parse_short_state_next_grid(text):
    body = _section_body(text, STATE_NEXT_HEADER_RE,
                         (BEHAVIOR_FAIL_HEADER_RE, METRICS_GATE_HEADER_RE))
    rows = []
    for rm in SHORT_STATE_NEXT_ROW_RE.finditer(body):
        cell = rm.group(1).strip()
        if cell.lower() in ("state / next", "state_next") or _is_separator_cells([cell]):
            continue
        if cell:
            rows.append({"state_next": cell})
    return rows


def _parse_short_metrics_gate(text):
    body = _section_body(text, METRICS_GATE_HEADER_RE, (BEHAVIOR_FAIL_HEADER_RE,))
    rows = []
    for rm in SHORT_METRICS_ROW_RE.finditer(body):
        status = rm.group(1).strip().upper()
        rows.append({"status": status})
    return rows


def _parse_short_behavior_fail_grid(text):
    body = _section_body(text, BEHAVIOR_FAIL_HEADER_RE, (METRICS_GATE_HEADER_RE,))
    rows = []
    for rm in SHORT_BEHAVIOR_FAIL_ROW_RE.finditer(body):
        cells = [rm.group(1).strip(), rm.group(2).strip()]
        if cells[0].lower() == "failure" or _is_separator_cells(cells):
            continue
        if cells[0].lower() == "none" and cells[1].upper() == "NO":
            rows.append({"failure": cells[0], "blocks_pass": "NO"})
    return rows


def _short_truth_footer_present(text):
    if not isinstance(text, str):
        return False
    return (
        bool(TRUTH_PROVEN_HEADER_RE.search(text) and _parse_short_truth_grid(text))
        and bool(TRUTH_GAP_HEADER_RE.search(text) and _parse_short_gap_grid(text))
        and bool(STATE_NEXT_HEADER_RE.search(text) and _parse_short_state_next_grid(text))
        and bool(METRICS_GATE_HEADER_RE.search(text) and _parse_short_metrics_gate(text))
        and bool(BEHAVIOR_FAIL_HEADER_RE.search(text) and _parse_short_behavior_fail_grid(text))
    )


def _calculate_short_metrics_from_grids(text):
    proven = _parse_short_truth_grid(text)
    gap = _parse_short_gap_grid(text)
    behavior = _parse_short_behavior_fail_grid(text)
    reasons = []

    discovery = 100 if proven else 0
    if not proven:
        reasons.append("SHORT TRUTH has no verified row")

    gap_none = any((g.get("gap") or "").strip().lower() == "none" for g in gap)
    gaps_filled = 100 if gap_none else 0
    if not gap_none:
        reasons.append("SHORT GAP is not none")

    behavior_none = any(
        (b.get("failure") or "").strip().lower() == "none"
        and (b.get("blocks_pass") or "").strip().upper() == "NO"
        for b in behavior
    )
    if not behavior_none:
        reasons.append("SHORT BEHAVIOR_FAIL is not none/NO")

    build_confidence = 100 if discovery == 100 and gaps_filled == 100 and behavior_none else 94
    metrics_gate = (
        "PASS"
        if gaps_filled >= 100 and discovery >= 100 and build_confidence >= 95
        else "FAIL"
    )
    return {
        "GAPS_FILLED": gaps_filled,
        "DISCOVERY": discovery,
        "BUILD_CONFIDENCE": build_confidence,
        "METRICS_GATE": metrics_gate,
        "reasons": reasons,
    }


def _calculate_metrics_from_grids(text):
    """Pack 1J-AM machine calculator. Reads TRUTH_PROVEN/PARTIAL/GAP grids
    and returns deterministic {GAPS_FILLED, DISCOVERY, BUILD_CONFIDENCE,
    METRICS_GATE} dict + reason list for each cap/fail."""
    proven = _parse_truth_proven_grid(text)
    partial = _parse_truth_partial_grid(text)
    gap = _parse_truth_gap_grid(text)
    behavior = _parse_behavior_fail_grid(text)
    reasons = []

    # DISCOVERY: every PROVEN row must have non-empty ledger_anchor + Verified=YES.
    proven_ok = 0
    proven_total = len(proven)
    for p in proven:
        if p["ledger_anchor"] and p["verified"] == "YES":
            proven_ok += 1
        else:
            reasons.append(f"PROVEN {p['id']} missing ledger_anchor or Verified!=YES")
    discovery = 100 if proven_total == 0 else int(100 * proven_ok / proven_total)

    # GAPS_FILLED: any blocks_pass GAP forces below 100.
    blocking_gap = [g for g in gap if g["blocks_pass"] == "YES"]
    if blocking_gap:
        for g in blocking_gap:
            reasons.append(f"GAP {g['id']} blocks_pass=YES")
        gaps_filled = 0
    else:
        gaps_filled = 100

    # BUILD_CONFIDENCE: cap at 94 on objective-critical PARTIAL or any blocking issue.
    cap = 100
    critical_partials = [p for p in partial if p["objective_critical"] == "YES"]
    if critical_partials:
        for p in critical_partials:
            reasons.append(f"PARTIAL {p['id']} objective_critical=YES caps BUILD_CONFIDENCE 94")
        cap = 94
    blocking_behavior = [b for b in behavior if b["blocks_pass"] == "YES"]
    if blocking_behavior:
        for b in blocking_behavior:
            reasons.append(f"BEHAVIOR_FAIL {b['id']} blocks_pass=YES")
        cap = 94
    if blocking_gap:
        cap = min(cap, 94)
    if proven_total > 0 and proven_ok < proven_total:
        cap = min(cap, 94)
    build_confidence = cap

    # METRICS_GATE
    if (gaps_filled >= 100 and discovery >= 100 and build_confidence >= 95):
        metrics_gate = "PASS"
    else:
        metrics_gate = "FAIL"

    return {
        "GAPS_FILLED": gaps_filled,
        "DISCOVERY": discovery,
        "BUILD_CONFIDENCE": build_confidence,
        "METRICS_GATE": metrics_gate,
        "reasons": reasons,
    }


def _displayed_metrics_match_calculated(text, calculated):
    """True iff displayed METRICS GATE row values equal calculated values."""
    displayed = _parse_metrics_gate(text)
    for k in ("GAPS_FILLED", "DISCOVERY", "BUILD_CONFIDENCE"):
        d = displayed.get(k)
        c = calculated.get(k)
        if not isinstance(d, int) or d != c:
            return False
    if displayed.get("METRICS_GATE") != calculated.get("METRICS_GATE"):
        return False
    return True


def _honest_terminal_fail_allowed(text, displayed=None, calculated=None):
    """Allow an honest red report to return without laundering it green.

    This is not an audit-mode bypass. It only applies when the displayed
    metrics exactly match the machine calculation, a blocking BF/GAP row is
    present, no closure claim is made, and no fillable blocking GAP is being
    skipped.
    """
    if not isinstance(text, str) or not _has_metrics_gate(text):
        return False
    displayed = displayed or _parse_metrics_gate(text)
    calculated = calculated or _calculate_metrics_from_grids(text)
    if displayed.get("METRICS_GATE") != "FAIL" or calculated.get("METRICS_GATE") != "FAIL":
        return False
    if not _displayed_metrics_match_calculated(text, calculated):
        return False
    claim_text = strip_non_claim_regions(text)
    if CLOSURE_RE.search(claim_text):
        return False
    gap_rows = _parse_truth_gap_grid(text)
    behavior_rows = _parse_behavior_fail_grid(text)
    blocking_gap = [g for g in gap_rows if g["blocks_pass"] == "YES"]
    blocking_behavior = [b for b in behavior_rows if b["blocks_pass"] == "YES"]
    if not blocking_gap and not blocking_behavior:
        return False
    if BODY_BLOCKING_FINDING_RE.search(claim_text) and not (blocking_gap or blocking_behavior):
        return False
    for g in gap_rows:
        if g["blocks_pass"] == "YES" and g["fillable"] == "YES":
            cls, _r = _classify_gap(g["gap"])
            if cls != "needs_approval":
                return False
    return True


def _displayed_short_metrics_match_calculated(text, calculated):
    rows = _parse_short_metrics_gate(text)
    if not rows:
        return False
    return rows[-1].get("status") == calculated.get("METRICS_GATE")


def _template_metrics_telemetry(text):
    if _short_truth_footer_present(text):
        calculated = _calculate_short_metrics_from_grids(text)
        displayed_rows = _parse_short_metrics_gate(text)
        displayed = (
            {"METRICS_GATE": displayed_rows[-1].get("status")}
            if displayed_rows else {}
        )
        return {
            "template_mode": "SHORT",
            "template_mode_reason": "short-footer",
            "metrics_calculated": calculated,
            "metrics_displayed": displayed,
            "metrics_match": _displayed_short_metrics_match_calculated(text, calculated),
        }
    if has_footer(text):
        calculated = _calculate_metrics_from_grids(text) if _has_metrics_gate(text) else {}
        displayed = _parse_metrics_gate(text) if _has_metrics_gate(text) else {}
        return {
            "template_mode": "FULL",
            "template_mode_reason": "canonical-full-footer",
            "metrics_calculated": calculated,
            "metrics_displayed": displayed,
            "metrics_match": (
                _displayed_metrics_match_calculated(text, calculated)
                if calculated else False
            ),
        }
    return {
        "template_mode": "NONE",
        "template_mode_reason": "no-canonical-footer",
        "metrics_calculated": {},
        "metrics_displayed": {},
        "metrics_match": False,
    }


def _has_truth_structure_candidate(text):
    if not isinstance(text, str):
        return False
    return any(rx.search(text) for rx in (
        FOOTER_HEADER_RE,
        TRUTH_PROVEN_HEADER_RE,
        TRUTH_PARTIAL_HEADER_RE,
        TRUTH_GAP_HEADER_RE,
        STATE_NEXT_HEADER_RE,
        CURRENT_STATE_HEADER_RE,
        NEXT_HEADER_RE,
        BEHAVIOR_FAIL_HEADER_RE,
        METRICS_GATE_HEADER_RE,
    ))


def _canonical_truth_footer_present(text):
    if not isinstance(text, str):
        return False
    if _short_truth_footer_present(text):
        return True
    return (
        bool(TRUTH_PROVEN_HEADER_RE.search(text) and _parse_truth_proven_grid(text))
        and bool(TRUTH_GAP_HEADER_RE.search(text) and _parse_truth_gap_grid(text))
        and bool(STATE_NEXT_HEADER_RE.search(text) and _parse_state_next_grid(text))
        and bool(BEHAVIOR_FAIL_HEADER_RE.search(text) and _parse_behavior_fail_grid(text))
        and _has_metrics_gate(text)
    )


def _canonical_behavior_fail_is_last_section(text):
    if not isinstance(text, str):
        return True
    bf = BEHAVIOR_FAIL_HEADER_RE.search(text)
    if not bf:
        return True
    mg = METRICS_GATE_HEADER_RE.search(text)
    if mg and bf.start() < mg.start():
        return False
    return True


def _canonical_schema_violations(text):
    violations = []
    if not _has_truth_structure_candidate(text):
        return violations

    footer_region = _truth_footer_region(text)
    if _has_blank_separated_pipe_table(footer_region):
        violations.append({
            "rule": "evidence.schema.box-table-rendering.required",
            "fix": ("Truth Gate footer must be Markdown pipe tables that render as Claude CLI box tables. "
                    "Remove the blank line between each pipe header row and its dash separator row."),
        })
    if _has_key_value_fake_grid(footer_region):
        violations.append({
            "rule": "evidence.schema.v3.fake-key-value-grid",
            "fix": "Truth Gate footer sections must use box-table-rendering Markdown pipe-grid rows, not standalone key/value labels.",
        })
    if BOX_DRAWING_RE.search(footer_region):
        violations.append({
            "rule": "evidence.schema.v3.fake-box-drawing-grid",
            "fix": "Truth Gate footer sections must use Markdown pipe rows that render as box tables, not literal box-drawing glyph bytes.",
        })

    if FOOTER_BULLET_RE.search(text) or FOOTER_BULLET_V2_RE.search(text):
        violations.append({
            "rule": "evidence.schema.legacy-truth-bullets-disallowed",
            "fix": "Use canonical pipe-grid sections only; legacy TRUTH bullets are not accepted.",
        })

    if _short_truth_footer_present(text):
        if not _canonical_behavior_fail_is_last_section(text):
            violations.append({
                "rule": "evidence.schema.section.behavior-fail-not-last",
                "fix": "BEHAVIOR_FAIL must be the final canonical section after BUILD METRICS GATE.",
            })
        if METRICS_GATE_HEADER_RE.search(text):
            short_calculated = _calculate_short_metrics_from_grids(text)
            short_metrics = _parse_short_metrics_gate(text)
            if not _displayed_short_metrics_match_calculated(text, short_calculated):
                violations.append({
                    "rule": "evidence.metrics-gate.calculated-mismatch",
                    "match": (
                        f"calculated={short_calculated['METRICS_GATE']}/"
                        f"{short_calculated['GAPS_FILLED']}/"
                        f"{short_calculated['DISCOVERY']}/"
                        f"{short_calculated['BUILD_CONFIDENCE']}"
                    ),
                    "fix": "Displayed SHORT BUILD METRICS GATE status does not match calculated footer state.",
                })
            if not short_metrics or short_metrics[-1].get("status") != "PASS":
                violations.append({
                    "rule": "evidence.metrics-gate.fail-return-disallowed",
                    "fix": "SHORT BUILD METRICS GATE must be PASS for a final simple answer.",
                })
        return violations

    required_sections = (
        ("TRUTH", TRUTH_PROVEN_HEADER_RE, _parse_truth_proven_grid),
        ("GAP", TRUTH_GAP_HEADER_RE, _parse_truth_gap_grid),
        ("STATE_NEXT", STATE_NEXT_HEADER_RE, _parse_state_next_grid),
        ("BEHAVIOR_FAIL", BEHAVIOR_FAIL_HEADER_RE, _parse_behavior_fail_grid),
    )
    for section_name, header_re, parser in required_sections:
        if not header_re.search(text) or not parser(text):
            violations.append({
                "rule": "evidence.schema.canonical-section-missing",
                "match": section_name,
                "fix": f"{section_name} must be present as a literal Markdown pipe table with at least one data row.",
            })

    canonical_headers = (
        (TRUTH_PROVEN_HEADER_RE, _parse_truth_proven_grid, "TRUTH"),
        (TRUTH_GAP_HEADER_RE, _parse_truth_gap_grid, "GAP"),
        (STATE_NEXT_HEADER_RE, _parse_state_next_grid, "STATE_NEXT"),
        (BEHAVIOR_FAIL_HEADER_RE, _parse_behavior_fail_grid, "BEHAVIOR_FAIL"),
    )
    for header_re, parser, section_name in canonical_headers:
        hm = header_re.search(text)
        if hm and not parser(text):
            tail = text[hm.end():hm.end() + 2000]
            if _has_key_value_fake_grid(tail):
                violations.append({
                    "rule": "evidence.schema.v3.fake-key-value-grid",
                    "match": section_name,
                    "fix": f"{section_name} must use literal Markdown pipe-table grid form.",
                })
            if BOX_DRAWING_RE.search(tail):
                violations.append({
                    "rule": "evidence.schema.v3.fake-box-drawing-grid",
                    "match": section_name,
                    "fix": f"{section_name} must use literal Markdown pipe-table grid form, not box-drawing glyphs.",
                })

    if INVALID_CELL_RE.search(text):
        for v in INVALID_CELL_VALUES:
            if re.search(r"\|\s*" + re.escape(v) + r"\s*\|", text):
                violations.append({
                    "rule": "evidence.metrics-gate.invalid-pass-fail-value",
                    "match": v,
                    "fix": f"Cell value '{v}' is not allowed. Use PASS/FAIL or YES/NO as required.",
                })

    if re.search(r"\|[ \t]*\|", text):
        violations.append({
            "rule": "evidence.schema.blank-cell",
            "fix": "Canonical pipe-grid cells must not be blank.",
        })

    if not _has_metrics_gate(text):
        violations.append({
            "rule": "evidence.schema.metrics-gate.missing",
            "fix": "Canonical Truth footer requires BUILD METRICS GATE table.",
        })
        return violations

    if not _canonical_behavior_fail_is_last_section(text):
        violations.append({
            "rule": "evidence.schema.section.behavior-fail-not-last",
            "fix": "BEHAVIOR_FAIL must be the final canonical section after BUILD METRICS GATE.",
        })

    if METRICS_GATE_WIDE_REQUIRED_RE.search(text):
        violations.append({
            "rule": "evidence.schema.metrics-gate.required-cell-too-wide",
            "fix": "Use compact METRICS_GATE Required cell: PASS only if all above pass.",
        })

    metrics = _parse_metrics_gate(text)
    for row in ("GAPS_FILLED", "DISCOVERY", "BUILD_CONFIDENCE", "METRICS_GATE"):
        if metrics.get(row) is None:
            violations.append({
                "rule": "evidence.schema.metrics-gate.row-missing",
                "match": row,
                "fix": "BUILD METRICS GATE row " + row + " missing or malformed.",
            })
    if METRIC_BLANK_RE.search(text):
        violations.append({
            "rule": "evidence.schema.metrics-gate.blank-placeholder",
            "fix": "Replace `__` placeholder with concrete number / PASS / FAIL.",
        })
    for rm in METRICS_ROW_ANY_RE.finditer(text):
        pf = rm.group(3).strip()
        if pf not in ("PASS", "FAIL"):
            violations.append({
                "rule": "evidence.metrics-gate.invalid-pass-fail-value",
                "match": pf,
                "fix": "BUILD METRICS GATE Pass/Fail column must be PASS or FAIL.",
            })

    gate_decl = metrics.get("METRICS_GATE")
    if gate_decl == "PASS" and not _metrics_gate_internally_consistent(metrics):
        violations.append({
            "rule": "evidence.schema.metrics-gate.numeric-pass-mismatch",
            "fix": ("METRICS_GATE PASS requires GAPS_FILLED>=" + str(METRICS_REQUIRED_GAPS_FILLED)
                    + " AND DISCOVERY>=" + str(METRICS_REQUIRED_DISCOVERY)
                    + " AND BUILD_CONFIDENCE>=" + str(METRICS_REQUIRED_BUILD_CONFIDENCE) + "."),
        })

    proven_rows = _parse_truth_proven_grid(text)
    partial_rows = _parse_truth_partial_grid(text)
    gap_rows = _parse_truth_gap_grid(text)
    behavior_rows = _parse_behavior_fail_grid(text)
    for p in proven_rows:
        if not p["ledger_anchor"]:
            violations.append({
                "rule": "evidence.proven.missing-ledger-anchor-field",
                "match": p["id"],
                "fix": f"PROVEN row {p['id']} Ledger anchor cell is empty.",
            })
        if p["verified"] != "YES":
            violations.append({
                "rule": "evidence.proven.verified-not-yes",
                "match": p["id"],
                "fix": f"PROVEN row {p['id']} Verified must be YES.",
            })
    honest_terminal_fail = _honest_terminal_fail_allowed(text, metrics, _calculate_metrics_from_grids(text))
    for g in gap_rows:
        if g["blocks_pass"] == "YES" and not honest_terminal_fail:
            violations.append({
                "rule": "evidence.gap.fillable-blocks-pass-forces-fail",
                "match": g["id"],
                "fix": f"GAP row {g['id']} Blocks PASS=YES forces METRICS_GATE FAIL until resolved.",
            })
    for p in partial_rows:
        if p["objective_critical"] == "YES":
            violations.append({
                "rule": "evidence.partial.objective-critical-caps-confidence",
                "match": p["id"],
                "fix": f"PARTIAL row {p['id']} Objective-critical=YES caps BUILD_CONFIDENCE at 94.",
            })
    for b in behavior_rows:
        if b["blocks_pass"] == "YES" and not honest_terminal_fail:
            violations.append({
                "rule": "evidence.behavior-fail.blocks-pass-forces-fail",
                "match": b["id"],
                "fix": f"BEHAVIOR_FAIL row {b['id']} Blocks PASS=YES forces BUILD_METRICS_GATE FAIL until resolved.",
            })

    claim_text = strip_non_claim_regions(text)
    has_body_blocker = bool(BODY_BLOCKING_FINDING_RE.search(claim_text))
    has_blocking_footer_row = any(g["blocks_pass"] == "YES" for g in gap_rows) or any(
        b["blocks_pass"] == "YES" for b in behavior_rows
    )
    if has_body_blocker and not has_blocking_footer_row and metrics.get("METRICS_GATE") == "PASS":
        violations.append({
            "rule": "evidence.body-blocking-claim.requires-behavior-fail",
            "match": BODY_BLOCKING_FINDING_RE.search(claim_text).group(0)[:120],
            "fix": ("Body reports an unresolved blocking production failure; footer must preserve "
                    "a blocking BEHAVIOR_FAIL or GAP row and BUILD METRICS GATE must fail until "
                    "the blocker is resolved."),
        })

    if METRICS_GATE_HEADER_RE.search(text):
        calc = _calculate_metrics_from_grids(text)
        if not _displayed_metrics_match_calculated(text, calc):
            violations.append({
                "rule": "evidence.metrics-gate.calculated-mismatch",
                "match": f"calculated={calc['METRICS_GATE']}/{calc['GAPS_FILLED']}/{calc['DISCOVERY']}/{calc['BUILD_CONFIDENCE']}",
                "fix": ("Displayed BUILD METRICS GATE values do not match calculated values from "
                        "canonical Truth grids."),
            })

    return violations


# ================================================================
# end Pack 1J-W schema v2 helpers
# ================================================================


def evaluate(text, used_tools_current_turn, used_tools_mode, session_id="", rewrite_flag_active=False):
    violations = []
    scan_text = strip_non_claim_regions(text)
    compact_present = _compact_truth_footer_present(text)
    compact_allowed = _compact_footer_allowed(text, used_tools_current_turn, rewrite_flag_active)

    if isinstance(text, str) and E2E_FORCE_SHORT_TOKEN in text:
        violations.append({
            "rule": "debug.e2e.short-forced-fail",
            "match": E2E_FORCE_SHORT_TOKEN,
            "fix": "E2E canary seen. Remove the canary and rewrite with the SHORT box-table footer.",
        })
    if isinstance(text, str) and E2E_FORCE_FULL_TOKEN in text:
        violations.append({
            "rule": "debug.e2e.full-forced-fail",
            "match": E2E_FORCE_FULL_TOKEN,
            "fix": "E2E canary seen. Remove the canary and rewrite with the FULL box-table footer.",
        })

    if PREAMBLE_RE.search(scan_text):
        violations.append({"rule": "style.preamble",
                           "fix": "remove preamble (no 'Let me' / 'I will' / 'Sure' / 'Here is what')"})
    if RECAP_RE.search(scan_text):
        violations.append({"rule": "style.recap",
                           "fix": "remove summary-tail phrases"})
    if MOTIVE_RE.search(scan_text):
        violations.append({"rule": "style.motive",
                           "fix": "remove motive/confession language; state system-state truth only"})

    has_f = has_footer(text)
    if compact_present and not compact_allowed:
        violations.append({
            "rule": "evidence.schema.compact-footer.not-allowed",
            "fix": "Inline SMALL footer is no longer accepted. Use the canonical box-table-rendering Markdown pipe-grid footer.",
        })

    if not has_f and not compact_allowed:
        violations.append({
            "rule": "evidence.schema.canonical-footer.always-required",
            "fix": "Every final answer must end with SHORT or FULL Truth Gate box-table-rendering Markdown pipe-grid footer plus BUILD METRICS GATE.",
        })

    risky_in_scan = False
    for _, rx in DANGER_CATEGORIES:
        if rx.search(scan_text):
            risky_in_scan = True
            break

    if risky_in_scan and not has_f and not compact_allowed:
        for name, rx in DANGER_CATEGORIES:
            m = rx.search(scan_text)
            if m:
                violations.append({
                    "rule": f"language.{name}.no-footer",
                    "match": m.group(0),
                    "fix": FIX_FOOTER,
                })

    if used_tools_mode == "transcript" and used_tools_current_turn and not has_f and not compact_allowed:
        violations.append({
            "rule": "tool-use.current-turn.no-footer",
            "fix": FIX_TOOL_FOOTER,
        })

    if not compact_allowed:
        violations.extend(_canonical_schema_violations(text))

    # Phase-2-lite ledger-anchor check: only fires when session_id known
    # AND anchors exist for that session. Empty-anchor case = skip.
    if (used_tools_mode == "transcript"
            and used_tools_current_turn
            and has_f
            and session_id):
        anchors = _ledger_anchors_for_session(session_id)
        if anchors:
            proven = _extract_proven_lines(text)
            if not _has_ledger_anchor(proven, anchors):
                violations.append({
                    "rule": "evidence.proven.no-ledger-anchor",
                    "fix": "PROVEN line(s) must cite a substring from a recent ledger row (command, file_path, stdout/stderr/error tail, or 'Exit code N')",
                })

    for rx, msg in SPECIFIC_RES:
        if rx.search(scan_text):
            violations.append({
                "rule": "specific.evidence-mismatch",
                "fix": msg,
            })

    # Pack 1 truth-gate-hook-repair: GAP classification.
    # Footer must label every GAP bullet as fillable (with discover path)
    # OR unfillable: <reason>. Lazy GAP body blocks.
    if has_f:
        gap_lines = _extract_gap_lines(text)
        for g in gap_lines:
            cls, _reason = _classify_gap(g)
            if cls == "unfillable_no_reason":
                violations.append({
                    "rule": "evidence.gap.unfillable.no-reason",
                    "match": g[:120],
                    "fix": "GAP body uses 'unfillable' without ': <exact reason>'. Add reason or relabel.",
                })
            elif cls == "fillable":
                violations.append({
                    "rule": "evidence.gap.fillable.no-discover",
                    "match": g[:120],
                    "fix": "Fillable GAP requires discover-then-rewrite next turn. Stop hook writes flag.",
                })
            elif cls == "unspecified":
                violations.append({
                    "rule": "evidence.gap.unspecified",
                    "match": g[:120],
                    "fix": "GAP must be labelled fillable (carries discover path) or unfillable: <reason>.",
                })

    # Pack 1J-W schema v2 gates. Fire only when implementation/closure language
    # present OR TRUTH_SCHEMA_V2_REQUIRED=1. Preserves backward compat for
    # legacy bare PROVEN/PARTIAL/GAP answers without closure language.
    if has_f and _schema_v2_required(scan_text):
        for section in ("PARTIAL", "GAP", "CURRENT_STATE", "NEXT", "BEHAVIOR_FAIL"):
            if not _section_present(text, section):
                violations.append({
                    "rule": "evidence.schema.section." + section.lower().replace("_", "-") + "-missing",
                    "fix": "Add `- " + section + ":` bullet to TRUTH footer (or `none` / `none blocking` if vacuous).",
                })
        if _section_present(text, "BEHAVIOR_FAIL") and not _behavior_fail_is_last_truth_section(text):
            violations.append({
                "rule": "evidence.schema.section.behavior-fail-not-last",
                "fix": "BEHAVIOR_FAIL must be the LAST TRUTH section after BUILD METRICS GATE.",
            })
        # Pack 1J-AD: missing-metrics-gate handled unconditionally above.
        if _has_metrics_gate(text):
            metrics = _parse_metrics_gate(text)
            for row in ("GAPS_FILLED", "DISCOVERY", "BUILD_CONFIDENCE", "METRICS_GATE"):
                if metrics.get(row) is None:
                    violations.append({
                        "rule": "evidence.schema.metrics-gate.row-missing",
                        "match": row,
                        "fix": "BUILD METRICS GATE row " + row + " missing.",
                    })
            if METRIC_BLANK_RE.search(text):
                violations.append({
                    "rule": "evidence.schema.metrics-gate.blank-placeholder",
                    "fix": "Replace `__` placeholder with concrete number / PASS / FAIL.",
                })
            gate_decl = metrics.get("METRICS_GATE")
            if gate_decl == "PASS" and not _metrics_gate_internally_consistent(metrics):
                violations.append({
                    "rule": "evidence.schema.metrics-gate.numeric-pass-mismatch",
                    "fix": ("METRICS_GATE PASS requires GAPS_FILLED>=" + str(METRICS_REQUIRED_GAPS_FILLED)
                            + " AND DISCOVERY>=" + str(METRICS_REQUIRED_DISCOVERY)
                            + " AND BUILD_CONFIDENCE>=" + str(METRICS_REQUIRED_BUILD_CONFIDENCE) + "."),
                })
            if gate_decl == "FAIL" and CLOSURE_RE.search(scan_text):
                violations.append({
                    "rule": "evidence.schema.metrics-gate.fail-with-implementation-claim",
                    "fix": "METRICS_GATE FAIL forbids closure language. Remove implementation/done/shipped claims or fill the gate.",
                })
        for partial in _extract_section_lines(text, "PARTIAL"):
            stripped = partial.strip().lower().rstrip(".")
            if stripped in ("none", "none blocking", ""):
                continue
            if not _partial_bullet_well_formed(partial):
                violations.append({
                    "rule": "evidence.partial.missing-why-closure",
                    "match": partial[:120],
                    "fix": "PARTIAL bullet must state: what proven / what not proven / why partial / what closes it.",
                })
        for gap in _extract_section_lines(text, "GAP"):
            stripped = gap.strip().lower().rstrip(".")
            if stripped in ("none", "none blocking", ""):
                continue
            if not _gap_bullet_well_formed(gap):
                violations.append({
                    "rule": "evidence.gap.missing-fillable-or-action",
                    "match": gap[:120],
                    "fix": "GAP bullet must state: fillable/unfillable + missing proof + next read/test/action.",
                })

    # Pack 1J-AD: METRICS GATE table mandatory whenever TRUTH footer present.
    # Fires unconditionally (no env gate, no closure-language gate). Pack
    # 1J-AC was reverted because legacy fixtures lacked the table; Pack 1J-AD
    # migrated those fixtures and now re-applies the rule.
    if has_f and not _has_metrics_gate(text):
        violations.append({
            "rule": "evidence.schema.metrics-gate.missing",
            "fix": "TRUTH footer requires BUILD METRICS GATE table (GAPS_FILLED / DISCOVERY / BUILD_CONFIDENCE / METRICS_GATE rows). No omission.",
        })

    # Pack 1J-BN: canonical strict-grid evaluator is unconditional.
    if _schema_v3_required():
        pass

    # Pack 1J-AB: METRICS_GATE FAIL no-return rules. Fire UNCONDITIONALLY
    # (no schema_v2_required env gate) whenever footer present and the body
    # declares a METRICS GATE table with METRICS_GATE = FAIL. If Claude
    # voluntarily uses the schema template and admits gate failure, the
    # answer is not a valid final answer regardless of activation env state.
    if has_f and _has_metrics_gate(text):
        metrics = _parse_metrics_gate(text)
        gate_decl = metrics.get("METRICS_GATE")
        if gate_decl == "FAIL" and not _honest_terminal_fail_allowed(text, metrics, _calculate_metrics_from_grids(text)):
            violations.append({
                "rule": "evidence.metrics-gate.fail-return-disallowed",
                "fix": "METRICS_GATE FAIL is not a final-answer state. Continue work in same session or relabel remaining items as unfillable with explicit reason.",
            })
            gap_lines = _extract_section_lines(text, "GAP")
            if any(
                "fillable" in g.lower() and not g.lower().lstrip().startswith("unfillable")
                and "(unfillable" not in g.lower()
                for g in gap_lines
            ):
                violations.append({
                    "rule": "evidence.metrics-gate.fillable-gap-return-disallowed",
                    "fix": "Fillable GAP item present with METRICS_GATE FAIL. Resolve the fillable gap via read/test/action before final answer.",
                })

    return violations


# ================================================================
# Pack 1H-A: classifier + auto-rewrite packet writer (helpers only)
# ================================================================
# Phase 1H-A adds helpers + classifier + atomic packet writer + stuck-flag
# writer. NOT wired into the existing block-handler / flag-clear paths in
# Pack 1H-A; wiring lives in Phase 1H-B. UPS prompt-inject advisory edit
# is also Phase 1H-B. No pipe write, no async child, no subprocess spawn,
# no settings mutation.
#
# Stuck flag schema (Pack 1H-A writes; Pack 1I will consume):
#   packet_id, packet_path, reason, loop_limit, prior_attempts_for_packet,
#   original_rule_ids, next_injection_prompt, created_at
PACKETS_DIR = Path(os.path.expanduser("~")) / ".claude" / "truth" / "packets"
STUCK_FLAG = Path(os.path.expanduser("~")) / ".claude" / "truth" / "auto-rewrite-stuck.flag"
NEEDS_DISCOVERY_FLAG = (
    Path(os.path.expanduser("~")) / ".claude" / "truth" / "auto-rewrite-needs-discovery.flag"
)
LOOP_LIMIT = 2
MSG_BYTES_L4_THRESHOLD = 12000
MULTI_VIOLATION_L4_THRESHOLD = 3

# Level defs per REV3 3.7: (label, target_min, target_max, max_tokens, needs_discovery)
LEVEL_DEFS = {
    1: ("format-only",   80,  150, 300, False),
    2: ("evidence-only", 200, 400, 600, False),
    3: ("fillable-gap",  400, 700, 900, True),
    4: ("full-content",  0,   0,   0,   False),
}

# Pack 1J-Z: schema v2 format rules. Added to L1 (format-only auto-rewrite)
# and excluded from MULTI_VIOLATION_L4_THRESHOLD count so a single answer
# missing several schema sections does not escalate to non-injectable L4.
SCHEMA_V2_FORMAT_RULES = {
    "evidence.schema.section.partial-missing",
    "evidence.schema.section.gap-missing",
    "evidence.schema.section.current-state-missing",
    "evidence.schema.section.next-missing",
    "evidence.schema.section.behavior-fail-missing",
    "evidence.schema.section.behavior-fail-not-last",
    "evidence.schema.metrics-gate.missing",
    "evidence.schema.metrics-gate.row-missing",
    "evidence.schema.metrics-gate.blank-placeholder",
    "evidence.schema.metrics-gate.required-cell-too-wide",
    "evidence.schema.metrics-gate.numeric-pass-mismatch",
    "evidence.schema.metrics-gate.fail-with-implementation-claim",
    "evidence.partial.missing-why-closure",
    "evidence.gap.missing-fillable-or-action",
    # Pack 1J-AA: no-return-on-fail rules.
    "evidence.metrics-gate.fail-return-disallowed",
    "evidence.metrics-gate.fillable-gap-return-disallowed",
    # Pack 1J-AM: v3 strict-grid rules.
    "evidence.metrics-gate.invalid-pass-fail-value",
    "evidence.proven.missing-ledger-anchor-field",
    "evidence.proven.verified-not-yes",
    "evidence.gap.fillable-blocks-pass-forces-fail",
    "evidence.partial.objective-critical-caps-confidence",
    "evidence.metrics-gate.calculated-mismatch",
    "evidence.body-blocking-claim.requires-behavior-fail",
    # Pack 1J-AP: fake key/value grid detector.
    "evidence.schema.v3.fake-key-value-grid",
    # Pack 1J-AS: fake box-drawing grid detector.
    "evidence.schema.v3.fake-box-drawing-grid",
    # Pack 1J-CD2: require Markdown pipe tables that render as Claude CLI box tables.
    "evidence.schema.box-table-rendering.required",
    # Pack 1J-CD2: canonical footer is always required.
    "evidence.schema.canonical-footer.always-required",
}

L1_RULES = {
    "language.behavior.no-footer",
    "language.verification.no-footer",
    "language.closure.no-footer",
    "language.cleanliness.no-footer",
    "language.universal.no-footer",
    "language.confidence.no-footer",
    "evidence.gap.unspecified",
    "evidence.gap.unfillable.no-reason",
    "evidence.rewrite.required.not-first-line",
    "evidence.rewrite.required.no-rule-ids",
    "evidence.rewrite.required.rule-ids-incomplete",
    "evidence.rewrite.required.not-satisfied",
    "evidence.rewrite.required.rule-ids-mismatch",
    "debug.e2e.short-forced-fail",
    "style.preamble",
    "style.recap",
    "style.motive",
    "specific.evidence-mismatch",
} | SCHEMA_V2_FORMAT_RULES
L2_RULES = {"evidence.proven.no-ledger-anchor"}
L3_RULES = {"evidence.gap.fillable.no-discover"}


def _classify_rewrite_level(rules, discover_flag_present, msg_bytes):
    """Per REV3 3.5 + 3.7. Returns (level, label, target_tokens, max_tokens,
    needs_discovery). target_tokens is the upper bound of the level's range."""
    rule_set = set(rules or [])
    if not rule_set:
        label, _tmin, tmax, mx, nd = LEVEL_DEFS[1]
        return (1, label, tmax, mx, nd)
    if "debug.e2e.full-forced-fail" in rule_set:
        label, _a, _b, _c, nd = LEVEL_DEFS[4]
        return (4, label, 0, 0, nd)
    if msg_bytes is not None and msg_bytes > MSG_BYTES_L4_THRESHOLD:
        label, _a, _b, _c, nd = LEVEL_DEFS[4]
        return (4, label, 0, 0, nd)
    # Pack 1J-Z: schema v2 format rules don't count toward the multi-violation
    # threshold. A bundle of schema-only rules is one logical fix (re-emit
    # with the required template); not N separate semantic violations.
    countable = rule_set - SCHEMA_V2_FORMAT_RULES
    if len(countable) > MULTI_VIOLATION_L4_THRESHOLD:
        label, _a, _b, _c, nd = LEVEL_DEFS[4]
        return (4, label, 0, 0, nd)
    unknown = rule_set - L1_RULES - L2_RULES - L3_RULES
    if unknown:
        label, _a, _b, _c, nd = LEVEL_DEFS[4]
        return (4, label, 0, 0, nd)
    in_l3 = bool(rule_set & L3_RULES) and bool(discover_flag_present)
    in_l2 = bool(rule_set & L2_RULES)
    in_l1 = bool(rule_set & L1_RULES)
    if in_l3 and not (in_l2 or in_l1):
        label, _tmin, tmax, mx, nd = LEVEL_DEFS[3]
        return (3, label, tmax, mx, nd)
    if in_l2 and not in_l1 and not in_l3:
        label, _tmin, tmax, mx, nd = LEVEL_DEFS[2]
        return (2, label, tmax, mx, nd)
    if in_l1 and not in_l2 and not in_l3:
        label, _tmin, tmax, mx, nd = LEVEL_DEFS[1]
        return (1, label, tmax, mx, nd)
    # Mixed: escalate to highest applicable level among present ones.
    if in_l2:
        label, _tmin, tmax, mx, nd = LEVEL_DEFS[2]
        return (2, label, tmax, mx, nd)
    label, _tmin, tmax, mx, nd = LEVEL_DEFS[1]
    return (1, label, tmax, mx, nd)


def _compute_packet_id(first_source_log_row_ts, original_rule_ids, first_assistant_msg_bytes):
    """Frozen anchor inputs per REV3 3.3. Same inputs -> same packet_id."""
    import hashlib as _hl
    rids = ",".join(sorted([str(r) for r in (original_rule_ids or []) if str(r)]))
    payload = f"{first_source_log_row_ts or ''}|{rids}|{first_assistant_msg_bytes or 0}"
    return _hl.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _packet_path_for(packet_id):
    return PACKETS_DIR / f"{packet_id}.json"


def _write_packet_atomic(packet):
    """Atomic temp + fsync + os.replace per REV3 3.7. Returns (status,
    path_or_error_repr). Cleans up temp on failure. No partial final file."""
    import tempfile as _tf
    pid = packet.get("packet_id") or ""
    if not pid:
        return ("error", "missing-packet-id")
    try:
        PACKETS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return ("error", repr(e))
    final = _packet_path_for(pid)
    fd = None
    tmp = None
    try:
        fd, tmp = _tf.mkstemp(prefix=".tmp-", suffix=".json", dir=str(PACKETS_DIR))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None
            json.dump(packet, f, sort_keys=True, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(final))
        tmp = None
        return ("ok", str(final))
    except Exception as e:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        if tmp is not None and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return ("error", repr(e))


def _read_packet(packet_id):
    p = _packet_path_for(packet_id)
    try:
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _increment_packet_attempts(packet_id):
    """Bump prior_attempts_for_packet on existing packet via atomic rewrite.
    Returns new count or None on failure."""
    pkt = _read_packet(packet_id)
    if pkt is None:
        return None
    new_count = int(pkt.get("prior_attempts_for_packet", 0)) + 1
    pkt["prior_attempts_for_packet"] = new_count
    status, _ = _write_packet_atomic(pkt)
    if status != "ok":
        return None
    return new_count


BLOCKING_EVIDENCE_RULES = frozenset({
    "evidence.behavior-fail.blocks-pass-forces-fail",
    "evidence.gap.fillable-blocks-pass-forces-fail",
    "evidence.body-blocking-claim.requires-behavior-fail",
    "evidence.metrics-gate.calculated-mismatch",
})


def _calc_mismatch_blocking(match) -> bool:
    """calculated-mismatch is a blocker UNLESS original calculated was fully
    green. Ambiguous -> blocking (honest-safe). Mirrors injector logic."""
    m = str(match or "").lower()
    if "calculated=pass/100/100/100" in m:
        return False
    return True


def _tsg_cell(value, limit: int = 120) -> str:
    """Sanitize a value before placing it in a generated Markdown table
    cell. Display-only; never used in rule-matching logic."""
    t = str(value or "").replace("|", "/").replace("\n", " ").replace("\r", " ").strip()
    t = " ".join(t.split())
    return (t[:limit] if t else "n/a")


def _packet_has_blocking_evidence(packet) -> bool:
    """True if the packet still carries a blocking-evidence violation that the
    next-injection prompt must NOT launder into a green footer."""
    p = packet or {}
    vmap = {}
    for v in p.get("violations", []) or []:
        if isinstance(v, dict):
            rid = str(v.get("rule") or "").strip()
            if rid:
                vmap.setdefault(rid, v)
    rids = set(vmap)
    for r in (p.get("original_rule_ids") or p.get("rule_ids") or []):
        rids.add(str(r).strip())
    for rid in rids:
        if not rid:
            continue
        if rid == "evidence.metrics-gate.calculated-mismatch":
            if _calc_mismatch_blocking((vmap.get(rid) or {}).get("match")):
                return True
            continue
        if rid in BLOCKING_EVIDENCE_RULES:
            return True
    return False


def _build_next_injection_prompt(packet):
    """Pack 1I consumer contract. Pack 1H-A writes this into stuck flag.
    Pack 1H-A does NOT inject anywhere; 1I will consume + inject."""
    pid = packet.get("packet_id", "")
    pp = packet.get("packet_path", "")
    rids = packet.get("original_rule_ids", []) or packet.get("rule_ids", []) or []
    rids_csv = ", ".join(str(r) for r in rids)
    anchor = pid or pp or "auto-rewrite-packet"
    if _packet_has_blocking_evidence(packet):
        rcsv = _tsg_cell(rids_csv or "unknown", 96)
        anchor = _tsg_cell(anchor, 80)
        return (
            "LOOP DETECTED -- same correction packet failed. The original "
            "answer carried blocking evidence. Preserve the blockers; do NOT "
            "launder them green. Return exactly this honest-red footer.\n"
            f"packet_id: {pid}\n"
            f"packet_path: {pp}\n"
            f"original_rule_ids: {rids_csv}\n\n"
            "TRUTH:\n"
            "| ID | Claim | Proof | Verified |\n"
            "|---|---|---|---|\n"
            f"| T1 | original blocking violation(s) {rcsv} recorded | packet_id: {anchor} | YES |\n\n"
            "GAP:\n"
            "| ID | Gap | Fillable | Missing proof | Next read-test-action | Blocks PASS |\n"
            "|---|---|---|---|---|---|\n"
            f"| G1 | unfillable: original blocking violation(s) {rcsv} still present after rewrite | NO | re-derive honest-red footer; keep blocker rows | preserve blocker rows | YES |\n\n"
            "STATE_NEXT:\n"
            "| State | Next action | Owner | Proof |\n"
            "|---|---|---|---|\n"
            f"| blocker still open | preserve blocker rows; do not launder green | assistant | packet_id: {anchor} |\n\n"
            "BUILD METRICS GATE:\n"
            "| Metric | Required | Actual | Pass/Fail |\n"
            "|---|---:|---:|---|\n"
            "| GAPS_FILLED | 100% | 0% | FAIL |\n"
            "| DISCOVERY | 100% | 100% | PASS |\n"
            "| BUILD_CONFIDENCE | >=95% | 94% | FAIL |\n"
            "| METRICS_GATE | PASS only if all above pass | FAIL | FAIL |\n\n"
            "BEHAVIOR_FAIL:\n"
            "| ID | Failure | Proof | Blocks PASS |\n"
            "|---|---|---|---|\n"
            f"| BF1 | original blocking violation(s) {rcsv} still present; auto-rewrite did not clear them | packet_id: {anchor} | YES |"
        )
    return (
        "LOOP DETECTED -- same correction packet failed. Read packet. Fix only "
        "failed rules. Return exactly the canonical Truth Gate footer below "
        "using literal Markdown pipe tables only.\n"
        f"packet_id: {pid}\n"
        f"packet_path: {pp}\n"
        f"original_rule_ids: {rids_csv}\n\n"
        "TRUTH:\n"
        "| ID | Claim | Proof | Verified |\n"
        "|---|---|---|---|\n"
        f"| T1 | Loop correction is anchored to the packet | {anchor} | YES |\n\n"
        "GAP:\n"
        "| ID | Gap | Fillable | Missing proof | Next read-test-action | Blocks PASS |\n"
        "|---|---|---|---|---|---|\n"
        "| G1 | No blocking footer-format gap remains | NO | none | none | NO |\n\n"
        "STATE_NEXT:\n"
        "| State | Next action | Owner | Proof |\n"
        "|---|---|---|---|\n"
        f"| loop correction requested | Continue only if more proof is required | assistant | {anchor} |\n\n"
        "BUILD METRICS GATE:\n"
        "| Metric | Required | Actual | Pass/Fail |\n"
        "|---|---:|---:|---|\n"
        "| GAPS_FILLED | 100% | 100% | PASS |\n"
        "| DISCOVERY | 100% | 100% | PASS |\n"
        "| BUILD_CONFIDENCE | >=95% | 100% | PASS |\n"
        "| METRICS_GATE | PASS only if all above pass | PASS | PASS |\n\n"
        "BEHAVIOR_FAIL:\n"
        "| ID | Failure | Proof | Blocks PASS |\n"
        "|---|---|---|---|\n"
        "| BF1 | none | no blocking behavior failure in this correction | NO |"
    )


def _write_stuck_flag(packet, reason):
    """Stuck flag carries Pack 1I-consumable next_injection_prompt + REV2/REV3
    fields enabling Pack 1I loop-prompt injection + quarantine state."""
    try:
        STUCK_FLAG.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if STUCK_FLAG.exists():
            try:
                existing = json.loads(STUCK_FLAG.read_text(encoding="utf-8")) or {}
            except Exception:
                existing = {}
        # Preserve REV2/REV3 counter fields across re-writes when same packet_id.
        same_packet = (existing.get("packet_id") == packet.get("packet_id", ""))
        injected_loop_attempts = int(existing.get("injected_loop_attempts", 0)) if same_packet else 0
        max_injected_loop_attempts = int(existing.get("max_injected_loop_attempts", 1)) if same_packet else 1
        last_injected_packet_id = existing.get("last_injected_packet_id", "") if same_packet else ""
        last_injected_at = existing.get("last_injected_at", "") if same_packet else ""
        quarantine_reason = existing.get("quarantine_reason", "") if same_packet else ""
        # Pack 1I session_id field carried from packet (or empty if absent).
        session_id = packet.get("session_id", "") if isinstance(packet, dict) else ""
        payload = {
            "version": 2,
            "packet_id": packet.get("packet_id", ""),
            "packet_path": packet.get("packet_path", ""),
            "session_id": session_id,
            "reason": reason,
            "loop_limit": LOOP_LIMIT,
            "prior_attempts_for_packet": int(packet.get("prior_attempts_for_packet", 0)),
            "original_rule_ids": list(packet.get("original_rule_ids", []) or packet.get("rule_ids", []) or []),
            "next_injection_prompt": _build_next_injection_prompt(packet),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # Pack 1I REV2/REV3 fields.
            "injected_loop_attempts": injected_loop_attempts,
            "max_injected_loop_attempts": max_injected_loop_attempts,
            "last_injected_packet_id": last_injected_packet_id,
            "last_injected_at": last_injected_at,
            "quarantine_reason": quarantine_reason,
        }
        tmp = STUCK_FLAG.with_suffix(".flag.tmp")
        tmp.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(STUCK_FLAG))
        return True
    except Exception:
        return False


# ----------------------------------------------------------------
# Pack 1I async injector spawn (REV3).
# ----------------------------------------------------------------
DISABLED_FLAG = Path(os.path.expanduser("~")) / ".claude" / "truth" / "auto-rewrite-disabled.flag"
INFLIGHT_FLAG = Path(os.path.expanduser("~")) / ".claude" / "truth" / "auto-rewrite-inflight.flag"
INFLIGHT_TTL_SEC = 60
PTY_PID_FILE = Path(os.path.expanduser("~")) / ".claude" / "claude-pty.pid"
INJECTOR_SCRIPT = Path(os.path.expanduser("~")) / ".claude" / "hooks" / "auto-rewrite-injector.py"

# Pack 1J-B: wrapper-session map. Stop hook updates the entry whose
# wrapper_pid+pipe_path+cwd match the env triple injected by claude-pty.js
# at child spawn. Stop hook NEVER creates synthetic entries (only
# claude-pty.js writes new entries) and NEVER opens a pipe.
SESSION_MAP_PATH = Path(os.path.expanduser("~")) / ".claude" / "claude-pty-session-map.json"


def _session_map_read():
    try:
        if not SESSION_MAP_PATH.exists():
            return None
        return json.loads(SESSION_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _session_map_atomic_write(obj):
    import tempfile as _tf
    tmp = None
    try:
        SESSION_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = _tf.mkstemp(prefix=".tmp-", suffix=".json",
                               dir=str(SESSION_MAP_PATH.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(SESSION_MAP_PATH))
        return True
    except Exception:
        try:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        return False


def _session_map_update_session(wrapper_pid_str, pipe_path, cwd_marker,
                                 session_id, transcript_path):
    """Find entry matching wrapper_pid+pipe_path+cwd; set session_id /
    transcript_path / last_seen_at / status="bound" via atomic rewrite.
    Returns (updated:bool, desync:bool). desync=True means env says
    wrapped but no matching map entry exists -- Stop hook does NOT
    create a synthetic entry (only claude-pty.js writes new entries).
    """
    obj = _session_map_read()
    if not obj or "wrappers" not in obj:
        return (False, True)
    wrappers = obj.get("wrappers", []) or []
    found_idx = -1
    for i, e in enumerate(wrappers):
        if (str((e or {}).get("wrapper_pid", "")) == str(wrapper_pid_str)
                and (e or {}).get("pipe_path", "") == pipe_path
                and (e or {}).get("cwd", "") == cwd_marker):
            found_idx = i
            break
    if found_idx < 0:
        return (False, True)
    now_iso_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
    wrappers[found_idx]["session_id"] = session_id
    wrappers[found_idx]["transcript_path"] = transcript_path
    wrappers[found_idx]["last_seen_at"] = now_iso_str
    wrappers[found_idx]["status"] = "bound"
    obj["wrappers"] = wrappers
    obj["updated_at"] = now_iso_str
    return (_session_map_atomic_write(obj), False)


def _injector_safe_to_spawn(packet, stuck_flag_data):
    """REV3 spawn gate. Returns (ok: bool, reason: str)."""
    if not isinstance(packet, dict) or not isinstance(stuck_flag_data, dict):
        return (False, "invalid-input")
    nip = stuck_flag_data.get("next_injection_prompt", "") or ""
    if not nip:
        return (False, "no-next-injection-prompt")
    level = int(packet.get("rewrite_level", 0))
    if level not in (1, 2, 3):
        return (False, f"level-not-injectable:{level}")
    if level == 3 and not bool(packet.get("discovery_preauthorized", False)):
        return (False, "level-3-no-preauth")
    inj_loop = int(stuck_flag_data.get("injected_loop_attempts", 0))
    max_loop = int(stuck_flag_data.get("max_injected_loop_attempts", 1))
    if inj_loop >= max_loop:
        return (False, "loop-limit-reached")
    if (stuck_flag_data.get("quarantine_reason") or "").strip():
        return (False, "already-quarantined")
    if DISABLED_FLAG.exists():
        return (False, "kill-switch-active")
    if INFLIGHT_FLAG.exists():
        try:
            existing = json.loads(INFLIGHT_FLAG.read_text(encoding="utf-8")) or {}
            ts_str = (existing.get("spawned_at") or "").replace("Z", "+00:00")
            if ts_str:
                ts = datetime.fromisoformat(ts_str)
                age = (datetime.now(timezone.utc) - ts).total_seconds()
                if age <= INFLIGHT_TTL_SEC:
                    return (False, "inflight-lock-active")
        except Exception:
            pass
    if not PTY_PID_FILE.exists():
        return (False, "pty-pid-missing")
    if not INJECTOR_SCRIPT.exists():
        return (False, "injector-script-missing")
    return (True, "ok")


def _spawn_injector_async(packet_id, dry_run=False, source="stuck"):
    """Spawn auto-rewrite-injector.py async. Stop hook never waits.
    NO synchronous pipe write. NO blocking. Returns True on Popen success.

    Pack 1J-I: source dispatch. source="stuck" (default) preserves
    Pack 1H-B / 1I behavior. source="rewrite_required" tells the
    injector to read rewrite-required.flag directly so every Truth
    Gate block triggers an immediate auto-rewrite for the same session.
    """
    try:
        import subprocess as _sp
        cmd = [sys.executable, str(INJECTOR_SCRIPT)]
        if dry_run:
            cmd.append("--dry-run")
        if source and source != "stuck":
            cmd.extend(["--source", source])
        # Detach: stdin=DEVNULL, stdout/stderr=DEVNULL, close_fds, start_new_session.
        kwargs = dict(
            stdin=_sp.DEVNULL,
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
            close_fds=True,
        )
        if os.name == "nt":
            kwargs["creationflags"] = (
                getattr(_sp, "DETACHED_PROCESS", 0)
                | getattr(_sp, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            kwargs["start_new_session"] = True
        _sp.Popen(cmd, **kwargs)
        return True
    except Exception:
        return False
# ----------------------------------------------------------------
# end Pack 1I spawn helpers
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# Pack 1H-B anti-fake clear gates 7-9 helpers (REV3 2.2).
# ----------------------------------------------------------------
_CHANGED_LITERAL_RE = re.compile(r"(?m)^\s*CHANGED\s*:")
_UNCHANGED_LITERAL_RE = re.compile(r"(?m)^\s*UNCHANGED\s*:")


def _assistant_has_changed_unchanged(text):
    """Pack 1H-B gate 7: literal `CHANGED:` AND literal `UNCHANGED:` substrings
    must both be present (line-anchored, ASCII colon, anywhere in body)."""
    if not isinstance(text, str) or not text:
        return False
    return bool(_CHANGED_LITERAL_RE.search(text)) and bool(_UNCHANGED_LITERAL_RE.search(text))


def _packet_anchor_match(text, packet, flag):
    """Pack 1H-B gate 8: PROVEN bullet body must contain at least one of:
    packet_id literal (16 hex), packet_path suffix (basename or absolute
    fragment), or first_source_log_row_ts (ISO timestamp).

    Returns (ok: bool, match_kind: str). match_kind is one of:
    'packet_id', 'packet_path', 'first_source_log_row_ts', 'none'.
    """
    if not isinstance(text, str) or not text:
        return (False, "none")
    proven_lines = _extract_proven_lines(text)
    if not proven_lines:
        return (False, "none")
    pid = (packet or {}).get("packet_id", "") if packet else ""
    ppath = (packet or {}).get("packet_path", "") if packet else ""
    fts = (flag or {}).get("first_source_log_row_ts", "") if flag else ""
    pbasename = os.path.basename(ppath) if ppath else ""
    for line in proven_lines:
        # Check most-specific anchors first so a citation of the full path
        # (which contains pid as a substring inside basename) reports as
        # packet_path rather than getting hijacked by the pid substring check.
        if ppath and ppath.replace("\\", "/") in line.replace("\\", "/"):
            return (True, "packet_path")
        if pbasename and pbasename in line:
            return (True, "packet_path")
        if fts and fts in line:
            return (True, "first_source_log_row_ts")
        if pid and pid in line:
            return (True, "packet_id")
    return (False, "none")


# ================================================================
# Pack 1H-A: end (helpers added; Pack 1H-B wires them into main below)
# ================================================================


def main():
    # Pack 1J-S: protected-session enforcement. If marker file is present
    # and same-session actuator env triple is absent, refuse hard before
    # any Truth Gate evaluation. Prevents silent unbound protected sessions
    # where Truth Gate blocks but auto-rewrite cannot fire.
    _pmarker = Path(os.path.expanduser("~")) / ".claude" / "truth" / "protected-session.required"
    if _pmarker.exists():
        _wpid = os.environ.get("CLAUDE_TG_WRAPPER_PID", "") or ""
        _wpipe = os.environ.get("CLAUDE_TG_PIPE_PATH", "") or ""
        _wcwd = os.environ.get("CLAUDE_TG_CWD", "") or ""
        if not (_wpid and _wpipe and _wcwd):
            sys.stderr.write(
                "PROTECTED SESSION NOT BOUND -- normal `claude` is not routed "
                "through same-session actuator.\n"
                "Fix the claude shim / PATH, then reopen terminal and run "
                "`claude`.\n"
                "Auto-Rewrite cannot work in this unbound session.\n"
                "To intentionally run unprotected, remove "
                "~/.claude/truth/protected-session.required.\n"
            )
            sys.exit(2)
    raw = ""
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
    except Exception:
        raw = ""
    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {}

    if bool(data.get("stop_hook_active", False)):
        print("{}")
        return 0

    last = data.get("last_assistant_message", "")
    if not isinstance(last, str) or not last.strip():
        print("{}")
        return 0

    transcript_path = data.get("transcript_path", "")
    session_id = data.get("session_id", "")
    used_tools_current_turn, used_tools_mode = current_turn_used_tools(transcript_path)
    diag = current_turn_diagnostics(transcript_path)

    # Pack 1J-B: wrapper-session binding map update. If env triple is set
    # by claude-pty.js (CLAUDE_TG_WRAPPER_PID + CLAUDE_TG_PIPE_PATH +
    # CLAUDE_TG_CWD), find the matching map entry and stamp it with
    # session_id/transcript_path/last_seen_at/status="bound". If env
    # absent: skip map mutation (current Claude is unwrapped). If env
    # present but no matching entry: log wrapper_desync=True without
    # creating a synthetic entry. NEVER open a pipe here.
    wrapper_pid_env = (os.environ.get("CLAUDE_TG_WRAPPER_PID", "") or "")
    wrapper_pipe_path_env = (os.environ.get("CLAUDE_TG_PIPE_PATH", "") or "")
    wrapper_cwd_env = (os.environ.get("CLAUDE_TG_CWD", "") or "")
    wrapper_bound = bool(wrapper_pid_env and wrapper_pipe_path_env and wrapper_cwd_env)
    wrapper_desync = False
    wrapper_map_updated = False
    if wrapper_bound:
        try:
            updated, desync = _session_map_update_session(
                wrapper_pid_env, wrapper_pipe_path_env, wrapper_cwd_env,
                session_id, transcript_path,
            )
            wrapper_map_updated = updated
            if desync:
                wrapper_desync = True
                wrapper_bound = False
        except Exception:
            wrapper_desync = True
            wrapper_bound = False

    # Pack 1 truth-gate-hook-repair: discover flag pre-check (clear if conditions met).
    flag = _flag_read()
    flag_cleared = False
    flag_clear_reason = ""
    if flag is not None:
        anchors_with_ts = _ledger_anchors_with_ts(session_id)
        cleared, reason = _flag_check_clear(flag, session_id, last, anchors_with_ts)
        if cleared:
            _flag_clear()
            flag_cleared = True
            flag_clear_reason = reason
            flag = None

    # Pack 1B + Pack 1J-AI: rewrite flag pre-read with per-session path.
    # _rewrite_flag_read(session_id) returns this session's flag only;
    # cross-session flags are physically separate files and cannot collide.
    rewrite_flag = _rewrite_flag_read(session_id)
    rewrite_flag_cleared = False
    rewrite_flag_clear_reason = ""
    rewrite_flag_active = False
    # Pack 1J-A orphan flag telemetry. Pack 1J-AI demoted: per-session reads
    # cannot return cross-session flags so this branch is dead code; kept for
    # log_row schema stability.
    orphan_rewrite_flag_archived = False
    orphan_rewrite_flag_session_id = ""
    orphan_rewrite_flag_archive_path = ""
    if rewrite_flag is not None and rewrite_flag.get("session_id") != session_id:
        # Defensive: if a session_id mismatch ever occurs (file corruption?),
        # treat as not-our-flag and leave the file alone. Pack 1J-AI no longer
        # archives cross-session content because there is no shared file.
        orphan_rewrite_flag_session_id = rewrite_flag.get("session_id", "") or ""
        rewrite_flag = None
    if rewrite_flag is not None:
        age = _rewrite_flag_age_seconds(rewrite_flag)
        ttl = rewrite_flag.get("ttl_seconds", REWRITE_FLAG_TTL_SEC)
        if age is not None and age > ttl:
            _rewrite_flag_clear(session_id)
            rewrite_flag_cleared = True
            rewrite_flag_clear_reason = "ttl-expired"
            rewrite_flag = None
        else:
            rewrite_flag_active = True

    # Rewrite lane no longer requires assistant output to begin with the
    # mechanical envelope line. The auto-rewrite actuator only needs Claude to
    # produce a valid corrected answer/footer. Keep original_rule_ids frozen for
    # packet generation, but do not add evidence.rewrite.required.* violations.
    rewrite_violations = []
    rewrite_correction_header_first_line = False
    rewrite_correction_rule_ids_cover_original = False
    original_ids = []
    active_ids = []
    if rewrite_flag_active:
        original_ids = list(rewrite_flag.get("original_rule_ids") or [])
        active_ids = original_ids

    violations_core = evaluate(
        last,
        used_tools_current_turn,
        used_tools_mode,
        session_id,
        rewrite_flag_active,
    )
    if rewrite_flag_active:
        violations_core = _rewrite_lane_filter_violations(violations_core, active_ids)
    violations = list(rewrite_violations) + list(violations_core)
    if rewrite_flag_active and rewrite_correction_rule_ids_cover_original:
        original_set = {str(r).strip() for r in (active_ids or []) if str(r).strip()}
        if "debug.e2e.short-forced-fail" in original_set and E2E_FORCE_FULL_TOKEN in last:
            violations.append({
                "rule": "evidence.rewrite.required.new-query-mixed",
                "match": E2E_FORCE_FULL_TOKEN,
                "fix": "pending SHORT rewrite must repair only original packet; do not answer newer FULL canary/user request",
            })
        if "debug.e2e.full-forced-fail" in original_set and E2E_FORCE_SHORT_TOKEN in last:
            violations.append({
                "rule": "evidence.rewrite.required.new-query-mixed",
                "match": E2E_FORCE_SHORT_TOKEN,
                "fix": "pending FULL rewrite must repair only original packet; do not answer newer SHORT canary/user request",
            })
        if not _assistant_has_changed_unchanged(last):
            violations.append({
                "rule": "evidence.rewrite.required.changed-unchanged-missing",
                "match": "missing CHANGED/UNCHANGED",
                "fix": "rewrite correction must include literal CHANGED: and UNCHANGED: while repairing only the original packet",
            })

    # Pack 1: write discover-required flag on first fillable-gap detection
    # for this session. If a flag already exists for the same session and
    # was not cleared above, do not overwrite (keeps original gap_text).
    flag_written = False
    fillable_violations = [v for v in violations if v.get("rule") == "evidence.gap.fillable.no-discover"]
    if fillable_violations and flag is None:
        first = fillable_violations[0]
        gap_match = first.get("match", "") or ""
        payload_flag = {
            "version": 1,
            "session_id": session_id,
            "rule_id": first.get("rule"),
            "gap_text": gap_match,
            "required_action": _build_required_action(gap_match),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ttl_seconds": DISCOVER_FLAG_TTL_SEC,
        }
        if _flag_write(payload_flag):
            flag_written = True

    has_block_rule = any(
        v.get("rule") != "evidence.gap.fillable.no-discover" for v in violations
    )
    has_fillable = bool(fillable_violations)
    if not violations:
        action = "pass"
    elif has_fillable and not has_block_rule:
        action = "discover"
    else:
        action = "block"

    # Pack 1B + 1H-B: rewrite flag write/update/clear + packet wiring.
    rewrite_flag_written = False
    rewrite_flag_first_source_log_row_ts = ""
    rewrite_flag_first_assistant_msg_bytes = 0
    packet_id = ""
    packet_path = ""
    packet_write_status = ""
    packet_write_error = ""
    rewrite_level = 0
    rewrite_label = ""
    prior_attempts_for_packet_now = 0
    loop_limit_reached = False
    stuck_flag_written = False
    needs_discovery_flag_written = False
    anti_fake_changed_unchanged_pass = False
    anti_fake_anchor_pass = False
    anti_fake_anchor_match_kind = "none"
    assistant_msg_excerpt = (last or "")[:240]
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _is_rewrite_required_rule(rule_id):
        return isinstance(rule_id, str) and rule_id.startswith("evidence.rewrite.required.")

    discover_flag_present_now = bool(_flag_read())

    if violations:
        original_ids_to_freeze = [
            v.get("rule") for v in violations
            if not _is_rewrite_required_rule(v.get("rule"))
        ]
        current_ids = [v.get("rule") for v in violations]
        had_rewrite_required = any(_is_rewrite_required_rule(v.get("rule")) for v in violations)

        block_reason_text = (
            "TRUTH GATE BLOCK -- final answer violates rules. "
            "Self-rewrite per fixes below. Do not ask Al."
        )

        if rewrite_flag is None:
            new_flag = {
                "version": 1,
                "session_id": session_id,
                "last_block_ts": now_iso,
                "original_rule_ids": original_ids_to_freeze,
                "current_rule_ids": current_ids,
                "rewrite_fail_count": 0,
                "auto_rewrite_spawned_count": 0,
                "auto_rewrite_last_spawned_at": "",
                "auto_rewrite_last_packet_id": "",
                "reason": block_reason_text,
                "msg_bytes": len(last),
                "action": action,
                "created_at": now_iso,
                "ttl_seconds": REWRITE_FLAG_TTL_SEC,
                "source_log_row": now_iso,
                # Pack 1H-B: frozen anchor inputs (never overwritten while flag active).
                "first_source_log_row_ts": now_iso,
                "first_assistant_msg_bytes": len(last),
            }
            if _rewrite_flag_write(new_flag):
                rewrite_flag_written = True
                rewrite_flag = new_flag
        else:
            updated = dict(rewrite_flag)
            updated["last_block_ts"] = now_iso
            updated["current_rule_ids"] = current_ids
            updated["msg_bytes"] = len(last)
            updated["action"] = action
            updated["source_log_row"] = now_iso
            # Pack 1H-B: backfill frozen fields if missing on legacy flag.
            if "first_source_log_row_ts" not in updated:
                updated["first_source_log_row_ts"] = updated.get("created_at", now_iso)
            if "first_assistant_msg_bytes" not in updated:
                updated["first_assistant_msg_bytes"] = updated.get("msg_bytes", len(last))
            if had_rewrite_required:
                updated["rewrite_fail_count"] = int(updated.get("rewrite_fail_count", 0)) + 1
            if _rewrite_flag_write(updated):
                rewrite_flag_written = True
                rewrite_flag = updated

        # Pack 1H-B: classify, compute packet_id, write packet atomically,
        # bump attempts on repeats, write stuck/needs-discovery flags as appropriate.
        if rewrite_flag is not None:
            rewrite_flag_first_source_log_row_ts = rewrite_flag.get("first_source_log_row_ts", "")
            rewrite_flag_first_assistant_msg_bytes = int(
                rewrite_flag.get("first_assistant_msg_bytes", 0)
            )
            frozen_rids = list(rewrite_flag.get("original_rule_ids") or [])
            packet_id = _compute_packet_id(
                rewrite_flag_first_source_log_row_ts,
                frozen_rids,
                rewrite_flag_first_assistant_msg_bytes,
            )
            packet_path = str(_packet_path_for(packet_id))
            level, label, target_t, max_t, needs_disc = _classify_rewrite_level(
                current_ids, discover_flag_present_now, len(last),
            )
            rewrite_level = level
            rewrite_label = label
            existing_pkt = _read_packet(packet_id)
            if existing_pkt is None:
                pkt = {
                    "version": 1,
                    "packet_id": packet_id,
                    "session_id": session_id,
                    "source_log_row_ts": now_iso,
                    "rewrite_level": level,
                    "label": label,
                    "max_tokens": max_t,
                    "target_tokens": target_t,
                    "needs_discovery": needs_disc,
                    "discovery_preauthorized": False,
                    "rule_ids": current_ids,
                    "original_rule_ids": frozen_rids,
                    "violations": [
                        {"rule": v.get("rule", ""),
                         "match": (v.get("match", "") or "")[:240],
                         "fix": v.get("fix", "")}
                        for v in violations
                    ],
                    "transcript_path": transcript_path or "",
                    "assistant_msg_ts": now_iso,
                    "assistant_msg_bytes": len(last),
                    "anchor_targets": [],
                    "prior_attempts_for_packet": 0,
                    "loop_limit": LOOP_LIMIT,
                    "created_at": now_iso,
                    "packet_path": packet_path,
                    "clear_anchor_options": {
                        "first_source_log_row_ts": rewrite_flag_first_source_log_row_ts,
                        "packet_id": packet_id,
                        "packet_path": packet_path,
                    },
                }
                status, path_or_err = _write_packet_atomic(pkt)
                packet_write_status = status
                if status == "ok":
                    packet_path = path_or_err
                    prior_attempts_for_packet_now = 0
                else:
                    packet_write_error = path_or_err
                    prior_attempts_for_packet_now = 0
            else:
                new_count = _increment_packet_attempts(packet_id)
                if new_count is None:
                    packet_write_status = "error"
                    packet_write_error = "increment-failed"
                    prior_attempts_for_packet_now = int(existing_pkt.get("prior_attempts_for_packet", 0))
                else:
                    packet_write_status = "ok"
                    prior_attempts_for_packet_now = new_count

            if (packet_write_status == "ok"
                    and prior_attempts_for_packet_now >= LOOP_LIMIT):
                pkt_now = _read_packet(packet_id) or {}
                pkt_now.setdefault("packet_path", packet_path)
                pkt_now.setdefault("original_rule_ids", frozen_rids)
                if _write_stuck_flag(pkt_now, f"loop-limit-reached:{packet_id}"):
                    stuck_flag_written = True
                    loop_limit_reached = True

            if level == 4 and not stuck_flag_written and packet_write_status == "ok":
                pkt_now = _read_packet(packet_id) or {}
                pkt_now.setdefault("packet_path", packet_path)
                pkt_now.setdefault("original_rule_ids", frozen_rids)
                if _write_stuck_flag(pkt_now, "level-4-full-content-not-auto"):
                    stuck_flag_written = True

            if level == 3 and packet_write_status == "ok":
                try:
                    NEEDS_DISCOVERY_FLAG.parent.mkdir(parents=True, exist_ok=True)
                    NEEDS_DISCOVERY_FLAG.write_text(
                        json.dumps({
                            "version": 1,
                            "packet_id": packet_id,
                            "packet_path": packet_path,
                            "session_id": session_id,
                            "created_at": now_iso,
                        }, indent=2),
                        encoding="utf-8",
                    )
                    needs_discovery_flag_written = True
                except Exception:
                    pass

            # Pack 1I REV3: stuck detection may still write state, but Pack
            # 1J-CD2 disables source="stuck" as an automatic injector lane.
            # Same-session automatic rewrite is handled only by the
            # source="rewrite_required" block below.
            if stuck_flag_written and packet_write_status == "ok":
                stuck_flag_data = None
                try:
                    if STUCK_FLAG.exists():
                        stuck_flag_data = json.loads(STUCK_FLAG.read_text(encoding="utf-8"))
                except Exception:
                    stuck_flag_data = None
                pkt_for_spawn = _read_packet(packet_id) or {}
                spawn_ok, spawn_reason = _injector_safe_to_spawn(pkt_for_spawn, stuck_flag_data or {})
                if spawn_ok:
                    pass

            # Pack 1J-CD2: immediate same-session auto-rewrite may spawn once
            # per rewrite flag only. The next-prompt prepend path is disabled,
            # so this is the sole automatic rewrite mechanism.
            if (rewrite_flag_written
                    and packet_write_status == "ok"
                    and rewrite_flag is not None
                    and rewrite_flag.get("session_id") == session_id
                    and wrapper_bound):
                spawned_count = int(rewrite_flag.get("auto_rewrite_spawned_count", 0) or 0)
                rewrite_fail_count = int(rewrite_flag.get("rewrite_fail_count", 0) or 0)
                retry_extra = rewrite_fail_count if (had_rewrite_required and rewrite_correction_header_first_line) else 0
                max_spawns_for_flag = min(LOOP_LIMIT, 1 + retry_extra)
                if spawned_count < max_spawns_for_flag:
                    marked = dict(rewrite_flag)
                    marked["auto_rewrite_spawned_count"] = spawned_count + 1
                    marked["auto_rewrite_last_spawned_at"] = now_iso
                    marked["auto_rewrite_last_packet_id"] = packet_id
                    if _rewrite_flag_write(marked):
                        rewrite_flag = marked
                        _spawn_injector_async(packet_id, source="rewrite_required")
    else:
        # Pack 1B + 1H-B clear logic. Packets-aware path enforces anti-fake
        # gates 7+8+9 when a packet exists for the flag; legacy flags without
        # packets fall through to Pack 1B 6-gate clear for backward compat.
        if rewrite_flag is not None and rewrite_flag.get("session_id") == session_id:
            # Headerless rewrite lane: reaching this branch means evaluate()
            # found no blocking violations. Treat the rewrite as satisfied and
            # clear the pending flag; do not require REWRITE CORRECTION,
            # rule_ids, CHANGED/UNCHANGED, or packet anti-fake anchors.
            if _rewrite_flag_clear(session_id):
                rewrite_flag_cleared = True
                rewrite_flag_clear_reason = "rewrite-correction-satisfied-headerless"
                rewrite_flag = None

    template_metrics = _template_metrics_telemetry(last)
    log_row = {
        "ts": now_iso,
        "session_id": session_id,
        "msg_bytes": len(last),
        "used_tools_current_turn": used_tools_current_turn,
        "used_tools_mode": used_tools_mode,
        "has_footer": has_footer(last),
        "template_mode": template_metrics.get("template_mode"),
        "template_mode_reason": template_metrics.get("template_mode_reason"),
        "metrics_calculated": template_metrics.get("metrics_calculated"),
        "metrics_displayed": template_metrics.get("metrics_displayed"),
        "metrics_match": template_metrics.get("metrics_match"),
        "has_risky_language": has_risky_language(last),
        "violation_count": len(violations),
        "rules": [v.get("rule") for v in violations],
        "action": action,
        "flag_written": flag_written,
        "flag_cleared": flag_cleared,
        "flag_clear_reason": flag_clear_reason,
        "rewrite_flag_written": rewrite_flag_written,
        "rewrite_flag_cleared": rewrite_flag_cleared,
        "rewrite_flag_clear_reason": rewrite_flag_clear_reason,
        "rewrite_flag_original_rule_ids": (
            rewrite_flag.get("original_rule_ids", []) if rewrite_flag else []
        ),
        "rewrite_correction_header_first_line": rewrite_correction_header_first_line,
        "rewrite_correction_rule_ids_cover_original": rewrite_correction_rule_ids_cover_original,
        "rewrite_fail_count": (
            int(rewrite_flag.get("rewrite_fail_count", 0)) if rewrite_flag else 0
        ),
        # Pack 1H-B additions (REV3).
        "rewrite_level": rewrite_level,
        "rewrite_label": rewrite_label,
        "packet_id": packet_id,
        "packet_path": packet_path,
        "prior_attempts_for_packet": prior_attempts_for_packet_now,
        "loop_limit_reached": loop_limit_reached,
        "stuck_flag_written": stuck_flag_written,
        "needs_discovery_flag_written": needs_discovery_flag_written,
        "discover_flag_present": discover_flag_present_now,
        "anti_fake_changed_unchanged_pass": anti_fake_changed_unchanged_pass,
        "anti_fake_anchor_pass": anti_fake_anchor_pass,
        "anti_fake_anchor_match_kind": anti_fake_anchor_match_kind,
        "assistant_msg_excerpt": assistant_msg_excerpt,
        "packet_write_status": packet_write_status,
        "packet_write_error": packet_write_error,
        # Pack 1J-A: orphan flag quarantine telemetry.
        "orphan_rewrite_flag_archived": orphan_rewrite_flag_archived,
        "orphan_rewrite_flag_session_id": orphan_rewrite_flag_session_id,
        "orphan_rewrite_flag_archive_path": orphan_rewrite_flag_archive_path,
        # Pack 1J-B: wrapper-session binding telemetry.
        "wrapper_bound": wrapper_bound,
        "wrapper_pid": wrapper_pid_env,
        "wrapper_pipe_path": wrapper_pipe_path_env,
        "wrapper_cwd": wrapper_cwd_env,
        "wrapper_desync": wrapper_desync,
        "wrapper_map_updated": wrapper_map_updated,
    }
    log_row.update(diag)
    log_gate(log_row)

    # Pack 1J-Y: metrics-gate-failed.flag lifecycle.
    # Write when this turn's violations include any schema-v2 rule.
    # Clear when this turn carries a valid Metrics Gate PASS table.
    schema_violation_rules = [
        v.get("rule") for v in violations
        if _is_schema_v2_rule(v.get("rule", ""))
    ]
    if schema_violation_rules:
        _metrics_gate_failed_write(session_id, schema_violation_rules)
    elif _answer_has_valid_metrics_gate_pass(last):
        _metrics_gate_failed_clear()

    if not violations:
        print("{}")
        return 0

    bullets = "\n".join(f"- {v['rule']}: {v.get('fix','')}" for v in violations)
    reason = (
        "TRUTH GATE BLOCK -- final answer violates rules. "
        "Self-rewrite per fixes below. Do not ask Al.\n"
        f"{bullets}"
    )
    payload = {
        "decision": "block",
        "reason": reason,
        "continue": False,
        "stopReason": "truth-gate-violation",
        "action": action,
        "rule_ids": [v.get("rule") for v in violations],
    }
    try:
        print(json.dumps(payload))
    except Exception:
        pass
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        try:
            print("{}")
        except Exception:
            pass
        sys.exit(0)
