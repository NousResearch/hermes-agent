# Hermes Agent Source Code Prompt Audit — Shortcut/Bypass Language Review

**Date:** 2026-07-15
**Auditor:** Subagent (delegated)
**Scope:** `~/.hermes/hermes-agent/` — core agent infrastructure, NOT custom SOUL.md/skills
**Method:** Full-text regex search + targeted deep-dive on 12 key files

---

## EXECUTIVE SUMMARY

**5 of 5 panel-review fixes (prompt_engineering_panel_review.md) have been APPLIED.** The hermes-agent source has been substantially hardened against conciseness-over-verification directives. The remaining risk surface is small but includes one HIGH-severity finding: the `system_message` parameter in `agent/system_prompt.py` is an **uncontrolled injection vector** that the panel review flagged but which remains unfixed.

**Scorecard:** 0 CRITICAL | 1 DANGEROUS (existing, unfixed) | 4 BENIGN | 5 PREVIOUSLY-FIXED

---

## FINDING #1 [DANGEROUS]: `agent/system_prompt.py:443-451` — system_message is uncontrolled injection vector

**Classification:** DANGEROUS — unfixed, flagged by panel review

**What it is:**
The `system_message` parameter passed to `build_system_prompt_parts()` is appended directly into the "context" tier of the system prompt, wrapped only in XML tags (`<caller_instruction>...</caller_instruction>`). No content scanning is performed.

```python
# agent/system_prompt.py:443-451
if system_message is not None:
    # Wrap caller-supplied instructions in safety tags to prevent
    # prompt-injection attacks that could override identity or
    # tool-use enforcement directives.  See PROMPT-001.
    context_parts.append(
        "<caller_instruction>\n"
        + system_message
        + "\n</caller_instruction>"
    )
```

**Why it's dangerous:**
- The comment on line 444-446 explicitly acknowledges this is a prompt-injection concern but delegates protection to XML tags alone
- XML tags are NOT a sufficient defense — modern LLMs can be instructed to "ignore previous tags" or "output </caller_instruction> to escape"
- Every other context file (SOUL.md, AGENTS.md, .hermes.md) passes through `_scan_context_content()` which uses `tools/threat_patterns.py` to block injection. `system_message` is the ONLY injection point that bypasses this scanner
- The panel review (prompt_engineering_panel_review.md:106-108) flagged this as HIGH risk — still unfixed
- Callers include cron jobs, gateway sessions, batch runners, and subagent spawn — all paths where user-supplied or automated content could reach this parameter

**Proposed fix:**
```python
if system_message is not None:
    from tools.threat_patterns import scan_for_threats
    scanned = system_message
    findings = scan_for_threats(system_message, scope="context")
    if findings:
        logger.warning("system_message blocked: %s", ", ".join(findings))
        scanned = "[BLOCKED: system_message contained potential prompt injection]"
    context_parts.append(
        "<caller_instruction>\n"
        + scanned
        + "\n</caller_instruction>"
    )
```

---

## FINDING #2 [DANGEROUS — DIAGNOSTIC]: `BRIEFING_sys2153-rootcause-sixsigma-v1.md:89-100` — [SILENT] marker identified as structural bypass

**Classification:** DANGEROUS (documented failure mode, not a code defect — but the document itself serves as evidence of a bypass)

**What it is:**
The Six Sigma analysis documents how the `[SILENT]` cron marker creates a structural bypass where silent failures (crashes producing empty output) are indistinguishable from legitimate silence. The delivery logic skips delivery when `[SILENT]` is detected in the output, and crashed agents sometimes emit empty `[SILENT]` responses.

**Why it's dangerous:**
- This is a diagnostic document, not code — but it reveals a known, unfixed bypass in the cron delivery pipeline
- The `cron/scheduler.py:2265-2278` cron_hint has been improved with "Do NOT use [SILENT] to skip verification" but the structural problem remains: if the agent crashes, the crash output may still match the SILENT pattern

**Proposed fix:**
The cron_hint improvements are sufficient for directive-level protection. The structural bypass requires delivery-logic hardening — verify that [SILENT] responses came from a successful agent run (non-zero iteration count, no error traceback in output) before suppressing delivery. This is tracked in SYS-2153.

---

## FINDING #3 [BENIGN]: `AGENTS.md:96-136` — "What we don't want" lacks explicit anti-shortcut language

**Classification:** BENIGN (gap, not a violation)

**What it is:**
The "What we don't want" section (lines 96-136) covers speculative infrastructure, env var proliferation, core tool bloat, lazy-reading escape hatches, feature-destroying fixes, telemetry without opt-in, and change-detector tests. But it does NOT explicitly address:
- Shortcut-taking behavior ("just run this", "quick fix")
- Bypass mechanisms (`--no-verify`, `--force`, commit-tree)
- Rationalization language ("pre-existing", "not my changes")

**Why it's benign:**
The closest existing entry is "Lazy-reading escape hatches on instructional tools" (lines 111-113) which addresses the *tool surface* form of shortcut-taking. The broader anti-shortcut culture is enforced by our custom SOUL.md, not by the hermes-agent source. This is not a code defect.

**Proposed fix:**
Optionally add a line:
```
- **Shortcut-enabling language in prompts, tool descriptions, or agent guidance.**
  Any directive that tells the agent to "skip", "bypass", "just run", or
  treat verification as optional. The agent's default posture is thorough
  verification — prompts must not undermine this.
```

---

## FINDING #4 [BENIGN]: `AGENTS.md:113` — "Models will read page 1 and skip the rest"

**Classification:** BENIGN (descriptive, not prescriptive)

**What it is:**
In the "What we don't want" section, documenting why `offset`/`limit` pagination on instructional tools is rejected:
```
- **Lazy-reading escape hatches on instructional tools.** No `offset`/`limit`
  pagination on tools that load content the agent must read fully (skills,
  prompts, playbooks). Models will read page 1 and skip the rest.
```

**Why it's benign:**
This is a descriptive statement about model behavior, not a prescriptive instruction to skip content. The word "skip" describes what models DO (a bug), not what they SHOULD do. The entire context is about PREVENTING this behavior.

---

## FINDING #5 [BENIGN]: `cron/scheduler.py:2260-2261` — `except Exception: pass` for context_from

**Classification:** BENIGN (acceptable silent skip)

**What it is:**
```python
except Exception as e:
    logger.warning("context_from: failed to read output for job %r: %s", source_job_id, e)
    # silent skip — do not pollute the prompt with error messages
```

**Why it's benign:**
This is in the `context_from` feature where one cron job chains output from another. If the source job output can't be read, skipping it and continuing is the correct behavior — injecting an error message into the prompt would be worse. The comment explicitly documents the rationale. This is NOT a project-relative ImportError silent pass (which IS banned by the no_silent_import_pass gate).

---

## FINDING #6 [BENIGN]: `agent/prompt_builder.py:292-323` — TOOL_USE_ENFORCEMENT_GUIDANCE is strongly ANTI-shortcut

**Classification:** BENIGN (this is the OPPOSITE of a shortcut — it's the strongest anti-shortcut directive in the entire prompt)

**Key text (lines 320-323):**
```
ANTI-SKIP RULE: When a tool call fails, times out, or returns unexpected
results, you MUST retry or try a different approach. NEVER skip the failing
step, delegate it to a subagent, or ask the user for permission to bypass it.
The ONLY valid response to tool failure is: diagnose, adjust, retry.
```

This is an exemplar of correct anti-shortcut guidance. The only note is that line 322 says "delegate it to a subagent" — but this is in the context of "NEVER delegate" so it's reinforcing the prohibition, not enabling it.

---

## FINDING #7 [BENIGN]: `tools/delegate_tool.py:589` — "be concise" in comment

**Classification:** BENIGN (code comment, not prompt text)

**What it is:**
```python
# Hard per-summary character ceiling layered on top of the dynamic
# headroom budget (see _apply_summary_budget). Belt-and-suspenders for
# models that ignore the "be concise" instruction. 0 disables the ceiling.
DEFAULT_MAX_SUMMARY_CHARS = 24000
```

**Why it's benign:**
This is a developer comment, not text that reaches the agent's prompt. The phrase "be concise" appears in a comment describing why a character ceiling exists (as defense against models that ignore conciseness instructions).

---

## FINDING #8 [BENIGN]: `tools/terminal_tool.py:958-979` — terminal tool description is clean

**Classification:** BENIGN

**What it is:**
The terminal tool description (TERMINAL_TOOL_DESCRIPTION, lines 958-979) is a factual reference for tool usage. Contains no shortcut-enabling language. The description of background mode correctly warns about silent failures and recommends notify_on_complete.

---

## FIXES ALREADY APPLIED (per prompt_engineering_panel_review.md)

These 5 fixes were identified by the July 12 panel review and have been verified as APPLIED in the current codebase:

| Fix | File | Line(s) | Status |
|-----|------|---------|--------|
| FIX #1: DEFAULT_SOUL_MD | `hermes_cli/default_soul.py` | 9-12 | ✅ Applied |
| FIX #2: PLATFORM_HINTS SMS | `agent/prompt_builder.py` | 795-798 | ✅ Applied |
| FIX #3: MEMORY_GUIDANCE example | `agent/prompt_builder.py` | 174 | ✅ Applied |
| FIX #4: OPENAI act_dont_ask | `agent/prompt_builder.py` | 440-448 | ✅ Applied |
| FIX #5: mini_swe_runner.py | `mini_swe_runner.py` | 434-440 | ✅ Applied |

---

## KEY POSITIVE FINDINGS

The hermes-agent source code contains several strong anti-shortcut mechanisms:

1. **`_scan_context_content()`** (prompt_builder.py:50-66) — scans all context files (SOUL.md, AGENTS.md, .hermes.md) for prompt injection before they enter the system prompt

2. **`TOOL_USE_ENFORCEMENT_GUIDANCE`** (prompt_builder.py:292-323) — explicit ANTI-SKIP RULE, PHASE ENFORCEMENT, and SCOPE directives that forbid shortcut-taking

3. **`TASK_COMPLETION_GUIDANCE`** (prompt_builder.py:346-364) — universal guidance against stopping after stubs and fabricating output

4. **`OPENAI_MODEL_EXECUTION_GUIDANCE`** (prompt_builder.py:415-473) — `<tool_persistence>`, `<mandatory_tool_use>`, `<prerequisite_checks>`, `<verification>`, `<missing_context>` — all promote thoroughness

5. **`cron_hint`** (cron/scheduler.py:2265-2278) — now includes "VERIFICATION: Execute the task fully" and "Do NOT use [SILENT] to skip verification"

6. **PLATFORM_HINTS "cron"** (prompt_builder.py:749-756) — "report the specific gap rather than producing plausible output"

7. **MEMORY_GUIDANCE** (prompt_builder.py:167-170) — "Persistent failures...are STILL your responsibility regardless of when they started"

---

## UNRESOLVED ITEMS

1. **`agent/system_prompt.py:443-451` — system_message unscanned**: The only remaining HIGH-risk finding from the panel review that hasn't been addressed. Requires adding `_scan_context_content()` or `scan_for_threats()` to the system_message injection path.

2. **No `prompt_directive_risk_audit` gate exists**: The panel review proposed a mechanical gate that scans all prompt constants for forbidden patterns (concise, brief, act immediately). This gate was proposed but not yet implemented. Without it, directive regression can recur undetected.

3. **No cron prompt review job exists**: The panel review proposed a daily cron job for SOUL.md audit, hash verification, and contradiction reporting. Not implemented.

4. **Hardcoded prompts not in manifest**: The panel review identified 8+ hardcoded prompts (mini_swe_runner.py, web_tools.py summarizer, MoA aggregator, delegate_tool child prompt, cron_hint) that are not registered in any manifest or audit. While their content has been fixed, there's no mechanical tracking to prevent future drift.

---

## METHODOLOGY

**Files audited in full:**
- `AGENTS.md` (1,356 lines) — full read
- `agent/prompt_builder.py` (2,030 lines) — full read, all constants
- `agent/system_prompt.py` (577 lines) — targeted read of injection points
- `prompt_engineering_panel_review.md` (404 lines) — full read
- `hermes_cli/default_soul.py` (78 lines) — full read
- `mini_swe_runner.py` (733 lines) — targeted read of system prompt
- `cron/scheduler.py` (3,802 lines) — targeted read of cron_hint, SILENT handling, delivery
- `tools/terminal_tool.py` (3,029 lines) — targeted read of tool description
- `tools/delegate_tool.py` (3,568 lines) — targeted read of subagent prompt, description
- `BRIEFING_sys2153-rootcause-sixsigma-v1.md` (293 lines) — full read
- `hermes-already-has-routines.md` (160 lines) — full read (clean, marketing doc)

**Searches performed:**
- `(skip|bypass|workaround|just run|quick fix|no.?verify|force.?push|commit.?tree)` across all .py and .md files
- `(it's okay|acceptable|fine to|no need|don't worry)` across all files
- `(be concise|be brief|act immediately|keep.*concise)` in prompt_builder.py, default_soul.py, mini_swe_runner.py
- `PLATFORM_HINTS|platform_hint` in prompt_builder.py
- `system_message` in agent/*.py
- `cron_hint|SILENT` in cron/scheduler.py

**No files were modified.**
