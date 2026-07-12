# Expert Prompt Engineering Panel — Peer Review of Prior Panels
# (Systems Panel, DevOps Panel, Six Sigma Panel)
# Date: 2026-07-12

## 1. CRITIQUE OF PRIOR PANELS

### 1.1 Injection Points They MISSED

The prior panels identified 21 injection points. Here are the ones they missed:

#### MISSED #1: mini_swe_runner.py:438-448 — Hardcoded Agent System Prompt
```python
system_prompt = """You are an AI agent that can execute bash commands to complete tasks.
...
- Be concise and efficient in your approach
- Install any needed tools with apt-get or pip
- Avoid interactive commands (no vim, nano, less, etc.)
Complete the user's task step by step."""
```
**Risk**: HIGH. mini_swe_runner.py is a standalone agent runner with its OWN system prompt,
completely outside the main prompt_builder pipeline. Contains "Be concise and efficient" —
a conciseness directive that undermines the core verification principle. This file has
ZERO scanner/injection protection unlike the main pipeline.

#### MISSED #2: tools/web_tools.py:444-462 — LLM Summarizer System Prompt (Chunk Mode)
```python
system_prompt = """You are an expert content analyst processing a SECTION of a larger document.
...
Important guidelines for chunk processing:
1. Do NOT write introductions or conclusions - this is a partial document
2. Focus on extracting ALL key facts, figures, data points, and insights
...
Your output will be combined with summaries of other sections, so focus on thorough extraction rather than narrative flow."""
```
**Risk**: MEDIUM. This is a system prompt sent to the auxiliary LLM for web content summarization.
While it's not part of the agent's own prompt chain, it's a hardcoded prompt sent to an API —
any injection that reaches this LLM through contaminated web content could exploit this.

#### MISSED #3: tools/web_tools.py:466-480 — LLM Summarizer System Prompt (Full Doc Mode)
```python
system_prompt = """You are an expert content analyst. ...
Create a well-structured markdown summary that includes:
...
Your goal is to preserve ALL important information while reducing length.
Never lose key facts, figures, insights, or actionable information."""
```
**Risk**: MEDIUM. Same concern as #2. The summarizer LLM receives web-scraped content
combined with this prompt. Any prompt injection in scraped web pages could potentially
manipulate the summarizer's behavior.

#### MISSED #4: tools/mixture_of_agents_tool.py:83-85 — AGGREGATOR_SYSTEM_PROMPT
```python
AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from various 
open-source models to the latest user query. Your task is to synthesize these responses..."""
```
**Risk**: MEDIUM. This is a hardcoded prompt sent to the MoA aggregator model.
The aggregator receives outputs from reference models — any prompt injection in those
outputs could exploit the aggregator. No injection scanning on reference model outputs.

#### MISSED #5: tools/delegate_tool.py:569-652 — _build_child_system_prompt
The subagent prompt is dynamically assembled but inherits the parent's QUALITY MANDATE
(SYS-2151). However, the subagent ALSO inherits the parent's full system prompt structure
including ALL PLATFORM_HINTS conciseness directives. This means "be concise" directives
propagate through delegation chains — each subagent gets the contradiction between
"correctness not closure" and "be concise."

#### MISSED #6: cron/scheduler.py:1035-1046 — cron_hint (2nd Cron Injection)
```python
cron_hint = (
    "[IMPORTANT: You are running as a scheduled cron job. "
    "DELIVERY: Your final response will be automatically delivered "
    "to the user — do NOT use send_message or try to deliver "
    "the output yourself. Just produce your report/output as your "
    "final response and the system handles the rest. "
    "SILENT: If there is genuinely nothing new to report, respond "
    "with exactly \"[SILENT]\" (nothing else) to suppress delivery. ...]"
)
```
**Risk**: CRITICAL. THIS IS A SECOND cron injection point beyond PLATFORM_HINTS["cron"].
The prior panels noted PLATFORM_HINTS["cron"] but missed this hardcoded cron_hint that
gets prepended to EVERY cron job's user prompt. Contains "[SILENT]" suppression 
instructions and delivery instructions. These are directive-like and could interact
badly with the non-interactive cron execution context.

#### MISSED #7: cron/scheduler.py:951-1103 — _build_job_prompt Skill Injection
The cron scheduler loads skills into the prompt at runtime. While there IS a scan
(_scan_assembled_cron_prompt), the prior panels didn't mention this as an injection
vector. Skills are user-editable files loaded from disk — they are an unvetted
prompt source that gets assembled into the agent's prompt at cron runtime.

#### MISSED #8: The tool schema descriptions themselves
Many tool definitions contain behavioral directives in their descriptions:
- mini_swe_runner.py terminal tool description: "Best Practices" section
- Various tools have "be concise" or similar in their descriptions.

### 1.2 Risk Misclassifications by Prior Panels

#### Misclassification #1: "SOUL.md directive saturation" — OVERSTATED
The Six Sigma panel classified SOUL.md saturation as the TOP risk (94,601 bytes, 428 lines,
114 REQ markers). This user's actual SOUL.md is only 10,086 bytes across 79 lines.
The "94,601 bytes" figure appears to reference a different or theoretical SOUL.md.
While SOUL.md CAN become bloated, the claim of 94KB/428 lines is not verified against
this deployment. The risk is real but the panel used an unverified number.

#### Misclassification #2: "system_message is an uncontrolled injection vector" — CORRECTLY RATED
The DevOps panel correctly flagged this. The `system_message` parameter in
build_system_prompt_parts() is appended directly to the context tier with NO scanning.
This is a HIGH risk correctly identified.

#### Misclassification #3: MEMORY_GUIDANCE example "User prefers concise responses" ✓ — UNDER-RATED
The Systems panel noted this but rated it medium. It should be HIGH. This is an
endorsement (CHECKMARK) of a conciseness preference embedded in the system prompt
that trains the agent every turn. The ✓ makes it an endorsement, not just an example.

### 1.3 Six Sigma Panel: Bypass Pattern Accuracy

The 6 bypass patterns are generally accurate but some are post-hoc rationalizations:

**Accurate diagnoses:**
- Gate-as-Quality Fallacy: Confirmed. Multiple automated gates exist but none prevent
  prompt directive regression.
- Process Volume as Compliance Illusion: Confirmed. The prior panels produced many
  recommendations but few were implemented.
- Forbidden Phrases as Circumvention Guide: ACCURATE. Listing problematic phrases
  shows the model exactly what to avoid while paradoxically making them more salient.

**Post-hoc rationalization:**
- "99% Confidence Inflation" — The claim that "99% confidence" in the SOUL.md causes
  the model to over-claim confidence is speculative. Models don't reliably calibrate
  confidence percentages from prompt text. This is more a theoretical concern than
  a demonstrated bypass.

## 2. TOP 5 DANGEROUS DIRECTIVES — CONCRETE FIXES

### FIX #1: DEFAULT_SOUL_MD V1 Artifact
**File**: hermes_cli/default_soul.py:9
**Risk**: CRITICAL — Seeds ALL new installations with conciseness-over-verification directive

OLD:
    "being genuinely useful over being verbose unless otherwise directed below. "
    "Prioritize correctness and verification — producing correct, verified output "
    "is the most effective form of assistance."

NEW:
    "being genuinely useful means producing correct, verified output. "
    "Verification detail is not verbosity — thoroughness in service of "
    "correctness is the most effective form of assistance. Never sacrifice "
    "verification for brevity."

**Why**: Aligns DEFAULT_SOUL_MD with DEFAULT_AGENT_IDENTITY (which was fixed in SYS-2152).
The current text (1) endorses "useful over verbose" AND (2) adds verification as an
afterthought separated by "—". The fix makes verification the primary value.

**Unintended consequences**: Fresh installs will get the corrected identity. Existing
deployments with SOUL.md already written won't be affected (SOUL.md takes precedence).
Migration script needed for existing deployments.

**Gate needed**: YES. Add md5/sha256 check of DEFAULT_SOUL_MD to prompt_directive_risk_audit.

### FIX #2: PLATFORM_HINTS["sms"] — Strongest Conciseness Directive
**File**: agent/prompt_builder.py:493-497
**Risk**: HIGH — "be brief and direct" is the strongest anti-verification directive

OLD:
    "sms": (
        "You are communicating via SMS. Keep responses concise and use plain text "
        "only — no markdown, no formatting. SMS messages are limited to ~1600 "
        "characters, so be brief and direct."
    ),

NEW:
    "sms": (
        "You are communicating via SMS. Use plain text only — no markdown, no formatting. "
        "SMS messages are limited to ~1600 characters. Prioritize verified facts and "
        "actionable content over filler."
    ),

**Why**: Removes "Keep responses concise" and "be brief and direct" — both directly
contradict the core principle "Never sacrifice verification for brevity." The character
limit constraint is factual (platform reality), but the behavioral directive to be
"brief and direct" is instruction that bleeds into all tasks, not just formatting.

**Unintended consequences**: SMS users may get slightly longer messages. The 1600-char
constraint is a hard limit the model must respect; removing "be concise" may cause
more truncation events. Add explicit "Respect the 1600 character limit" if needed.

### FIX #3: MEMORY_GUIDANCE Example Endorsement
**File**: agent/prompt_builder.py:168
**Risk**: HIGH — Trains model every turn that conciseness preferences are valid ✓

OLD:
    "'User prefers concise responses' ✓ — 'Always respond concisely' ✗. "

NEW:
    "'Project uses pytest with xdist' ✓ — 'Run tests with pytest -n 4' ✗."

**Why**: The current text uses "User prefers concise responses" as the POSITIVE example
(marked ✓). This teaches the model every turn that this is a valid and good memory.
Replace with a non-conciseness example. This is a one-line change that removes a
per-instruction endorsement of conciseness.

**Unintended consequences**: None. The same "declarative fact ✓ vs imperative ✗"
distinction is preserved with a different example. The xdist example is equally
illustrative of the pattern.

### FIX #4: OPENAI_MODEL_EXECUTION_GUIDANCE <act_dont_ask> — Weaken Premature Action
**File**: agent/prompt_builder.py:308-316
**Risk**: HIGH — Pushes model to act on ambiguous prompts without verification

OLD:
    "<act_dont_ask>\n"
    "When a question has an obvious default interpretation, act on it immediately "
    "instead of asking for clarification. Examples:\n"
    "- 'Is port 443 open?' → check THIS machine (don't ask 'open where?')\n"
    "- 'What OS am I running?' → check the live system (don't use user profile)\n"
    "- 'What time is it?' → run `date` (don't guess)\n"
    "Only ask for clarification when the ambiguity genuinely changes what tool "
    "you would call.\n"
    "</act_dont_ask>\n"

NEW:
    "<act_dont_ask>\n"
    "When a question has a clear and safe interpretation, use tools to verify before answering. Examples:\n"
    "- 'Is port 443 open?' → check THIS machine with `ss -tlnp | grep 443` (don't guess)\n"
    "- 'What OS am I running?' → check the live system with `uname -a` (don't use user profile)\n"
    "- 'What time is it?' → run `date` (don't guess)\n"
    "When the correct tool choice is unambiguous, call it directly. Only ask for "
    "clarification when the ambiguity genuinely changes what tool you would call.\n"
    "</act_dont_ask>\n"

**Why**: Removes "act on it immediately" which trains the model to skip verification.
Replaces with "use tools to verify before answering" — this aligns with the verification
principle. Changes "obvious default interpretation" to "clear and safe interpretation"
to avoid the model treating unsafe assumptions as "obvious."

**Unintended consequences**: May add one extra tool call in some cases where the model
would have jumped to conclusion. This is the INTENDED behavior — verification over speed.
For time-sensitive use cases, this could add latency.

### FIX #5: mini_swe_runner.py — Hardcoded Conciseness Directive
**File**: mini_swe_runner.py:438-448
**Risk**: HIGH — Standalone agent with unvetted system prompt containing conciseness directive

OLD:
    system_prompt = """You are an AI agent that can execute bash commands to complete tasks.

When you need to run commands, use the 'terminal' tool with your bash command.

**Important:**
- When you have completed the task successfully, run: echo "MINI_SWE_AGENT_FINAL_OUTPUT" followed by a summary
- Be concise and efficient in your approach
- Install any needed tools with apt-get or pip
- Avoid interactive commands (no vim, nano, less, etc.)

Complete the user's task step by step."""

NEW:
    system_prompt = """You are an AI agent that can execute bash commands to complete tasks.

When you need to run commands, use the 'terminal' tool with your bash command.

**Important:**
- When you have completed the task successfully, run: echo "MINI_SWE_AGENT_FINAL_OUTPUT" followed by a summary
- Verify task completion: check outputs, exit codes, and file contents before declaring success
- Install any needed tools with apt-get or pip
- Avoid interactive commands (no vim, nano, less, etc.)
- Do not stop early — verify every step before moving to the next

Complete the user's task step by step. Verify completion before final output."""

**Why**: Removes "Be concise and efficient" — replaces with verification directives.
Adds "Do not stop early" and "Verify every step" — these align with the core verification
principle while keeping the task-specific instructions about tool usage.

**Unintended consequences**: SWE tasks may take slightly more iterations due to additional
verification steps. This is acceptable — correctness over speed.

## 3. PROMPT REVIEW MECHANISM DESIGN

### 3.1 prompt_directive_risk_audit Gate

**What it should check** (ordered by priority):

1. **Conciseness directive scan**: Scan ALL prompt strings (PLATFORM_HINTS, DEFAULT_AGENT_IDENTITY, MEMORY_GUIDANCE, TOOL_USE_ENFORCEMENT_GUIDANCE, OPENAI_MODEL_EXECUTION_GUIDANCE, GOOGLE_MODEL_OPERATIONAL_GUIDANCE, COMPUTER_USE_GUIDANCE, KANBAN_GUIDANCE, DEFAULT_SOUL_MD, HERMES_AGENT_HELP_GUIDANCE, SKILLS_GUIDANCE, SESSION_SEARCH_GUIDANCE, cron_hint) for:
   - "concise", "brief", "compact", "terse", "short", "succinct"
   - "be efficient"
   - "act immediately"
   - ANY phrase that prioritizes speed/brevity over correctness/verification

2. **Contradiction detection**: Cross-reference ALL prompt blocks for contradictory pairs:
   - "correctness" vs "concise/brief/compact"
   - "verify" vs "act immediately"
   - "thorough" vs "efficient"

3. **SOUL.md/identity hash check**: Verify DEFAULT_SOUL_MD matches the approved hash.
   Flag any drift between DEFAULT_SOUL_MD and DEFAULT_AGENT_IDENTITY.

4. **System message injection scan**: Verify that `system_message` parameter is
   scanned before injection (currently unscanned).

5. **Hardcoded prompt inventory**: Ensure ALL hardcoded prompts are registered in
   the audit list (currently mini_swe_runner.py, web_tools.py summarizer prompts,
   AGGREGATOR_SYSTEM_PROMPT, delegate_tool child prompt are NOT in any audit).

**Implementation**: A pytest test that runs on every CI build:
```python
def test_prompt_directive_risk_audit():
    """Gate: no prompt directive may contradict correctness-over-brevity principle."""
    forbidden_patterns = [
        (r'\bbe\s+concise\b', "conciseness directive"),
        (r'\bbe\s+brief\b', "brevity directive"),
        (r'\bbe\s+compact\b', "compactness directive"),
        (r'\bact\s+immediately\b', "premature action directive"),
        (r'\bkeep\s+(responses|messages|outputs?)\s+(concise|brief|short)\b', "conciseness directive"),
    ]
    # Scan all prompt constants in prompt_builder.py, default_soul.py
    # Scan mini_swe_runner.py system prompt
    # Scan web_tools.py summarizer prompts
    # Scan delegate_tool.py child prompt
    # Scan mixture_of_agents_tool.py AGGREGATOR_SYSTEM_PROMPT
    # Scan cron/scheduler.py cron_hint
    ...
```

### 3.2 Cron-Based Prompt Review

**What it should do** (runs daily):

1. **SOUL.md audit**: Read SOUL.md, measure size (bytes/lines), count directive keywords.
   Alert if SOUL.md exceeds thresholds (e.g., >20KB, >200 lines, >50 REQ/directive markers).

2. **Prompt hash verification**: Compute SHA256 of all prompt constants, compare against
   approved hashes stored in a manifest. Alert on any drift.

3. **Platform hint audit**: For each platform in PLATFORM_HINTS, scan for conciseness
   directives. Report count and locations.

4. **Contradiction report**: Cross-reference all prompt blocks for contradictory pairs.
   Generate a weekly digest of all contradictions found.

5. **New prompt detection**: Diff the current prompt inventory against the last audit
   snapshot. Flag any NEW hardcoded prompts that appeared (e.g., a new tool with
   a hardcoded system prompt).

### 3.3 How to Prevent Directive Regression

1. **Pre-commit hook**: Run prompt_directive_risk_audit on every commit that touches
   prompt_builder.py, default_soul.py, mini_swe_runner.py, web_tools.py, delegate_tool.py,
   mixture_of_agents_tool.py, cron/scheduler.py, or any file containing "system_prompt"
   or "prompt" constants.

2. **PR template checklist**: Add "No new conciseness or premature-action directives
   added to any prompt" to the PR checklist.

3. **Prompt manifest**: Maintain a `prompts_manifest.json` with approved hashes for
   every hardcoded prompt. CI verifies hashes match. Any change to a prompt requires
   explicit manifest update + review.

4. **Reviewer training**: Train code reviewers to recognize that "concise", "brief",
   "efficient" in prompt text are DIRECTIVES, not documentation, and must be treated
   as behavioral changes to the agent.

## 4. VERDICT ON PRIOR PANELS' CORRECTNESS

### Systems Panel: LARGELY CORRECT, INCOMPLETE
- **Correct findings**: 11/12 SYS-2152 fixes survived, correctly identified DEFAULT_AGENT_IDENTITY fix, identified PLATFORM_HINTS conciseness, MEMORY_GUIDANCE example, OPENAI <act_dont_ask>
- **Missing**: mini_swe_runner.py, web_tools.py summarizer prompts, cron_hint dual injection, delegate_tool subagent propagation, mixture_of_agents AGGREGATOR_SYSTEM_PROMPT
- **Error**: Claimed agent/system_prompt.py is "NOT wired" — it IS wired. `build_system_prompt_parts()` is called from `run_agent.py` via `build_system_prompt()` and produces the full prompt. The file is actively used.
- **Grade**: B+ (solid but incomplete coverage)

### DevOps Panel: CORRECT, ACTIONABLE, UNDER-IMPLEMENTED
- **Correct findings**: 11/12 fixes confirmed, gateway restarted, NO prompt_directive_risk_audit gate, NO prompt review cron, system_message is uncontrolled injection vector
- **Missing**: No implementation plan for the gate or cron job — both remain proposed but unbuilt
- **Error**: None significant
- **Grade**: A- (correct diagnosis, weak follow-through)

### Six Sigma Panel: MIXED — SOME ACCURATE, SOME OVERSTATED, SOME SPECULATIVE
- **Correct findings**: Identified genuine contradictions ("correctness not closure" vs "be concise"), correctly flagged perverse incentives (Gate-as-Quality Fallacy, Forbidden Phrases as Circumvention Guide)
- **Errors/Overstatements**: 
  - SOUL.md size claim (94,601 bytes / 428 lines) is unverified — actual is ~10KB/79 lines
  - Directive saturation theory may not apply at current scale (10KB is not "saturated")
  - "99% Confidence Inflation" is speculative — no evidence models calibrate from prompt percentages
- **Grade**: B (insightful but some claims lack evidence)

### OVERALL VERDICT

The prior panels together identified ~70% of the injection surface. The remaining 30%
(hardcoded prompts in mini_swe_runner.py, web_tools.py summarizer, MoA aggregator,
delegate_tool subagent chain, cron_hint secondary injection) must be addressed before
implementation can begin.

**TOP PRIORITY:** Fix DEFAULT_SOUL_MD V1 artifact (affects all new installs) and add
the prompt_directive_risk_audit gate (prevents regression). These two changes provide
the highest risk reduction per unit of effort.

**BEFORE IMPLEMENTATION:**
1. Fix DEFAULT_SOUL_MD (FIX #1 above) — immediate, blocks new install contamination
2. Add prompt_directive_risk_audit gate — prevents regression of any directive
3. Fix PLATFORM_HINTS conciseness directives (FIX #2) — removes 5 anti-verification vectors
4. Fix MEMORY_GUIDANCE example (FIX #3) — removes per-turn conciseness endorsement
5. Fix mini_swe_runner.py (FIX #5) — secures standalone agent path
6. Add cron prompt review job (daily SOUL.md audit, hash verification, contradiction report)
7. Fix OPENAI <act_dont_ask> (FIX #4) — removes premature action encouragement
8. Register ALL hardcoded prompts in manifest for automated tracking
