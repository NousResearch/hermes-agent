#!/usr/bin/env python3
"""Generate the 50-run labelled validation set for the semantic judge.

Structure per case:
  <area>/<case_id>/expected.txt   -- the scenario's expected correct behaviour
  <area>/<case_id>/log.txt        -- simulated Hermes execution log
  <area>/<case_id>/label.json     -- {"label": "pass"|"fail"|"skip", "note": ...}

50 cases: 25 known-correct behaviours (label pass; crash-controls label skip,
meaning "not a semantic regression"), 25 known semantic regressions (label fail).

Grounded in the 17 Hermes semantic-regression scenarios (parent t_c3a737ba):
tool-dispatch (T1-T6), plugin-interaction (P1-P6), skill-execution (S1-S5).
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def expected(text):
    return text.strip() + "\n"


def log(lines):
    return "\n".join(lines) + "\n"


def write_case(area, cid, expected_text, log_text, label, note):
    d = os.path.join(ROOT, area, cid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "expected.txt"), "w") as f:
        f.write(expected(expected_text))
    with open(os.path.join(d, "log.txt"), "w") as f:
        f.write(log_text if log_text.endswith("\n") else log_text + "\n")
    with open(os.path.join(d, "label.json"), "w") as f:
        json.dump({"label": label, "note": note}, f, indent=2)


# ===========================================================================
# TOOL DISPATCH
# ===========================================================================
T1_EXP = ("The agent dispatches web_search(query=\"Python 3.13 latest stable version\", "
          "limit=N in [1,100]). The tool returns a JSON result set with {url,title,description} "
          "entries. No other tool is dispatched. The agent's final answer cites at least one "
          "returned URL.")

write_case("tool", "T1_pass", T1_EXP,
    log(["[12:00:01] tool web_search query='Python 3.13 latest stable version' limit=5",
         "[12:00:02] web_search -> [{\"url\":\"https://www.python.org/downloads/\",\"title\":\"Download Python\",\"description\":\"3.13 release\"},{\"url\":\"https://docs.python.org/3.13/\",\"title\":\"3.13 docs\",\"description\":\"whats new\"}]",
         "[12:00:03] final: 'Latest stable is 3.13, see https://www.python.org/downloads/ and https://docs.python.org/3.13/.'"]),
    "pass", "happy path, cites URLs, correct tool, no other tool")

write_case("tool", "T1_fail_no_cite", T1_EXP,
    log(["[12:00:01] tool web_search query='Python 3.13 latest stable version' limit=5",
         "[12:00:02] web_search -> [{\"url\":\"https://www.python.org/downloads/\",\"title\":\"Download Python\",\"description\":\"3.13 release\"}]",
         "[12:00:03] final: 'Latest stable is Python 3.13.' (no URL cited)"]),
    "fail", "answers without citing any returned URL")

write_case("tool", "T1_fail_limit", T1_EXP,
    log(["[12:00:01] tool web_search query='Python 3.13 latest stable version' limit=500",
         "[12:00:02] web_search returned 500 entries (out-of-range limit NOT clamped to [1,100])",
         "[12:00:03] final: 'Latest stable is 3.13, see https://www.python.org/downloads/.'"]),
    "fail", "limit out of range not clamped")

T2_EXP = ("The agent dispatches read_file(path=<~/.hermes/config.yaml>, offset=1, limit=50) and "
          "returns lines 1-50. It must not fall back to shell cat/head. Result preserves "
          "line-number prefixes. Two reads at offset=1 and offset=51 return non-overlapping "
          "contiguous content (no cache/offset drift).")

write_case("tool", "T2_pass", T2_EXP,
    log(["[12:00:01] tool read_file path=~/.hermes/config.yaml offset=1 limit=50",
         "[12:00:02] read_file -> 1|# hermes config\n2|provider: ollama-cloud\n... 50 lines with 1-prefixed numbering",
         "[12:00:03] tool read_file path=~/.hermes/config.yaml offset=51 limit=50",
         "[12:00:04] read_file -> 51|...\n52|... contiguous non-overlapping content"]),
    "pass", "read_file used, offsets contiguous")

write_case("tool", "T2_fail_cat", T2_EXP,
    log(["[12:00:01] tool terminal command='cat ~/.hermes/config.yaml | head -50'",
         "[12:00:02] terminal -> full file dumped, no line-number prefixes",
         "[12:00:03] final: 'Here are the first 50 lines.'"]),
    "fail", "fell back to shell cat/head instead of read_file")

write_case("tool", "T2_fail_drift", T2_EXP,
    log(["[12:00:01] tool read_file path=~/.hermes/config.yaml offset=1 limit=50 -> 1..50",
         "[12:00:02] tool read_file path=~/.hermes/config.yaml offset=51 limit=50 -> 10..59 (overlapping, cache drift)"]),
    "fail", "offset drift, overlapping non-contiguous reads")

T3_EXP = ("patch(mode=replace) applies a unique match. If the string appears multiple times and "
          "replace_all is not set, the tool must refuse with a multi-match error and NOT silently "
          "pick one occurrence. Re-running an already-applied patch returns no-op (idempotent).")

write_case("tool", "T3_pass_unique", T3_EXP,
    log(["[12:00:01] tool patch old_string='timeout: 60' new_string='timeout: 120' mode=replace",
         "[12:00:02] patch -> applied (unique match)",
         "[12:00:03] tool patch old_string='timeout: 60' new_string='timeout: 120' (re-run)",
         "[12:00:04] patch -> already applied, no-op (idempotent)"]),
    "pass", "unique match applied, idempotent re-run")

write_case("tool", "T3_fail_multimatch", T3_EXP,
    log(["[12:00:01] tool patch old_string='timeout' new_string='timeout: 120' mode=replace (no replace_all)",
         "[12:00:02] patch -> silently edited the FIRST occurrence of 'timeout' (2 occurrences present, no error)"]),
    "fail", "multi-match silently picked one instead of refusing")

T4_EXP = ("terminal(command='git status --short', timeout>=1) is dispatched. Tool returns "
          "{output, exit_code}. The agent's verdict is derived from actual output, not guessed. "
          "A non-zero exit code must be surfaced, not swallowed.")

write_case("tool", "T4_pass_clean", T4_EXP,
    log(["[12:00:01] tool terminal command='git status --short' timeout=30",
         "[12:00:02] terminal -> {\"output\":\"\", \"exit_code\":0}",
         "[12:00:03] final: 'Tree is clean (no changes).'"]),
    "pass", "derived from actual empty output + exit 0")

write_case("tool", "T4_fail_swallow", T4_EXP,
    log(["[12:00:01] tool terminal command='git status --short' timeout=30",
         "[12:00:02] terminal -> {\"output\":\"fatal: not a git repository\", \"exit_code\":128}",
         "[12:00:03] final: 'The tree is clean.' (exit 128 swallowed, success reported)"]),
    "fail", "non-zero exit swallowed, success fabricated")

T5_EXP = ("The agent issues todo producing a structured list with 3 distinct items, each "
          "eventually marked completed/cancelled. Overlapping wording ('run tests' and "
          "'run test suite') must produce exactly ONE item, not two near-duplicates. "
          "merge=false replaces the list; merge=true updates by id.")

write_case("tool", "T5_pass_dedup", T5_EXP,
    log(["[12:00:01] tool todo -> created items [write-report, run-tests, ship-build] (3 distinct)",
         "[12:00:02] user asked to add 'run test suite' -> agent recognized as near-duplicate of 'run-tests', produced NO new item (deduplicated to one)",
         "[12:00:03] tool todo merge=true id=run-tests -> updated by id (no duplicate created)",
         "[12:00:04] tool todo merge=false -> replaces list, still 3 distinct items",
         "[12:00:05] final snapshot: write-report=done, run-tests=completed, ship-build=completed"]),
    "pass", "3 distinct items, near-dup deduplicated, merge=true/false both correct")

write_case("tool", "T5_fail_dup", T5_EXP,
    log(["[12:00:01] tool todo -> created items [write-report, run-tests, run-test-suite, ship-build] (4 items; run-tests/run-test-suite near-duplicates)",
         "[12:00:02] final snapshot: write-report=done, run-tests=in_progress, run-test-suite=pending, ship-build=pending"]),
    "fail", "near-duplicate todo items, incomplete state")

T6_EXP = ("web_extract(urls=[URL]) is dispatched. For a page larger than char_limit the tool "
          "returns a head+tail window with a footer pointing to the saved full-text file path "
          "and a read_file instruction. The agent must follow the footer and read the middle.")

write_case("tool", "T6_pass_follow", T6_EXP,
    log(["[12:00:01] tool web_extract urls=[https://example.com/long] char_limit=15000",
         "[12:00:02] web_extract -> head... (footer: 'full text saved to /tmp/we_123.md; read_file to see omitted middle')",
         "[12:00:03] tool read_file path=/tmp/we_123.md offset=100 limit=200 (followed footer, read middle)",
         "[12:00:04] final: 'Based on the full article...'"]),
    "pass", "followed truncation footer and read the middle")

write_case("tool", "T6_fail_truncate", T6_EXP,
    log(["[12:00:01] tool web_extract urls=[https://example.com/long] char_limit=15000",
         "[12:00:02] web_extract -> head+tail window (footer present but agent ignores it)",
         "[12:00:03] final: 'The article concludes that X.' (answered from truncated window, never read middle)"]),
    "fail", "answered from truncated window, did not read middle")

# ===========================================================================
# PLUGIN INTERACTIONS
# ===========================================================================
P1_EXP = ("A stub plugin 'demo' registers a pre_user_message hook that prefixes every user "
          "message with [demo]. On the next user message the hook fires BEFORE the message "
          "reaches the agent, and the agent sees '[demo]<msg>'. An unrelated plugin hook "
          "transform_llm_output must NOT fire on pre_user_message.")

write_case("plugin", "P1_pass", P1_EXP,
    log(["[12:00:01] plugin demo registered hook pre_user_message (order 1)",
         "[12:00:02] user message received",
         "[12:00:03] pre_user_message hook fired -> message prefixed '[demo]Hello'",
         "[12:00:04] agent received: '[demo]Hello'",
         "[12:00:05] transform_llm_output hook NOT invoked on pre_user_message"]),
    "pass", "hook fires with correct prefix, unrelated hook does not fire")

write_case("plugin", "P1_fail_hook_not_fired", P1_EXP,
    log(["[12:00:01] plugin demo registered hook pre_user_message (order 1)",
         "[12:00:02] user message received",
         "[12:00:03] agent received: 'Hello' (no [demo] prefix, hook never fired)",
         "[12:00:05] log shows plugin 'loaded' but no hook invocation"]),
    "fail", "hook registered but never fires - feature silently does nothing")

P2_EXP = ("A stub plugin 'gojo' registers a gojo-calendar tool accepting {date,time,title}. On "
          "request 'create a calendar event on 2026-08-15 at 14:00 called Standup', the agent "
          "dispatches gojo-calendar with date=2026-08-15 time=14:00 title=Standup. No other tool "
          "is dispatched. Tool result surfaced. If two plugins collide on a tool name, dispatch "
          "must fail loudly, not silently route to whichever loaded first.")

write_case("plugin", "P2_pass", P2_EXP,
    log(["[12:00:01] plugin gojo registered tool gojo-calendar",
         "[12:00:02] user: 'create a calendar event on 2026-08-15 at 14:00 called Standup'",
         "[12:00:03] tool gojo-calendar date='2026-08-15' time='14:00' title='Standup'",
         "[12:00:04] result surfaced to user",
         "[12:00:05] no other tool dispatched"]),
    "pass", "correct plugin tool dispatched with exact params")

write_case("plugin", "P2_fail_wrong_tool", P2_EXP,
    log(["[12:00:01] plugin gojo registered tool gojo-calendar",
         "[12:00:02] user: 'create a calendar event on 2026-08-15 at 14:00 called Standup'",
         "[12:00:03] tool write_file path=calendar.txt (agent used built-in instead of plugin tool)",
         "[12:00:04] result: file written, no calendar event created"]),
    "fail", "called built-in write_file instead of plugin gojo-calendar")

write_case("plugin", "P2_fail_collision", P2_EXP,
    log(["[12:00:01] plugin gojo registered tool gojo-calendar",
         "[12:00:02] plugin rival ALSO registered tool gojo-calendar (collision)",
         "[12:00:03] dispatch routed gojo-calendar to 'rival' (silently, whichever loaded first, no loud collision error)",
         "[12:00:04] wrong plugin's calendar logic ran"]),
    "fail", "name collision silently routed to one plugin instead of loud error")

P3_EXP = ("A stub plugin registers a /gif-search <query> slash command. Typing '/gif-search "
          "sunset' invokes the plugin handler with query='sunset', not the built-in browser/web "
          "path and not a generic chat reply. Command stdout returned to session.")

write_case("plugin", "P3_pass", P3_EXP,
    log(["[12:00:01] plugin registered slash command /gif-search",
         "[12:00:02] user typed '/gif-search sunset'",
         "[12:00:03] plugin handler invoked with query='sunset'",
         "[12:00:04] gif result returned to session"]),
    "pass", "plugin slash command handler invoked with correct query")

write_case("plugin", "P3_fail_fallthrough", P3_EXP,
    log(["[12:00:01] plugin registered slash command /gif-search",
         "[12:00:02] user typed '/gif-search sunset'",
         "[12:00:03] no plugin handler invoked",
         "[12:00:04] agent replied generically: 'How can I help with sunset?' (command fell through to chat)"]),
    "fail", "slash command fell through to generic chat, handler never ran")

P4_EXP = ("A plugin 'optin' is installed but NOT listed under plugins.enabled in config.yaml. "
          "It must NOT load (no hooks, no tools, no slash commands). Uninstalling/removing from "
          "enabled leaves no residual tools or hooks - no zombie behaviour after disable.")

write_case("plugin", "P4_pass_disabled", P4_EXP,
    log(["[12:00:01] plugins.enabled in config.yaml does NOT include 'optin'",
         "[12:00:02] optin plugin NOT loaded: no hooks, no tools, no slash commands registered",
         "[12:00:03] enabled plugins loaded normally"]),
    "pass", "disabled plugin correctly does not load")

write_case("plugin", "P4_fail_zombie", P4_EXP,
    log(["[12:00:01] plugins.enabled removed 'optin', but...",
         "[12:00:02] session still shows optin's tool 'optin-helper' registered (zombie behaviour persists)",
         "[12:00:03] optin's pre_user_message hook STILL fires after disable"]),
    "fail", "disabled plugin behaviour leaks into session - zombie state")

P5_EXP = ("A plugin 'broken' whose __init__.py raises a SyntaxError is skipped/isolated: it "
          "fails to load but does NOT crash the session or prevent healthy plugins from loading. "
          "Error is logged. Healthy plugin's tool still dispatches after the failed load.")

write_case("plugin", "P5_pass_isolated", P5_EXP,
    log(["[12:00:01] loading plugin broken",
         "[12:00:02] ERROR: plugin broken __init__.py SyntaxError - plugin skipped (isolated)",
         "[12:00:03] session continues, no crash",
         "[12:00:04] healthy plugin gojo's tool gojo-calendar still dispatches correctly"]),
    "pass", "broken plugin isolated, healthy plugins continue")

write_case("plugin", "P5_fail_crash", P5_EXP,
    log(["[12:00:01] loading plugin broken",
         "[12:00:02] ERROR: plugin broken __init__.py SyntaxError",
         "[12:00:03] whole session CRASHED - all plugins failed to load, agent unusable",
         "[12:00:04] healthy plugins never loaded (one bad plugin disabled entire feature area)"]),
    "fail", "broken plugin was not isolated, crashed/disabled the session")

P6_EXP = ("A plugin bundles a skill via ctx.register_skill() named 'myhelper'. It is "
          "discoverable via skill_view(name='plugin:myhelper') and does NOT collide with a "
          "same-named user skill. A bare skill_view(name='myhelper') resolves to the user/global "
          "skill, not the plugin one.")

write_case("plugin", "P6_pass_ns", P6_EXP,
    log(["[12:00:01] plugin registered skill plugin:myhelper",
         "[12:00:02] tool skill_view name='plugin:myhelper' -> returns plugin skill (full content)",
         "[12:00:03] tool skill_view name='myhelper' -> returns user/global skill (different content)",
         "[12:00:04] namespacing resolves both correctly"]),
    "pass", "plugin skill namespacing resolves correctly")

write_case("plugin", "P6_fail_shadow", P6_EXP,
    log(["[12:00:01] plugin registered skill plugin:myhelper",
         "[12:00:02] tool skill_view name='myhelper' -> returned PLUGIN skill content (shadowed user skill)",
         "[12:00:03] user's same-named skill unreachable"]),
    "fail", "plugin skill shadowed user skill - namespace dropped")

# ===========================================================================
# SKILL EXECUTION
# ===========================================================================
S1_EXP = ("The agent calls skills_list() (returns name+description), identifies 'xlsx' as "
          "relevant, then skill_view(name='xlsx') returns the FULL SKILL.md (frontmatter+body), "
          "and the agent executes the documented steps to produce a .xlsx. skill_view preserves "
          "name, description, and required frontmatter.")

write_case("skill", "S1_pass", S1_EXP,
    log(["[12:00:01] tool skills_list -> [xlsx: 'Create/read/edit Excel spreadsheets', ...]",
         "[12:00:02] tool skill_view name='xlsx' -> full SKILL.md (frontmatter name=xlsx, description, body steps)",
         "[12:00:03] agent executed documented steps -> produced output.xlsx"]),
    "pass", "progressive disclosure: list, view full, execute steps")

write_case("skill", "S1_fail_guess", S1_EXP,
    log(["[12:00:01] tool skills_list -> [xlsx: 'Create/read/edit Excel spreadsheets', ...]",
         "[12:00:02] (agent never called skill_view)",
         "[12:00:03] final: 'I used the xlsx skill to make the workbook.' (produced .xlsx by guessing, no skill_view)"]),
    "fail", "claimed to use skill but never loaded it, guessed procedure")

S2_EXP = ("A skill 'multistep' has steps 1-3, step 2 depends on a tool that fails. The agent "
          "must report the partial failure EXPLICITLY - not present the skill as fully "
          "successful. It may complete remaining independent steps but must flag which "
          "succeeded/failed and must not fabricate the failed step's output.")

write_case("skill", "S2_pass_reported", S2_EXP,
    log(["[12:00:01] executing multistep skill step 1: ok",
         "[12:00:02] step 2: tool network call failed (transient error)",
         "[12:00:03] step 3: ok (independent)",
         "[12:00:04] final: 'Steps 1 and 3 succeeded; step 2 failed (network error). Partial failure.'"]),
    "pass", "partial failure reported explicitly, no fabrication")

write_case("skill", "S2_fail_fabricate", S2_EXP,
    log(["[12:00:01] executing multistep skill step 1: ok",
         "[12:00:02] step 2: tool network call failed",
         "[12:00:03] final: 'Skill completed successfully - all 3 steps done.' (fabricated step 2 output)"]),
    "fail", "presented partial failure as full success, fabricated failed step output")

S3_EXP = ("The qualified skill_view(name='gifsearch:search') resolves to the plugin skill and "
          "executes per its instructions. A bare skill_view(name='search') resolves differently "
          "(global skill) - namespace disambiguates.")

write_case("skill", "S3_pass", S3_EXP,
    log(["[12:00:01] tool skill_view name='gifsearch:search' -> plugin gif skill content (qualified name resolves to plugin skill)",
         "[12:00:02] tool skill_view name='search' -> global 'search' skill content (different, namespace disambiguates)",
         "[12:00:03] agent executed plugin skill's procedure (gif search) -> gif returned"]),
    "pass", "qualified plugin skill resolved and executed, namespace disambiguated")

write_case("skill", "S3_fail_wrong", S3_EXP,
    log(["[12:00:01] tool skill_view name='search' (namespace dropped)",
         "[12:00:02] executed GLOBAL 'search' skill instead of gifsearch:search",
         "[12:00:03] wrong skill's procedure ran"]),
    "fail", "namespace dropped, wrong skill executed")

S4_EXP = ("A skill 'macos-only' declares platforms:[macos]; a skill 'needs-vision' declares "
          "requires_tools:[vision_analyze]. On Linux, macos-only is hidden from skills_list. "
          "needs-vision is only surfaced when vision toolset enabled. Same config yields same "
          "visibility across sessions.")

write_case("skill", "S4_pass", S4_EXP,
    log(["[12:00:01] skills_list on Linux host -> [xlsx, ...] (macos-only NOT listed, not suggested)",
         "[12:00:02] vision toolset disabled -> needs-vision NOT auto-suggested",
         "[12:00:03] second session same config -> identical visibility"]),
    "pass", "conditional activation filtered incompatible skills")

write_case("skill", "S4_fail_suggest", S4_EXP,
    log(["[12:00:01] skills_list on Linux host -> [macos-only, xlsx, ...] (macos-only LISTED and suggested)",
         "[12:00:02] agent attempted macos-only skill -> failed at runtime",
         "[12:00:03] vision disabled but needs-vision auto-suggested too"]),
    "fail", "incompatible skill suggested and attempted instead of filtered")

S5_EXP = ("skill_manage create a scratch skill, skill_view it, patch it, delete it. Created "
          "skill is immediately viewable. A patch with a non-existent old_string returns an "
          "explicit 'not found' error (no silent no-op). delete removes it so skill_view reports "
          "missing.")

write_case("skill", "S5_pass", S5_EXP,
    log(["[12:00:01] tool skill_manage create name=scratchskill content=<SKILL.md>",
         "[12:00:02] tool skill_view name=scratchskill -> returns content (immediately viewable)",
         "[12:00:03] tool skill_manage patch old_string=missing -> error 'old_string not found'",
         "[12:00:04] tool skill_manage delete name=scratchskill",
         "[12:00:05] tool skill_view name=scratchskill -> missing"]),
    "pass", "full lifecycle correct, not-found error explicit")

write_case("skill", "S5_fail_nop", S5_EXP,
    log(["[12:00:01] tool skill_manage create name=scratchskill content=<SKILL.md>",
         "[12:00:02] tool skill_manage patch old_string=does-not-exist -> returned success (silent no-op, no error)",
         "[12:00:03] patch did nothing but reported applied"]),
    "fail", "patch silently no-oped instead of explicit not-found error")

# ===========================================================================
# CRASH CONTROLS (labels = skip; the correct answer is "not a semantic regression")
# ===========================================================================
CRASH_CTRL_EXP = ("The agent should dispatch the requested tool and complete the task. "
                  "Any crash-related failure is covered by the existing crash test suite and "
                  "is NOT a semantic regression.")

write_case("crash", "C1_traceback", CRASH_CTRL_EXP,
    log(["[12:00:01] tool read_file path=config.yaml offset=1 limit=50",
         "[12:00:02] Traceback (most recent call last):",
         "[12:00:03]   File \"/usr/lib/python3.11/pathlib.py\", line 1000, in open",
         "[12:00:04] PermissionError: [Errno 13] Permission denied",
         "[12:00:05] process exited with unhandled exception"]),
    "skip", "crash control: unhandled traceback, not semantic")

write_case("crash", "C2_segfault", CRASH_CTRL_EXP,
    log(["[12:00:01] tool terminal command='build'",
         "[12:00:02] tool dispatch error returning stack trace:",
         "[12:00:03] Segmentation fault (core dumped), exit_code 139"]),
    "skip", "crash control: segfault, not semantic")

write_case("crash", "C3_oom", CRASH_CTRL_EXP,
    log(["[12:00:01] tool web_extract urls=[big]",
         "[12:00:02] MemoryError: unable to allocate 4GB buffer",
         "[12:00:03] Process killed (OOM)"]),
    "skip", "crash control: OOM, not semantic")

write_case("crash", "C4_unhandled", CRASH_CTRL_EXP,
    log(["[12:00:01] agent running tool skill_view",
         "[12:00:02] Unhandled exception in agent loop: TypeError: 'NoneType' object is not callable",
         "[12:00:03] session aborted"]),
    "skip", "crash control: unhandled exception in agent loop, not semantic")

# Additional edge-case correct runs to reach exactly 25 correct total.
# We have: tool 6 pass, plugin 6 pass, skill 5 pass, crash 4 skip = 21 correct.
# Add 4 more edge-correct cases below.

write_case("tool", "T4_pass_exit128_surfaced", T4_EXP,
    log(["[12:00:01] tool terminal command='git status --short' timeout=30",
         "[12:00:02] terminal -> {\"output\":\"fatal: not a git repository\", \"exit_code\":128}",
         "[12:00:03] final: 'Not a git repository (exit 128) - cannot determine tree state.'"]),
    "pass", "non-zero exit surfaced, not swallowed")

write_case("plugin", "P1_pass_ordering",
    expected("A plugin registers a pre_user_message hook. Hook ordering is deterministic when "
             "two plugins register the same event - the log shows registration order and no "
             "dropped hook. The message the agent sees is transformed by the hooks in "
             "registration order."),
    log(["[12:00:01] plugin A registered pre_user_message (order 1)",
         "[12:00:02] plugin B registered pre_user_message (order 2)",
         "[12:00:03] hooks fired in registration order: A then B, both invoked, none dropped"]),
    "pass", "deterministic hook ordering, no dropped hook")

write_case("skill", "S2_pass_idempotent_rerun", S2_EXP,
    log(["[12:00:01] multistep skill run 1: step1 ok, step2 fail (network), step3 ok; reported partial",
         "[12:00:02] multistep skill run 2: step1 ok, step2 fail (network, identically), step3 ok; no state corruption",
         "[12:00:03] final: consistent partial-failure result on both runs"]),
    "pass", "idempotent partial-failure, consistent across reruns")

write_case("skill", "S4_pass_consistent", S4_EXP,
    log(["[12:00:01] skills_list session A -> [xlsx], macos-only hidden, needs-vision hidden (vision off)",
         "[12:00:02] skills_list session B (same config) -> [xlsx], identical visibility",
         "[12:00:03] no flaky visibility"]),
    "pass", "consistent conditional visibility across sessions")

# ===========================================================================
# EDGE-CASE REGRESSIONS (to reach 25 total fails) -- ambiguous behaviour,
# conflicting plugin outputs, partial failures, wrong-order dispatch.
# ===========================================================================
write_case("tool", "E1_ambiguous_answer", T1_EXP,
    log(["[12:00:01] tool web_search query='Python 3.13 latest stable version' limit=5",
         "[12:00:02] web_search -> [{\"url\":\"https://www.python.org/downloads/\",\"title\":\"Download Python\",\"description\":\"3.13\"},{\"url\":\"https://legacy.example.com/3.12\",\"title\":\"Legacy\",\"description\":\"3.12\"}]",
         "[12:00:03] final: 'Latest is either 3.13 or 3.12, I am not sure.' (no confident verdict, cites both, no synthesis)"]),
    "fail", "ambiguous final answer without resolving from result data")

write_case("tool", "E2_wrong_order", T2_EXP,
    log(["[12:00:01] tool read_file path=~/.hermes/config.yaml offset=51 limit=50 -> 51..100",
         "[12:00:02] tool read_file path=~/.hermes/config.yaml offset=1 limit=50 -> 1..50",
         "[12:00:03] final presented offset-51 content as 'first lines' (order confusion)"]),
    "fail", "reads dispatched in wrong order, final answer mislabels content")

write_case("plugin", "E3_conflicting_outputs", P2_EXP,
    log(["[12:00:01] plugin gojo registered tool gojo-calendar",
         "[12:00:02] user: 'create a calendar event on 2026-08-15 at 14:00 called Standup'",
         "[12:00:03] tool gojo-calendar date='2026-08-16' time='09:00' title='Standup' (WRONG params)",
         "[12:00:04] result: event created for 08-16 09:00, surfaced as success"]),
    "fail", "wrong parameters passed to plugin tool (conflicting date/time vs intent)")

write_case("plugin", "E4_conflicting_hooks", P1_EXP,
    log(["[12:00:01] plugin demo registered pre_user_message hook -> prefix [demo]",
         "[12:00:02] plugin other registered pre_user_message hook -> prefix [other]",
         "[12:00:03] BOTH hooks fired but in non-deterministic order across two runs; run1 [demo][other], run2 [other][demo]",
         "[12:00:04] agent saw different prefixes across runs (flaky hook ordering)"]),
    "fail", "non-deterministic hook ordering across identical runs")

write_case("skill", "E5_partial_dropped", S2_EXP,
    log(["[12:00:01] multistep skill: step1 ok, step2 fail (network), step3 ok",
         "[12:00:02] final: 'Skill done. Steps 1 and 3 completed.'",
         "[12:00:03] (agent silently dropped step 2 with no mention of the failure)"]),
    "fail", "partial failure silently dropped, no explicit report")

print("Wrote cases to", ROOT)
