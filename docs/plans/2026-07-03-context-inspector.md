# Context Inspector (Desktop) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** A Hermes Desktop overlay that shows Alfredo everything in the live/active session's model context — System, User, Assistant, Tools — both as layered composition and as the ordered role-tagged message sequence, honestly labeled by fidelity.
**Problem:** Today the only window into context is the token-count breakdown popover. Alfredo cannot read the actual bytes the model sees (system tiers, tool schemas, transcript). This is the debugging + trust surface for "what does the model actually get".
**Architecture:** Add a sibling `compute_session_context_full` that returns raw *content* (not just counts) + ordered messages. For the ordered system message it reads the agent's **already-cached** system prompt bytes (never rebuilds mid-conversation); layered slices reuse `context_breakdown.py`'s decomposition, labeled as approximate. Exposed via a new read-only `session.context_full` RPC (mirrors `session.context_breakdown` lookup+lock). Desktop adds a full-screen overlay view, entered from the context-usage statusbar popover, lazy-fetched on click. No new core model tool, no new system-prompt surface, prompt cache untouched (copy-only reads).
**Tech Stack:** Python (agent core + tui_gateway RPC), React/TypeScript + Nanostores (Desktop renderer), vitest + pytest.

---

created:   2026-07-03T11:16Z
modified:  2026-07-03T12:30Z
commits:   -
decision:  APPROVED via plannotator 2026-07-03. Default UI = Variant B (Ordered transcript-first). Variant C folded as Advanced/Raw JSON tab. Honest "Reconstructed base" label, Exact last-sent deferred to Phase 4. Entry = context-usage statusbar popover. Copy JSON deferred (v1 has neither).
agents:    opus-4.8/dual-plan (planner-codex + planner-opus), opus-4.8/parent-reconcile, opus-4.8/dual-review-fold (reviewer-codex + reviewer-opus + reviewer-codex55)
sessions:  telegram-66984
back refs: docs/plans/*subagent-context-inspector* (prior art, security pattern)
fwd refs:  -

---

## SPEC

**Goal:** Local Desktop Context Inspector for the live/active session's model context across System / User / Assistant / Tools, without changing what the model receives, and with honest fidelity labels.

**Acceptance criteria (observable):**
1. New read-only `session.context_full` RPC returns raw **unredacted** context slices + an ordered role-tagged message list, reusing the `session.context_breakdown` session-lookup/`history_lock`-copy pattern. Resolves the **runtime/active** session id only (same value `ContextUsagePanel` receives).
2. The ordered **system** message is the agent's **already-cached** system prompt bytes (`_cached_system_prompt` + `ephemeral_system_prompt` when present) — NOT a fresh `build_system_prompt_parts` rebuild. Layered composition slices (system core / rules / skills / memory / tools) are provided as best-effort decomposition and each carries `source_accuracy` (`cached_exact` vs `reconstructed_current`).
3. Source is labeled **"Reconstructed base context (cached prefix + history; excludes per-turn ephemeral injections)"** — not "what the model receives next". Known omissions (memory-prefetch/plugin user-context injection, reasoning→`reasoning_content` conversion, prefill, provider/middleware rewrites, cache markers) are stated in the source-pill help text.
4. Message contract carries the full raw message JSON per turn (at minimum `role`, `content`, `tool_calls`, `tool_call_id`, `name`, `reasoning`) plus a display `content_text` and per-message token estimate — not just role+text.
5. Desktop overlay entered from the context-usage statusbar popover; lazy-fetches only on user action; presents both **Layered composition** and **Ordered transcript**.
6. Raw secret-looking strings stay VISIBLE (no redaction). Rendering is React text nodes / `<pre>{text}</pre>` ONLY — no `dangerouslySetInnerHTML`, no backend `html.escape` transformation (that double-escapes and corrupts the raw bytes). `<script>`/`<img onerror>` therefore create no DOM element.
7. No new core model tool, no new system-prompt surface; zero mutation of `_cached_system_prompt`, `api_messages`, `api_kwargs`, `agent.tools`, history, or role alternation (copy-only reads).
8. Raw context appears ONLY in the direct `session.context_full` response frame — never in any `event` frame (progress / roster / status / usage / session.info / tool.* / subagent.*) and never interpolated into an RPC error string.
9. No-agent / agent-not-built session returns an explicit `ContextFull`-shaped payload with `available:false` + `state:"agent_not_built"` and empty `slices`/`messages` — never `categories:[]` (sibling's shape), never a silent empty success mistaken for "no context".

**Validation method:**
- `python -m pytest tests/agent/test_context_full.py tests/agent/test_context_breakdown.py -q -o 'addopts='` (new full-context assembly + no regression of the token popover)
- `python -m pytest tests/agent/test_context_full_no_leak.py -q -o 'addopts='` (raw content only in RPC response frame; absent from every event frame AND from error envelopes)
- `python -m pytest tests/tui_gateway/test_session_context_full_rpc.py -q -o 'addopts='` (runtime-id success; stored-id/no-agent → available:false ContextFull shape; static error string)
- `python -m pytest tests/tui_gateway/test_context_full_ws_auth.py -q -o 'addopts='` (E2E: unauthenticated `/api/ws` cannot invoke `session.context_full`; authenticated/loopback can — security-boundary E2E, per AGENTS.md)
- `cd apps/desktop && npm run test:ui -- src/store/context-inspector.test.ts src/app/context/context-inspector.test.tsx src/app/routes.test.ts` (store passes runtime id + empty-state; overlay escaping; route wiring)
- `cd apps/desktop && npm run typecheck && npm run build`
- `cd apps/desktop && npx eslint <changed + untracked .ts/.tsx>`
- Manual smoke: open inspector on a live session → 4 buckets + ordered transcript render; planted `<script>` shows as text (no DOM node); planted fake secret stays visible; statusbar/roster/usage frames carry no raw content.

**Out-of-scope (v1):** editing/injecting context; per-turn history diffing; exact last-sent capture (Phase 4 follow-up, NOT v1); Copy JSON / export-to-file (deferred — clipboard is outside the local read boundary for unredacted secrets); mobile/TUI parity; i18n beyond adding the new string keys.

**Constraints + assumptions:**
- Prompt caching is sacred — reads are copy-only; the ordered system message is the cached bytes, not a rebuild. Never call `build_system_prompt_parts` when a cached prompt exists (use it only for approximate layered slices, clearly labeled).
- `.env` = secrets only; any toggle → `config.yaml`.
- Reconstruction uses the same char/4 estimate as `context_breakdown.py`; per-message token sum should reconcile with the aggregate the popover shows (note the intent, tolerate rounding).
- "Auth-gated" is precise only off-loopback: the default local Desktop bind is **loopback, unauthenticated locally**; auth engages on non-loopback bind. Shipping unredacted content over a non-loopback bind is an accepted, Alfredo-scoped risk. This is parity with the existing `session.context_breakdown`, but the blast radius is wider (raw secrets vs counts).

---

## Relevant Files / blast radius

```file-tree
{"title":"Touched files","entries":[
  {"path":"agent/context_breakdown.py","change":"modified","note":"add compute_session_context_full: cached system prompt bytes + approximate layered slices (source_accuracy) + full ordered messages + token estimates"},
  {"path":"tests/agent/test_context_full.py","change":"added","note":"RED: cached-prompt-not-rebuilt, full message fields, memory slice present, ordered roles, raw content preserved"},
  {"path":"tests/agent/test_context_full_no_leak.py","change":"added","note":"raw content only in RPC response frame; absent from event frames + error envelope"},
  {"path":"tui_gateway/server.py","change":"modified","note":"@method('session.context_full'): _sess_nowait+history_lock copy; no-agent→available:false ContextFull; STATIC error string (never interpolate exc)"},
  {"path":"tests/tui_gateway/test_session_context_full_rpc.py","change":"added","note":"runtime-id success; no-agent ContextFull shape; static error"},
  {"path":"tests/tui_gateway/test_context_full_ws_auth.py","change":"added","note":"E2E WS auth: unauth cannot call; auth/loopback can; raw only in response frame"},
  {"path":"apps/desktop/src/types/hermes.ts","change":"modified","note":"ContextFull, ContextSlice, ContextMessage types (full message fields + source_accuracy + available/state)"},
  {"path":"apps/desktop/src/store/context-inspector.ts","change":"added","note":"Nanostores: open state, source, data, activeBucket/tab; openContextInspector(runtimeSessionId, requestGateway) — INJECTED requestGateway, resolves stored→runtime via runtimeIdByStoredSessionIdRef before fetch"},
  {"path":"apps/desktop/src/store/context-inspector.test.ts","change":"added","note":"passes runtime id (not stored); empty-state on no live agent; one fetch per open"},
  {"path":"apps/desktop/src/app/routes.ts","change":"modified","note":"add 'context' AppView + AppRoute + OVERLAY_VIEWS entry"},
  {"path":"apps/desktop/src/app/routes.test.ts","change":"added","note":"appViewForPath + OVERLAY_VIEWS include 'context'"},
  {"path":"apps/desktop/src/app/shell/hooks/use-overlay-routing.ts","change":"modified","note":"CONTEXT_ROUTE, contextOpen, openContextInspector, close-to-previous"},
  {"path":"apps/desktop/src/app/context/context-inspector.tsx","change":"added","note":"overlay: rail buckets + Layered/Ordered tabs; React-text/<pre> render only; i18n strings; source pill + help text"},
  {"path":"apps/desktop/src/app/context/context-inspector.test.tsx","change":"added","note":"escaping (no img/script DOM node, secret visible); tab switch; large-payload loading/error"},
  {"path":"apps/desktop/src/app/shell/context-usage-panel.tsx","change":"modified","note":"add 'Inspect full context' action (passes activeSessionId + requestGateway); i18n key"},
  {"path":"apps/desktop/src/app/desktop-controller.tsx","change":"modified","note":"mount overlay view"},
  {"path":"apps/desktop/src/i18n/en.ts","change":"modified","note":"contextInspector string keys (title, tabs, buckets, source pill/help, empty/error states)"}
]}
```

---

## Data source decision (the key architecture call)

```tabs
{"tabs":[
  {"label":"A: Reconstructed base (v1)","body":"Ordered **system** message = the agent's already-cached `_cached_system_prompt` bytes (+ `ephemeral_system_prompt` if set); user/assistant/tool messages = copied session history with full fields. Layered slices = `context_breakdown.py` decomposition, labeled `reconstructed_current`.\n\n**Pros:** always available, no provider-call dependency, cache-safe (reads cached bytes, no rebuild).\n\n**Cons:** NOT the exact next request — omits memory-prefetch/plugin user-context injection, reasoning→`reasoning_content` conversion, prefill, provider/middleware rewrites, cache markers. Labeled honestly.\n\n**Verdict:** ship as v1 default with the honest label."},
  {"label":"B: Exact last-sent (Phase 4)","body":"Capture a serialization-safe copy of the FINAL `next_api_kwargs` INSIDE `_perform_api_call(next_api_kwargs)` — after `_build_api_kwargs`, Codex preflight, request + execution middleware, before the provider call — using `agent.subagent_context_artifacts.json_safe_copy` (NOT raw `copy.deepcopy`). Store out-of-band (artifact + session pointer).\n\n**Pros:** byte-exact ground truth incl. provider fields (`input`/Responses shape), cache markers, middleware rewrites.\n\n**Cons:** artifact lifecycle + cleanup (mirror subagent artifacts table/prune), large-payload cost, unavailable before first turn, wider privacy blast radius, must not perturb the live payload.\n\n**Verdict:** Phase 4 follow-up. Do NOT block v1 on it."}
]}
```

> [!DECISION]
> V1 ships **Reconstructed base** only, with a source pill + help text stating the omissions. The overlay is built source-aware from day one (a `source` field + switcher stub), so Phase 4 **Exact last-sent** slots in with no UI rework.

> [!RISK]
> **Phase-4 capture must not perturb the live request.** `_perform_api_call` receives the payload; `_sync_failover_system_message` mutates `api_messages[0]` in place on failover. Capture must (a) `json_safe_copy` the final `next_api_kwargs` at a single point, never re-referencing the live list, and (b) a byte-equality regression must assert the LIVE object identity + serialized bytes are unchanged after the capture call — not merely that the copy equals the original. Decide pre- vs post-failover capture explicitly.

> [!RISK]
> **Payload size.** Raw tool schemas + full copied conversation in one non-streaming RPC frame can be large. v1: add explicit loading/error states + a `truncated`/`byte_budget` metadata field; if a slice exceeds budget, mark it truncated with a "fetch full slice" affordance rather than freezing the renderer. No silent drop.

---

## Backend payload contract

```data-model
{"entities":[
  {"name":"ContextFull","fields":[
    {"name":"available","type":"bool","pk":false},
    {"name":"state","type":"'ready' | 'agent_not_built'","pk":false},
    {"name":"source","type":"'reconstructed_base' | 'exact_last_sent'","pk":false},
    {"name":"source_label","type":"string (honest fidelity label)","pk":false},
    {"name":"raw_unredacted","type":"bool (true)","pk":false},
    {"name":"model","type":"string","pk":false},
    {"name":"context_max","type":"int","pk":false},
    {"name":"context_used","type":"int","pk":false},
    {"name":"slices","type":"ContextSlice[]","pk":false},
    {"name":"messages","type":"ContextMessage[]","pk":false},
    {"name":"exact_capture_available","type":"bool","pk":false}
  ]},
  {"name":"ContextSlice","fields":[
    {"name":"id","type":"'system_prompt'|'rules'|'skills'|'memory'|'tool_definitions'|'mcp'|'subagent_definitions'|'conversation'","pk":true},
    {"name":"label","type":"string","pk":false},
    {"name":"bucket","type":"'system'|'tools'|'conversation'","pk":false},
    {"name":"content_text","type":"string (raw, unredacted)","pk":false},
    {"name":"source_accuracy","type":"'cached_exact' | 'reconstructed_current'","pk":false},
    {"name":"tokens","type":"int (char/4 estimate)","pk":false},
    {"name":"truncated","type":"bool","pk":false}
  ]},
  {"name":"ContextMessage","fields":[
    {"name":"index","type":"int","pk":true},
    {"name":"role","type":"'system'|'user'|'assistant'|'tool'","pk":false},
    {"name":"content_text","type":"string (raw display)","pk":false},
    {"name":"raw","type":"object (full msg JSON: content, tool_calls, tool_call_id, name, reasoning)","pk":false},
    {"name":"tokens","type":"int","pk":false}
  ]}
],"relations":[
  {"from":"ContextFull","to":"ContextSlice","kind":"one-to-many","label":"layered composition"},
  {"from":"ContextFull","to":"ContextMessage","kind":"one-to-many","label":"ordered transcript"}
]}
```

---

## UX framing — mock variants (pick one before build)

Open UX taste, so three real screen options below. All share the same header (honest source pill + token bar) and the four buckets Alfredo named. **Copy JSON removed from v1** (unredacted-secret clipboard footgun — deferred). Pick density/flow in the form at the bottom.

### Variant A — Layered + Ordered split (recommended)

Left rail = the 4 buckets. Main = two tabs: **Layered composition** (slices with token badges + accuracy chip) and **Ordered transcript** (the literal role-tagged sequence). Matches Alfredo's four mental buckets while keeping the engineering truth (system is cached bytes, layered slices are approximate, messages are ordered).

```wireframe
surface: desktop
<div class="wf-col" style="gap:0">
  <div class="wf-row" style="justify-content:space-between;padding:10px 14px;border-bottom:1px solid #2a2a2a">
    <div class="wf-row" style="gap:10px"><b>Context Inspector</b><span class="wf-pill accent" title="cached prefix + history; excludes per-turn ephemeral injections">Reconstructed base ⓘ</span></div>
    <div class="wf-row" style="gap:8px"><span class="wf-pill">124k / 200k</span><button>Close</button></div>
  </div>
  <div class="wf-row" style="align-items:stretch;gap:0">
    <div class="wf-col" style="width:150px;border-right:1px solid #2a2a2a;padding:8px;gap:2px">
      <div class="wf-card" style="padding:6px 8px"><b>System</b> <span class="wf-pill">41k</span></div>
      <div class="wf-card" style="padding:6px 8px">User <span class="wf-pill">12k</span></div>
      <div class="wf-card" style="padding:6px 8px">Assistant <span class="wf-pill">58k</span></div>
      <div class="wf-card" style="padding:6px 8px">Tools <span class="wf-pill">13k</span></div>
    </div>
    <div class="wf-col" style="flex:1;padding:10px;gap:8px">
      <div class="wf-row" style="gap:6px;border-bottom:1px solid #2a2a2a;padding-bottom:6px">
        <b style="border-bottom:2px solid #6ea8fe">Layered composition</b><span>Ordered transcript</span>
      </div>
      <div class="wf-card wf-row" style="justify-content:space-between"><span>Core system prompt <span class="wf-pill">cached exact</span></span><span class="wf-pill">18k</span></div>
      <div class="wf-card wf-row" style="justify-content:space-between"><span>Rules / AGENTS context <span class="wf-pill">approx</span></span><span class="wf-pill">9k</span></div>
      <div class="wf-card wf-row" style="justify-content:space-between"><span>Skills index <span class="wf-pill">approx</span></span><span class="wf-pill">6k</span></div>
      <div class="wf-card wf-row" style="justify-content:space-between"><span>Memory + profile <span class="wf-pill">approx</span></span><span class="wf-pill">8k</span></div>
      <div class="wf-card" style="font-family:monospace;font-size:11px;color:#aaa">You are a blunt, practical Hermes Agent operating for Alfredo. Be concise...</div>
    </div>
  </div>
</div>
```

### Variant B — Ordered transcript-first (single stream)

One scrollable role-tagged stream, top to bottom, as the model reads it. System block collapsible at top; each turn a labeled card. Bucket chips act as filters. Closest to "read the raw feed".

```wireframe
surface: desktop
<div class="wf-col" style="gap:0">
  <div class="wf-row" style="justify-content:space-between;padding:10px 14px;border-bottom:1px solid #2a2a2a">
    <div class="wf-row" style="gap:10px"><b>Context Inspector</b><span class="wf-pill accent">Reconstructed base ⓘ</span></div>
    <div class="wf-row" style="gap:8px"><span class="wf-pill">124k / 200k</span><button>Close</button></div>
  </div>
  <div class="wf-row" style="gap:6px;padding:8px 14px;border-bottom:1px solid #2a2a2a">
    <span class="wf-pill accent">All</span><span class="wf-pill">System</span><span class="wf-pill">User</span><span class="wf-pill">Assistant</span><span class="wf-pill">Tools</span>
  </div>
  <div class="wf-col" style="padding:10px;gap:8px">
    <div class="wf-card" style="border-left:3px solid #b57edc"><div class="wf-row" style="justify-content:space-between"><b>SYSTEM</b><span class="wf-pill">41k · cached · collapse</span></div><span style="font-family:monospace;font-size:11px;color:#aaa">You are a blunt, practical Hermes Agent... [+ rules + skills + memory + tools]</span></div>
    <div class="wf-card" style="border-left:3px solid #6ea8fe"><div class="wf-row" style="justify-content:space-between"><b>USER</b><span class="wf-pill">0.3k</span></div><span style="font-size:12px">add a customization in the Hermes desktop app...</span></div>
    <div class="wf-card" style="border-left:3px solid #63c58e"><div class="wf-row" style="justify-content:space-between"><b>ASSISTANT</b><span class="wf-pill">2k · reasoning · tool_calls</span></div><span style="font-size:12px">Loading grounding refs + scoping the repo...</span></div>
    <div class="wf-card" style="border-left:3px solid #e0a458"><div class="wf-row" style="justify-content:space-between"><b>TOOL · read_file · tool_call_id</b><span class="wf-pill">1.1k</span></div><span style="font-family:monospace;font-size:11px;color:#aaa">{"content":"1|import ..."}</span></div>
  </div>
</div>
```

### Variant C — Raw JSON explorer (advanced)

Two-pane debug view: left = tree/TOC of slices+messages, right = raw pretty-printed payload for the selected node. Strong for byte-level debugging + Phase-4 exact-capture comparison; weaker for casual browsing. Best folded in as an "Advanced / Raw JSON" tab inside A, not the default.

```wireframe
surface: desktop
<div class="wf-col" style="gap:0">
  <div class="wf-row" style="justify-content:space-between;padding:10px 14px;border-bottom:1px solid #2a2a2a">
    <div class="wf-row" style="gap:10px"><b>Context Inspector</b><span class="wf-pill">Reconstructed base</span><span class="wf-pill" style="opacity:.5">Exact last-sent (Phase 4)</span></div>
    <div class="wf-row" style="gap:8px"><button>Close</button></div>
  </div>
  <div class="wf-row" style="align-items:stretch;gap:0">
    <div class="wf-col" style="width:200px;border-right:1px solid #2a2a2a;padding:8px;font-family:monospace;font-size:11px;gap:3px">
      <span>▼ slices</span><span style="padding-left:12px">system_prompt · cached</span><span style="padding-left:12px;color:#6ea8fe">tool_definitions · approx</span><span style="padding-left:12px">conversation</span>
      <span>▼ messages [42]</span><span style="padding-left:12px">0 system</span><span style="padding-left:12px">1 user</span><span style="padding-left:12px">2 assistant</span>
    </div>
    <div class="wf-col" style="flex:1;padding:10px">
      <div class="wf-card" style="font-family:monospace;font-size:11px;color:#aaa;white-space:pre">{
  "id": "tool_definitions",
  "source_accuracy": "reconstructed_current",
  "tokens": 13120,
  "content_text": "[{\"name\":\"terminal\",...}]"
}</div>
    </div>
  </div>
</div>
```

```question-form
{"title":"Decisions before build","questions":[
  {"id":"variant","prompt":"Which UI ships as the default?","type":"choice","options":["A Layered+Ordered split","B Ordered transcript-first","C Raw JSON explorer"],"default":"A Layered+Ordered split"},
  {"id":"c_as_tab","prompt":"Fold Variant C in as an 'Advanced / Raw JSON' tab inside the chosen default?","type":"bool","default":"true"},
  {"id":"source_label","prompt":"OK with the honest 'Reconstructed base (excludes per-turn injections)' label for v1, with Exact last-sent deferred to Phase 4?","type":"bool","default":"true"},
  {"id":"entry_point","prompt":"Entry point to open the overlay?","type":"choice","options":["From context-usage statusbar popover","Titlebar action","Both"],"default":"From context-usage statusbar popover"},
  {"id":"copy_export","prompt":"Copy JSON / export (unredacted secrets) — keep deferred or add with a warning?","type":"choice","options":["Deferred (v1 has neither)","Add Copy JSON behind explicit warning+confirm","Add copy + file export behind warning"],"default":"Deferred (v1 has neither)"},
  {"id":"notes","prompt":"Any tweak to the chosen variant?","type":"text"}
]}
```

---

## Phase 1 — Backend: full-context assembly (reconstructed base)

### Task 1.1: RED — cached-prompt + full-message + memory-slice tests
**Objective:** Lock the observable JSON shape AND the cache-safety invariant of `compute_session_context_full` before implementation.
**Files:** Create `tests/agent/test_context_full.py`
**Steps (write failing tests):**
- Configure a fake `_memory_store` so the `memory` slice is real (don't rely on `volatile` string containing `MEMORY_BLOCK`).
- Set `agent._cached_system_prompt = "CACHED SYS BYTES"`. Assert the ordered `messages[0]` (role `system`) `content_text == "CACHED SYS BYTES"`.
- Patch `agent.system_prompt.build_system_prompt_parts` with a `MagicMock`; after `compute_session_context_full`, assert it was **NOT called** when a cached prompt exists (or called only for approximate layered slices, never for the ordered system message).
- Assert slice ids `>= {"system_prompt","rules","skills","memory","tool_definitions","mcp","subagent_definitions","conversation"}` (memory INCLUDED).
- Assert every message carries `raw` with `tool_calls`/`tool_call_id`/`name` preserved on an assistant-tool-call + tool-result fixture; assistant `reasoning` preserved.
- Assert `[m["role"] for m in messages] == ["system","user","assistant","tool"]`; raw `<script>alert(1)</script>` and `SECRET_TOKEN=abc123` present verbatim in `content_text`.
- Assert `source == "reconstructed_base"`, `raw_unredacted is True`, each slice has `source_accuracy in {"cached_exact","reconstructed_current"}`.
**Run RED** — `python -m pytest tests/agent/test_context_full.py -q -o 'addopts='` → FAIL (import missing).
**Stop.** No production code this task.

### Task 1.2: GREEN — implement `compute_session_context_full`
**Objective:** Build the raw-content sibling; ordered system message = cached bytes; layered slices = approximate.
**Files:** Modify `agent/context_breakdown.py`
**Steps:**
- Ordered system message: read `getattr(agent, "_cached_system_prompt", None)`; if present, use it verbatim (+ `agent.ephemeral_system_prompt` when set, matching the conversation-loop prepend). Only if no cached prompt exists (cold) fall back to `build_system_prompt_parts` and mark `source_accuracy="reconstructed_current"`.
- Layered slices: reuse `_SKILLS_BLOCK_RE`, `_memory_blocks`, `_strip_blocks`, `_split_tools`, `_chars_to_tokens`, `_json_tokens` from the existing module; tag layered slices `reconstructed_current` (they may not sum to the cached bytes after filesystem/memory drift — that's expected and labeled).
- Messages: map copied history to `{index, role, content_text, raw, tokens}` — `raw` is the full copied message dict (copy-only; never mutate history or `agent.tools`).
- Per-message tokens via `_chars_to_tokens`/`_json_tokens`; note reconciliation intent with the aggregate.
- `source="reconstructed_base"`, `source_label="Reconstructed base context (cached prefix + history; excludes per-turn ephemeral injections)"`, `raw_unredacted=True`, `available=True`, `state="ready"`, `exact_capture_available=False`.
- Do NOT `html.escape` — return raw strings; escaping is the renderer's job.
**Run GREEN** — same command → PASS.
**Commit** — `git commit -m "feat(agent): add compute_session_context_full raw-content assembly (cached prompt + approx layers)"`

### Task 1.3: RED+GREEN — no-leak regression (events + error envelope)
**Objective:** Prove raw content is isolated to the RPC response frame.
**Files:** Create `tests/agent/test_context_full_no_leak.py`
**Steps:** Integration-style around the emit/transport seam: plant a secret + `<script>` in the fixture; call the full-context path and capture `_emit`/event frames (`progress`, `roster`, `status`, `usage`, `session.info`, `tool.*`, `subagent.*`). Assert the planted strings appear ONLY in the direct response, never in any event frame. Add a case where assembly raises and assert the error envelope carries a STATIC string, not the exception content. RED first against a deliberately-leaky stub, then GREEN with the real static-error handler.
**Phase 1 gate:**
- [ ] `python -m pytest tests/agent/test_context_full.py tests/agent/test_context_full_no_leak.py tests/agent/test_context_breakdown.py -q -o 'addopts='` — assembly + no-leak + no popover regression

---

## Phase 2 — RPC: `session.context_full`

### Task 2.1: RED — RPC contract + no-agent shape + static error
**Objective:** Lock RPC behavior before wiring.
**Files:** Create `tests/tui_gateway/test_session_context_full_rpc.py`
**Steps:** Live-agent runtime-id session returns full `ContextFull`. No-agent (or agent-not-built) session returns `available:False, state:"agent_not_built"`, empty `slices`/`messages`, maxes from usage — **ContextFull shape, not `categories:[]`**. Assembly-raises path returns a static error envelope (assert no raw content in the error string). Passing a stored (non-runtime) id resolves to no `_sessions` entry → the same `available:false` state (documents the runtime-id-only contract).
**Run RED** → FAIL.

### Task 2.2: GREEN — add `@method("session.context_full")`
**Objective:** Wire the RPC like `session.context_breakdown`, with the corrected error + no-agent contract.
**Files:** Modify `tui_gateway/server.py` (beside L6213)
**Steps:** `_sess_nowait(params, rid)` → `agent = session.get("agent")` → if `None`: return `available:false` ContextFull. Else under `session["history_lock"]` copy history → `compute_session_context_full(agent, history)`. Wrap in try/except that returns a **static** error string (never `f"...{exc}"`), logging the detail server-side only if that log path is itself content-free.
**Run GREEN** → PASS.
**Commit** — `git commit -m "feat(gateway): add session.context_full RPC (static error, available:false no-agent shape)"`

### Task 2.3: RED+GREEN — E2E WS auth boundary
**Objective:** Prove the raw-secret RPC is reachable only through the gated path (AGENTS.md security-boundary E2E requirement).
**Files:** Create `tests/tui_gateway/test_context_full_ws_auth.py`
**Steps:** Drive the real `/api/ws` boundary (`_ws_auth_ok`): unauthenticated connection cannot invoke `session.context_full`; authenticated/loopback can; raw content appears only in the response frame, never an `event` frame. Document the loopback-unauthed-local reality in a comment + assert the non-loopback path requires auth.
**Phase 2 gate:**
- [ ] `python -m pytest tests/tui_gateway/test_session_context_full_rpc.py tests/tui_gateway/test_context_full_ws_auth.py -q -o 'addopts='`

---

## Phase 3 — Desktop: overlay view, store, wiring

### Task 3.1: RED — store test (runtime id + empty-state)
**Objective:** Lock the store fetch/resolve contract.
**Files:** Create `apps/desktop/src/store/context-inspector.ts` + `.test.ts`; modify `apps/desktop/src/types/hermes.ts`
**Steps:** Add `ContextFull`/`ContextSlice`/`ContextMessage` types (full fields + `source_accuracy` + `available`/`state`). Store: `$contextInspectorOpen`, `$contextSource`, `$contextData`, `$activeBucket`, `$activeTab`, and `openContextInspector(sessionId, requestGateway)` — **`requestGateway` is injected** (hermes.ts cannot call the hook-local callback). If the passed id is a stored lineage id, resolve via `runtimeIdByStoredSessionIdRef` to the runtime id BEFORE calling `requestGateway('session.context_full', {session_id: runtimeId})`. RED tests: given `$selectedStoredSessionId='stored-1'` and map `stored-1 → runtime-1`, the RPC is called with `runtime-1`; a session with no live agent renders the empty/`agent_not_built` state; opening triggers exactly one fetch; error → inline failure state.
**Run RED** — `cd apps/desktop && npm run test:ui -- src/store/context-inspector.test.ts` → FAIL.

### Task 3.2: GREEN — store + injected bridge
**Files:** Implement the store. Do NOT add a misleading `getSessionContextFull` wrapper to `hermes.ts` (that file is REST/`window.hermesDesktop.api`; `requestGateway` is a hook callback). The overlay/statusbar passes `requestGateway` into `openContextInspector`.
**Run GREEN** → PASS.

### Task 3.3: RED — overlay component test (escaping headline)
**Objective:** Prove React-text render + tab switch + large-payload handling.
**Files:** Create `apps/desktop/src/app/context/context-inspector.test.tsx`
**Steps:** Mount with a payload whose `content_text` includes `<img src=x onerror=alert(1)>` and `SECRET_TOKEN=abc`. Assert the string renders inside a `<pre>` (text node), `document.querySelector('img')` and `('script')` are null, and the secret string IS present (not redacted, not `&lt;`-escaped in the visible text). Assert bucket rail + Layered/Ordered tabs switch. Assert a `truncated` slice shows the truncation affordance, and a loading/error state renders. Reuse the shipped `SubagentContextInspector` `<pre>` primitive/test style.
**Run RED** → FAIL.

### Task 3.4: GREEN — overlay + route + entry point + i18n
**Files:** Create `apps/desktop/src/app/context/context-inspector.tsx`; modify `apps/desktop/src/app/routes.ts` (add `'context'` to `AppView`/`AppRouteId`/`APP_ROUTES`/`OVERLAY_VIEWS`) + create `routes.test.ts`; modify `apps/desktop/src/app/shell/hooks/use-overlay-routing.ts` (`CONTEXT_ROUTE`, `contextOpen`, opener, close-to-previous); modify `apps/desktop/src/app/desktop-controller.tsx` (mount overlay); modify `apps/desktop/src/app/shell/context-usage-panel.tsx` (add "Inspect full context" action passing `activeSessionId` + `requestGateway`); modify `apps/desktop/src/i18n/en.ts` (string keys).
**Steps:** Render chosen variant (default A). ALL content via React text / `<pre>{text}</pre>`, never `dangerouslySetInnerHTML`. Source pill reads `data.source_label` with the omissions in help/tooltip. Reuse `formatK` for token badges. No Copy JSON in v1.
**Run GREEN** — `cd apps/desktop && npm run test:ui -- src/app/context/context-inspector.test.tsx src/app/routes.test.ts` → PASS.
**Commit** — `git commit -m "feat(desktop): add Context Inspector overlay (runtime-id, react-text render, honest source label)"`
**Phase 3 gate:**
- [ ] `cd apps/desktop && npm run test:ui -- src/store/context-inspector.test.ts src/app/context/context-inspector.test.tsx src/app/routes.test.ts`
- [ ] `cd apps/desktop && npm run typecheck`
- [ ] `npx eslint` on changed + untracked `.ts/.tsx` (include new store/component/route/i18n files; use `git ls-files --others --exclude-standard`)
- [ ] `cd apps/desktop && npm run build`

---

## Phase 4 (optional / follow-up) — Exact last-sent capture

Gated on the `copy_export`/exact-capture decision. Capture a `json_safe_copy` of the FINAL `next_api_kwargs` INSIDE `_perform_api_call(next_api_kwargs)` (after middleware, before provider send), deriving ordered messages from `next_api_kwargs["messages"]` or `["input"]` per api mode. Store out-of-band in a dedicated main-session artifact table + pointer (mirror `agent/subagent_context_artifacts.py` schema + delete/prune helpers; wire every `DELETE FROM sessions` call site + compression lineage). Expose as `source='exact_last_sent'` with an empty-state until first capture. Byte-equality regression asserts the LIVE payload object identity + serialized bytes are unmutated by capture; test middleware-modified payloads, Codex/Responses `input`, tools, non-serializable values, failover `api_messages[0]` rewrite, and timeout/failure paths.

---

## Validation loops (global)

- [ ] `python -m pytest tests/agent/test_context_full.py tests/agent/test_context_full_no_leak.py tests/agent/test_context_breakdown.py tests/tui_gateway/test_session_context_full_rpc.py tests/tui_gateway/test_context_full_ws_auth.py -q -o 'addopts='` — backend end to end incl. no-leak + WS auth
- [ ] `cd apps/desktop && npm run test:ui -- src/store/context-inspector.test.ts src/app/context/context-inspector.test.tsx src/app/routes.test.ts` — store + overlay + routes
- [ ] `cd apps/desktop && npm run typecheck && npm run build` — types + bundle
- [ ] `cd apps/desktop && npx eslint <changed + untracked .ts/.tsx>`
- [ ] `git -C . diff --check` — whitespace
- [ ] Review checklist: core tool registry/schema count unchanged; `agent/system_prompt.py` unmodified (no new system-prompt surface).
- [ ] Manual smoke: open inspector on a live session → 4 buckets + ordered transcript render; planted `<script>` shows as text (no DOM node); planted fake secret stays visible; statusbar/roster/usage frames carry no raw content; changing a context file mid-session does NOT change the shown system message (proves cached bytes, not rebuild).

## Risks

> [!RISK]
> **Cache safety.** Reads are copy-only and the ordered system message is the cached bytes; `build_system_prompt_parts` is NOT called when a cached prompt exists. Add an assertion that `_cached_system_prompt`, `agent.tools`, and history identity are unchanged after the call.

> [!RISK]
> **Privacy blast radius.** Unredacted-by-design (Alfredo's explicit ask). Mitigation: loopback-only local read (auth only off-loopback — labeled precisely, not called "auth-gated" as if that's the protection), React-text render, no-leak tests (events + error envelope), no Copy JSON in v1, never in model context or side frames.

> [!ASSUMPTION]
> The inspector targets the **runtime/active** session id — the same value `ContextUsagePanel` already receives (`activeSessionId`). Stored lineage ids are resolved to runtime via `runtimeIdByStoredSessionIdRef` before the RPC; an unresolvable/expired runtime id shows the `agent_not_built`/empty state.
