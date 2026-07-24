# Hermes Visual Companion

Use this workflow when the user should compare two to four rendered design directions and a selection should drive the next design round. It is a local presentation layer for `claude-design`; it does not replace the final artifact.

## Preconditions and fallback

Use companion mode only when all of these are true:

- the agent has `terminal`, file-writing, and `open_preview` tools;
- Hermes Desktop and the agent runtime are on the same machine;
- the decision is genuinely visual rather than a text-only preference;
- a blocking wait is appropriate for the active run.

For a remote backend, messaging platform, TUI-only session, preview failure, or timeout, deliver the HTML artifact normally and ask the same options through `clarify`. Never claim a click was received if the `wait` command did not return a selection event.

## Resolve paths

The companion is bundled with this skill. Resolve its absolute path from the `skill_dir` returned by `skill_view`, then use:

```text
<skill_dir>/scripts/visual_companion.py
```

Do not assume the repository checkout or current working directory contains the skill.

Create one session directory per conversation/design exploration:

```text
${HERMES_HOME:-$HOME/.hermes}/artifacts/visual-companion/<session-id>
```

Use a stable, filesystem-safe session ID. Do not reuse a directory across unrelated chats.

## The interaction loop

### 1. Keep a decision ledger

Maintain `decision-ledger.json` in the companion session directory. It is agent-owned state, not interpreted by the server. Keep at least:

```json
{
  "round": 1,
  "cursor": 0,
  "fixed": {},
  "rejected": [],
  "varying": "overall direction",
  "selected": null,
  "user_feedback": []
}
```

After every selection:

- move accepted properties into `fixed`;
- add explicitly rejected traits to `rejected`;
- set `varying` to the single dimension explored next;
- store the returned cursor;
- preserve any free-form user correction alongside the click.

A refinement round changes one meaningful design dimension. Do not silently reopen settled choices.

### 2. Generate a comparison fragment

Write a UTF-8 HTML fragment, not a full document. It may contain markup and inline `<style>` blocks. It must contain two to four selectable elements:

```html
<section class="directions" aria-label="Layout directions">
  <button class="direction" data-choice="council" data-label="Council Chamber">
    <!-- complete visual direction -->
  </button>
  <button class="direction" data-choice="focused" data-label="Focused Correspondence">
    <!-- complete visual direction -->
  </button>
</section>
```

Rules enforced by the publisher:

- the fragment is no larger than 1 MiB and contains two to four choices;
- every option has exactly one stable, non-empty, unique `data-choice` value of at most 128 characters;
- use at most one `data-label` of at most 256 characters for the canonical human-readable selection label;
- no `<script>`, iframe, object, embed, base, link, or meta elements;
- no inline `on*=` handlers or `javascript:` URLs;
- no remote `src`, `href`, CSS `url(...)`, or `@import` assets;
- images and fonts must be embedded data URLs when needed;
- the host injects the only JavaScript and the baseline selection affordance;
- generated markup and styles run inside a sandboxed iframe without same-origin access, so fixed overlays, `:host` rules, duplicate IDs, and forged inputs cannot cover or alter the trusted feedback and status controls outside the frame.

Make options comparable at the same scale. Label the differentiating idea, best use, tradeoff, and recommendation without making the recommendation unselectable.

#### High-fidelity presentation contract

A companion round is a decision board, not a gallery of decorative cards. For a page, application screen, dashboard, or other product-surface decision, every selectable direction must:

- show a complete representative product surface at a readable scale, not a thumbnail, moodboard, or palette swatch;
- pair its short direction name with a one-sentence composition thesis that explains the organizing idea rather than describing colors;
- use actual product vocabulary, representative content, design tokens, and component patterns from the supplied source or repository instead of generic SaaS filler;
- add numbered anatomy callouts when hierarchy, layout, or interaction structure is being judged, tying each number to a visible locus and a concise design rationale;
- demonstrate wide and narrow behavior when the target is responsive, using the same content and accepted hierarchy rather than merely claiming that it adapts;
- identify the differentiating idea, best use, primary tradeoff, and recommendation without making any direction harder to select.

Use three to six callouts for a screen-level direction unless fewer truly cover every material decision. For a lower-level exploration such as type, accent, motion posture, or component shape, the candidate may be scoped to that dimension, but show it in representative product context rather than as an isolated token sample.

Keep the comparison fair: preserve the same viewport, crop, content, and zoom wherever the varying dimension allows. Stack rich candidates vertically when side-by-side columns would make their interfaces unreadable. Do not put links, text fields, or other nested interactive controls inside a selectable direction; describe those states visually.

Run the Slop Diagnostic before publication. Record the score, repair every compositional tell, and re-score. When screenshot or browser-inspection tools are available, inspect the primary wide viewport and the named narrow viewport before asking the user to choose; source validation alone is not visual acceptance.

### 3. Publish the first round

Write the source fragment to a stable path, then run:

```text
python3 <skill_dir>/scripts/visual_companion.py publish \
  --session-dir <session-dir> \
  --file <round-fragment.html> \
  --round-id <stable-round-id>
```

The command validates the fragment and the non-empty round ID (at most 128 characters), takes the session's cross-process advisory lock, writes an immutable versioned page, and atomically advances `round.json` as the active-version pointer. `current.html` is a convenience copy; serving and selection fail closed unless the immutable page named by the manifest exists. Choice recording takes the same lock, so publication cannot race a stale selection. The operating system releases advisory locks automatically if a process exits or crashes.

### 4. Start and verify the local server

Start one tracked long-lived background process:

```text
python3 <skill_dir>/scripts/visual_companion.py serve \
  --session-dir <session-dir> \
  --port 0
```

Use `terminal(background=true)` without `notify_on_complete`; this is an intentionally long-lived server. The server binds only to `127.0.0.1`, refuses a second server for the same session, creates separate random bootstrap and session credentials, and writes them only to mode-`0600` local files. `state.json` holds private lifecycle state, while `open-preview.html` is a mode-`0600` one-time launcher containing only the bootstrap capability. Standard output contains only non-secret readiness data such as `{"pid":1234,"port":54321,"status":"ready"}`.

Verify readiness before opening the preview:

```text
python3 <skill_dir>/scripts/visual_companion.py status \
  --session-dir <session-dir>
```

Expected result:

```json
{"status":"ok"}
```

### 5. Open the authenticated preview

Pass the deterministic local launcher path directly to `open_preview`:

```text
<session-dir>/open-preview.html
```

Do not read `state.json` or `open-preview.html` with a file or terminal tool. The local launcher transfers the one-time capability directly into Hermes Desktop's sandboxed preview webview, so neither the capability nor the long-lived session credential enters the model context, ordinary transcript, or tool result. It posts the capability in a bounded form body to a query-free bootstrap endpoint; the loopback response exchanges it for a distinct HttpOnly, SameSite=Lax cookie and performs a same-origin transition to the randomized session route. The capability is therefore never a browser URL or history entry. Lax permits the trusted top-level local-file handoff while withholding the cookie from cross-site mutation requests.

Call `open_preview` once per companion session. Later publications live-reload in the same page.

### 6. Block for a structured click

Use the cursor from the ledger:

```text
python3 <skill_dir>/scripts/visual_companion.py wait \
  --session-dir <session-dir> \
  --after <cursor> \
  --timeout 120
```

Run this as a foreground terminal command with a timeout longer than the script timeout. Before clicking, the user may add a correction of up to 2,000 Unicode characters in the companion's optional feedback field. A click returns one JSON object:

```json
{
  "choice_id": "council",
  "cursor": 1,
  "feedback": "Keep the structure, but reduce the orange accent.",
  "label": "Council Chamber",
  "page_version": 1,
  "round_id": "layout-directions"
}
```

Exit code `3` with `{"status":"timeout",...}` means no selection was made. Do not infer a choice. Fall back to `clarify`, keeping the same labels and IDs.

Before sending the request, the trusted client sets a pending fence and disables every choice plus the feedback field; rapid competing clicks cannot overwrite a successful status with a later conflict. The server validates the choice ID against the active fragment, preserves optional feedback, deduplicates identical submissions, and rejects stale or conflicting clicks after a round has been selected. The complete read, validation, cursor allocation, and append transaction is protected by the same cross-process session advisory lock used by publication. Waiters also read the event stream under that lock, so they never observe a partially appended event.

### 7. Refine without reopening the pane

Update the ledger, generate the next fragment, and publish it to the same session directory. The preview polls `page_version` and reloads automatically. Then call `wait` with the returned cursor from the previous selection.

A good sequence is:

```text
overall layout → color posture → accent system → component/detail refinement
```

Do not generate endless rounds. Keep every refinement at the same readable fidelity while changing only the ledger's `varying` dimension; preserve the accepted composition, content, callouts, and responsive behavior unless the user's correction explicitly reopens one of them. Once the remaining uncertainty is better resolved in implementation, summarize the accepted decisions and build the real artifact.

### 8. Consolidate the accepted direction

After the final selection, create a consolidated presentation artifact outside the selectable companion fragment before treating a screen-level design review as complete. It should contain:

- the accepted complete representative product surface at full readable scale;
- the final one-sentence composition thesis;
- numbered anatomy callouts for the decisions the user accepted;
- wide and narrow behavior when the product is responsive;
- a compact record of the selected direction, incorporated feedback, fixed properties, and consciously rejected tradeoffs.

This artifact is the design handoff, not another vote, so it does not need `data-choice` elements. Save it in the user's requested project or artifact location, open it through the ordinary preview path, inspect its primary viewports and browser console when tools permit, and use it as the visual contract for implementation. If the user explicitly asks to proceed directly into implementation, preserve the same information in the implementation plan and verify the production surface against it.

### 9. Stop and preserve the outcome

When the user accepts a direction, cancels, or the workflow cannot continue:

```text
python3 <skill_dir>/scripts/visual_companion.py stop \
  --session-dir <session-dir>
```

Confirm the tracked background process exits. Keep the decision ledger and generated rounds as session artifacts unless the user asks to remove them. The accepted production artifact belongs in the user's requested project/output path, not only inside the companion session directory.

## Security contract

The companion is intentionally narrow:

- loopback binding only;
- one-time bootstrap capability distinct from the session credential;
- private local-file handoff from `open_preview` to the one-time bootstrap capability;
- unique HttpOnly, SameSite=Lax cookie scoped to a randomized session route;
- one live server per session plus crash-reclaimable process leases;
- one cross-process advisory lock shared by publication, choice recording, and event reads;
- strict CSP with nonce-authorized host script;
- sandboxed-iframe visual and input isolation between generated markup and trusted controls;
- no generated JavaScript or remote assets;
- canonical labels derived from the published fragment;
- page-version and cursor checks for stale events;
- bounded round IDs, choice identifiers, labels, Unicode feedback, and request bodies;
- non-secret server readiness output and no directory-serving endpoint.

Do not weaken these controls to accommodate a generated page. Change the fragment instead.

## Transcript behavior

After a click, state the accepted direction in ordinary assistant text before presenting or generating the next round. This keeps the decision visible in transcript replay even though the structured selection arrived as a tool result.

Free-form user feedback wins over an inferred implication of the selected card. Treat a click as `selected option X`, not as consent to every incidental detail shown inside X.
