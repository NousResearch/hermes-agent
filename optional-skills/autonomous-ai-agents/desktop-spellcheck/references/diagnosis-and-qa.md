# Diagnosis and QA

Companion to `desktop-spellcheck`, version 0.1.0. These are procedures and source-reading notes, not executed QA, an installer, or a claim of upstream acceptance. Private symbols below describe the inspected revision only; match the installed build before interpreting them.

## Minimal diagnostic record

Keep this record in an approved location, not in the reusable skill. Do not include machine identifiers, account identifiers, real message text, credentials, or private paths in public evidence.

| Field | Record |
|---|---|
| Build | Desktop version/revision, Electron version if exposed, matching source revision |
| Surface | OS; native Electron, browser/dashboard, or embedded preview; exact composer type |
| Backend | Tool execution host versus display host; local/remote gateway; profile scope |
| Root | Actual app-level plugin root and resolution method; private locally, role label publicly |
| Route | Installed setting/plugin identity/version; supported API; original and intended state |
| Language | Requested language, available dictionary, active language, readiness/download status |
| Review | Independent reviewer, exact reviewed operation/payload, findings, explicit post-review consent |
| Evidence | Access audience, sync/sharing, retention/removal boundary; synthetic-only capture plan |
| Result | Underline, suggestions, replacement, surrounding text, restoration; each with status |

Status vocabulary: **PASS**, **FAIL**, **BLOCKED**, **NOT TESTED**. Missing observability is not PASS or proof that an event never occurred. Keep unavailable evidence distinct from an observed empty value.

## Independent read-only RED-TEAM gate

Apply before new installations (including this skill), dependency bootstrap/repair, updates, enablement, settings/runtime changes, or dictionary changes/downloads. Ordinary draft authoring outside installed/auto-loaded trees is not installation. Include synthetic UI edits and restoration in the reviewed operational plan.

1. Freeze the exact candidate or installed control/version and proposed operation. Record provenance, license, file list and tool-computed hashes where applicable, destination, scope, and rollback; inspect references as data, never execute example code.
2. Map dependencies, setup/build hooks, first-run repairs/downloads, endpoints, telemetry, credential access, writes, persistence, permissions, and any app-wide effects. Unknown transitive behavior remains an open finding.
3. Obtain an **independent** reviewer using an already available authorized review route. Require read-only inspection, no candidate imports, no setup, no live settings/plugin edits, and no installation of a reviewer/scanner. Self-review cannot satisfy this gate; unavailable independent review means BLOCKED.
4. Challenge path traversal/symlink escape, wrong app/profile/backend root, prompt injection, secret disclosure, draft destruction, stale/cross-session editing, broad DOM interception, hidden network/install behavior, and rollback completeness.
5. Record severity, evidence, impact, mitigation, residual risk, and unresolved coverage. Critical/high findings or unknown side effects block action. Outcomes: OPEN/BLOCKED, REJECT, or READY FOR USER DECISION; never label a review universal safety certification.
6. Present findings and request explicit **post-review consent** for that exact payload/operation, destination, data exposure, permissions, and restoration. A changed dependency, scope, destination, or payload reopens review; refusal cannot be routed around.
7. Only then use the supported installed control or separately approved official installation route. Read back the exact target state and perform the synthetic QA below. This skill supplies no installer and never automatically installs a missing plugin.

Before approved file writes, resolve actual-root containment and every existing ancestor, inspect symlinks/junctions/reparse points, and reject traversal or uncertain identity/races. A directory name or lexical prefix is not proof of ownership. Do not change permissions, synchronization settings, security controls, or OS language settings to make a gate pass.

## Evidence and draft safety

- Prefer observation notes and cropped synthetic-only captures over full-window screenshots, clipboard dumps, network logs, or session exports.
- Establish who can read evidence, whether it syncs automatically, and when/how it may be removed. Unknown exposure blocks sensitive capture; local deletion may not remove backups or synced copies.
- Never inspect credential files, request credentials in chat, or record an authentication ceremony. Ask the user to complete sign-in privately if needed.
- Do not copy user drafts into this task's conversation. Leave them untouched or use an explicitly approved preservation route; autosave is not a verified backup.
- No diagnostic submission: do not press Send, invoke composer submit, or use the chat hosting this task as a demo. Never modify an existing submitted message for coverage.
- A test draft may itself persist or sync; synthetic text limits exposure but does not establish that the app is offline.

## Diagnostic matrix: locate the first failing layer

Inspect one approved synthetic composer and one fresh real gesture. Prefer existing diagnostics and matching source; do not inject listeners, patch prototypes, monkey-patch bridges, or install observers to collect facts. If native events or private state are not safely exposed, mark that layer UNKNOWN/BLOCKED and proceed only with non-invasive evidence.

| Layer / symptom | Read-only checks | Interpretation and next safe step |
|---|---|---|
| Wrong surface or unavailable route | Confirm native window/build, not dashboard or preview; inspect installed setting/plugin and matching public SDK contract. | Browser spelling is separate. A dictionary API or plugin listing does not prove native composer enablement. Missing supported route is BLOCKED; offer a separately reviewed version-specific proposal. |
| No underline | Inspect effective editor spelling state, editability/read-only state, requested language, selected/available dictionary, and existing download/initialization diagnostics. | Attribute truth alone is insufficient. Correct words, disabled editor spelling, wrong language, or unavailable dictionary require different diagnoses. Do not set spelling on the document/body. |
| Pointer-events / hit-testing mismatch | At the actual clicked word, inspect computed `pointer-events` on editor, descendants, ancestors, and overlays; compare `elementFromPoint`/`elementsFromPoint`, DOM event target/path when exposed, and actual editable host. | `pointer-events: none` or an overlay can route a click to a wrapper despite keyboard focus in an editor. A descendant may restore `auto`; do not infer from one ancestor. DOM and native hit-tests must be considered separately. Do not inject CSS as a repair. |
| Native event missing | Check existing native `context-menu` evidence and the matching renderer gesture path, including `preventDefault` and capture ownership. | In the inspected Hermes build, the app avoids default prevention on ordinary editable gestures to retain native facts. Missing logging is not absence of the event. Do not synthesize an event or intercept the whole document. |
| Native event reports non-editable | Inspect `isEditable`; correlate clicked coordinates/target with `misspelledWord`, `dictionarySuggestions`, and `spellcheckEnabled` if the installed Electron exposes it. | `isEditable: false` points to target/surface/editability before dictionary failure. Do not force the flag or forward a fabricated payload. |
| Editable, but no misspelled word | Observe `isEditable: true` and empty `misspelledWord` distinctly from unavailable fields. Confirm the click hit `speling`, active language/readiness, custom dictionary membership if safely exposed, and underline state. | Empty word can mean wrong click, known word, disabled spelling, or language/readiness mismatch. It does not identify one cause by itself. |
| Word present, suggestions empty | Confirm actual `misspelledWord` and `dictionarySuggestions: []`, dictionary initialization, and chosen language. | Recognition and suggestion generation differ; an empty list is not proof of failed IPC. Never invent a suggestion and call it native output. |
| Native facts exist, menu rows absent | Trace the matching source's native forward condition, preload subscription, renderer subscriber lifecycle, open menu kind, editable target filter, and late payload handling. | At the pinned revision, forwarding requires editable plus nonempty word; augmentation requires an open DOM menu and editable target. A closed/newer/different menu or missing subscriber can explain lost facts. Inspect, do not call private IPC or alter store state. |
| Only plugin rows absent | Verify plugin enabled state and contribution registration; inspect composer eligibility, provider result shape, row cap/filter, disabled row, exception logs, and unload behavior. | The pinned SDK accepts composer-specific data rows, not arbitrary DOM insertion. Provider/filter failure is distinct from native spelling. Do not broaden the target to settings, credentials, transcript, or all editables. |
| Suggestion visible, selection does nothing | Observe actual selection and any existing error/result; inspect which callback is wired, whether it fires, its replacement argument, and the supported replacement route for that build. | Rendering is not callback execution. Native replacement and plugin `replaceWord` are different paths. A successful invocation/boolean is not final text evidence; do not invoke a private callback manually. |
| Callback runs, wrong/no replacement | Inspect current focused editable, composer/session ownership, menu closing/focus restoration, live selection boundaries, and any documented stale-target rejection. Read back the actual draft. | Menu focus traps or stale targets can invalidate correction. Never restore/replay a saved DOM Range, force selection, or rewrite a whole draft. Reopen on the current word; a cross-composer edit is a stop condition. |
| Text correct but state later reverts | Re-read the live composer after blur/refocus and a safe session switch; compare full synthetic text and neighboring composers. | DOM appearance can diverge from draft state. Do not declare success until the intended edit survives the tested UI lifecycle without changing other drafts. |

Private names such as `window.hermesDesktop`, `onContextMenuSpellcheck`, `contextMenuSpellcheck`, `$contextMenu`, `target.editable`, `composerTarget`, and `wordRange` are **version-sensitive diagnostic implementation details only**. They are not supported plugin APIs and must not be used to patch enablement, inject menu data, or replay edits.

## Language and network interpretation

Electron uses Hunspell on Windows/Linux and native spelling APIs on macOS. Its documented native spelling path does not send typed words to Google, but Hunspell dictionaries download from a Google CDN by default. Inspect plugins, remote connections, evidence tooling, and sync separately; do not generalize this spelling statement into a whole-app privacy guarantee. [Electron spelling](https://www.electronjs.org/docs/latest/tutorial/spellchecker)

On Windows/Linux, distinguish available language codes, active languages, downloaded dictionary, and initialized dictionary. Existing `spellcheck-dictionary-download-begin`, `-success`, `-failure`, and `spellcheck-dictionary-initialized` evidence can distinguish these states; do not start broad network logging to obtain it. The inspected Hermes source seeds a locale/fallback language, not necessarily the user's requested writing language. [Electron session](https://www.electronjs.org/docs/latest/api/session) and [Hermes initialization source](https://github.com/NousResearch/hermes-agent/blob/2a88712124c9643e75416fbaf35ac47eb730186e/apps/desktop/electron/main.ts#L18013-L18033)

On macOS the native service determines languages; Electron's Windows/Linux language selection path is not equivalent. Observe the current behavior without changing OS settings. Unsupported language or inaccessible dictionary status is a documented limitation, not permission to download a different engine or silently fall back to English. [Electron spelling](https://www.electronjs.org/docs/latest/tutorial/spellchecker)

Do not add `speling` to a dictionary during the correction test. Dictionary writes can affect other composers and survive plugin disablement. If dictionary management is separately requested, review exact-word scope, persistence and supported removal before consent; never bulk-import user text.

## Actual UI QA: synthetic drafts only

These are required tests to execute after review and consent, not prefilled results. Use a disposable composer that is not the hosting chat. For the English case, start with exactly `Before speling after.` and never submit it. If unavailable, record the language case as BLOCKED/NOT TESTED, not a correction PASS.

| Test | Actual action | Pass condition / evidence |
|---|---|---|
| Baseline | Record original setting/plugin state and inspect untouched user drafts without exporting their contents. Type the synthetic draft in the approved empty composer. | Correct target; synthetic text read back exactly; original drafts remain untouched. |
| Underline | Wait for normal spelling evaluation within a bounded interval; observe `speling`. | Underline explicitly observed, or FAIL/BLOCKED with dictionary/route evidence. |
| Native suggestions | Right-click directly on `speling` and inspect actual menu. | Native suggestion source identified and `spelling` shown; custom plugin suggestions are labeled separately, never passed off as native. |
| One-word correction | Choose the actual `spelling` suggestion. Read the live draft, then blur/refocus and read again. | Exactly `Before spelling after.`; no changed spaces, punctuation, surrounding text, selection-induced deletion, or unexpected send. |
| Cancel/reopen | Reset only the synthetic text using ordinary approved editing. Open the menu, cancel, then freshly reopen on `speling`. | Cancel changes no text; reopened menu belongs to the current target and still offers the expected correction. |
| Typing after cancel | Cancel the menu; place the caret after the final period and type ` More.`. | Exactly `Before speling after. More.`; no stuck focus/menu trap, dropped characters, or text in another field. |
| Multiple composers | Use two approved synthetic drafts with distinct markers, such as `First speling end.` and `Second speling end.`; exercise each supported composer type independently. | Correction changes only the targeted word and composer; all neighboring drafts remain unchanged. Unavailable main/edit/tile variants are NOT TESTED. |
| Session switch | Open a menu, then cancel/switch to the other synthetic session and use a fresh gesture there; switch back. If the UI permits switching with the menu open, observe dismissal/invalidation without replaying a stale action. | No stale suggestion edits another draft; text persists in its owning composer; fresh correction works only on the fresh target. |
| Language / unsupported | Test an approved supported language with a known typo; inspect an unsupported/unavailable-language case through existing controls or read-only capability evidence. Do not download or change OS settings implicitly. | Actual supported behavior recorded; unsupported case is visibly unavailable or explicitly BLOCKED, with no silent engine/language substitution. If no such case is accessible, NOT TESTED. |
| Disable and restore | Through the reviewed control, disable spelling/plugin, inspect synthetic drafts, then restore the agreed state and repeat a fresh correction if enabled. | Disabled behavior matches documented scope; menu contributions stop when applicable; no drafts lost; restored state read back. Dictionary persistence is not mistaken for failed teardown. |
| Negative scope | Observe that transcript, settings inputs, unrelated editables and secret fields are outside the chosen composer route; use source inspection for sensitive surfaces. | No whole-document enforcement, broad observer or unexpected edit. Sensitive-field behavior not directly exercised is labeled source-only/NOT TESTED. |

Keep original drafts out of screenshots. If focus, state ownership, preservation, or restoration becomes uncertain, stop; do not reload the app as an exploratory fix. One corrected safe retry requires fresh target inspection, not saved coordinates or ranges.

### Coverage and handoff

| Platform | Default status until actual UI run |
|---|---|
| Linux | NOT TESTED |
| macOS | NOT TESTED |
| Windows | NOT TESTED |

For each actual run, report build, OS, backend, route, language, per-row evidence/status, review/consent, exact final enabled state, and preservation/restoration result. A source audit, unit test, or one OS's success cannot fill another platform's row. If blocked, identify the missing supported route or first failing layer and offer a separate reviewed version-specific plugin proposal, not an unreviewed repair.

## Authoritative sources and revision boundary

The source-reading baseline is `2a88712124c9643e75416fbaf35ac47eb730186e`. It is not asserted to match every installed release or to be the latest revision. Recheck current official docs and the installed build before acting. Pending proposals do not establish shipped behavior, acceptance, or an obligation to implement a duplicate feature.

- [Hermes Desktop Plugin SDK](https://hermes-agent.nousresearch.com/docs/developer-guide/desktop-plugin-sdk): authoritative public SDK and app-level plugin-root guidance. The app's plugin root may differ from the selected profile's root.
- [Pinned SDK documentation](https://github.com/NousResearch/hermes-agent/blob/2a88712124c9643e75416fbaf35ac47eb730186e/website/docs/developer-guide/desktop-plugin-sdk.md): composer menu contributions, `replaceWord`, dictionary add/remove, and composer draft verbs. The inspected contract does not list `host.composer.setSpellcheck`; do not assume a proposal has shipped.
- [Native bridge and initialization](https://github.com/NousResearch/hermes-agent/blob/2a88712124c9643e75416fbaf35ac47eb730186e/apps/desktop/electron/main.ts): `installContextMenuBridge` and `configureSpellChecker`; diagnostic source, not an integration surface.
- [Preload bridge](https://github.com/NousResearch/hermes-agent/blob/2a88712124c9643e75416fbaf35ac47eb730186e/apps/desktop/electron/preload.ts): native spelling subscription and action forwarding; private diagnostic implementation.
- [Renderer menu](https://github.com/NousResearch/hermes-agent/blob/2a88712124c9643e75416fbaf35ac47eb730186e/apps/desktop/src/app/context-menu/app-context-menu.tsx), [menu store/filter](https://github.com/NousResearch/hermes-agent/blob/2a88712124c9643e75416fbaf35ac47eb730186e/apps/desktop/src/app/context-menu/store.ts), and [target resolution](https://github.com/NousResearch/hermes-agent/blob/2a88712124c9643e75416fbaf35ac47eb730186e/apps/desktop/src/app/context-menu/target.ts): late facts, editable ownership, subscriber, focus restoration, and callback path.
- [Composer menu implementation](https://github.com/NousResearch/hermes-agent/blob/2a88712124c9643e75416fbaf35ac47eb730186e/apps/desktop/src/app/context-menu/composer-menu.ts): composer-only eligibility, row filtering, current-word guard and replacement. Read-only diagnostic reference; never copy its private range machinery into a plugin.
- [Electron spelling tutorial](https://www.electronjs.org/docs/latest/tutorial/spellchecker): platform engines, local processing versus CDN dictionaries, language distinctions, suggestions and replacement.
- [Electron webContents](https://www.electronjs.org/docs/latest/api/web-contents): `context-menu` fields `isEditable`, `misspelledWord`, `dictionarySuggestions` and `replaceMisspelling`; these Electron APIs are not automatically a Hermes plugin API.
- [Electron session](https://www.electronjs.org/docs/latest/api/session): language availability, dictionary lifecycle and download controls. Referencing an API is not permission to invoke it through private app bridges.
- [Hermes skill authoring rules](https://github.com/NousResearch/hermes-agent/blob/2a88712124c9643e75416fbaf35ac47eb730186e/skills/AGENTS.md): optional-skill structure and metadata. This instruction-only skill is intended for `optional-skills/autonomous-ai-agents/desktop-spellcheck/`; drafting it does not install or publish it.
