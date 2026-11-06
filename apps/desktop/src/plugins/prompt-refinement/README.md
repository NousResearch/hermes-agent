# Prompt Refinement (Desktop, bundled plugin)

Turns a draft into a deeper version of the same task before it is sent. Before
rewriting, the refiner works out what a top professional in the field would do
that the user didn't ask for, the rawest material they'd start from, the
specific methods and tools they'd use, and what outside evidence would show the
result is right; then it rewrites the prompt to ask for exactly that (same goal
and scope, nothing specific dropped). Ships **Off**.

**Depth** decides how far a rewrite goes: **Polish** fixes wording and
structure and adds nothing; **Clarify** spells out what the draft implies (goal,
requirements, expected output, assumptions to confirm), at most about twice as
long; **Deepen** is the four-question professional rewrite, scaled to the task.
**Auto** (default) lets the refiner pick the lightest depth that fits each
draft and report it (`"depth"` in its answer); the drawer shows which it chose.
A fixed depth is one switch away (settings slider, or the composer pill). Every
depth keeps the user's specifics exactly and rewrites the plain wording rather
than copying it.

| Mode           | What happens                                                                                                                                                                                                      |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Off            | Sends pass straight through.                                                                                                                                                                                      |
| Approval-First | Enter refines the draft and opens the review drawer. **Nothing is sent until the user decides**: Approve and send (the refined or edited text), Send original, or Cancel (the draft goes back into the composer). |
| Automatic      | Enter refines and **sends without a click**. A slim drawer shows progress (Send original / Cancel). It stops for the user when refinement fails or the preservation or intent check flags anything.               |

Approval-First is stored as `manual` (its earlier name), so saved preferences
carry over unchanged; only the label changed.

Both modes skip slash commands, queued / voice / hidden / external sends, and
prompts under the word threshold. Enter is the only trigger: there is no
separate refine button or command, so a draft is never refined twice.

The review is a drawer that slides up out of the typing box (the composer's own
drawer primitive, `ComposerDrawer`, in `COMPOSER_AREAS.top`), opening only in
the composer the prompt came from. It shows the refined prompt, a collapsed
"What was added and why" with the four answers, the preservation result, and
Cancel · Send original · Edit · Approve and send. Esc cancels, ⌘/Ctrl+Enter
approves (focus inside the drawer only).

## How a send flows

```
Enter ─► useMiddlewareSubmit (stamps origin) ─► middleware.ts (failClosed)
           ├─ not eligible ───────────────► normal submit, unchanged
           └─ eligible ─► hold composer ─► review-store ─► engine ─► host-bridge ─► llm.oneshot
                 Automatic, clean ───────► refined text + the same attachments array, sent
                 Approval-First, or flagged/failed ► drawer: Approve and send / Edit / Send original / Cancel
                 Cancel / any failure ───► null ─► composer restores draft + attachments
```

`llm.oneshot` is stateless: no session row, no history, no tools, no skills, no
memory. It is routed to the backend that owns the chat (its profile scope).
Model: the `prompt_refinement` auxiliary slot, or the chat's own model.

The Python half (`plugins/prompt-refinement/`, opt-in per home:
`hermes plugins enable prompt-refinement`) registers the slot so a model can be
picked for it under Settings ▸ Model ▸ Auxiliary. Without it the call still runs,
on the default auxiliary route. A backend that refuses an unregistered task
(`invalid_params` with `field: 'task'`, or the sidecar's 4031) fails closed, and
the drawer says how to turn the plugin on.

## Guarantees and where they live

- **Only a clean or approved refinement is sent**: `review-store.ts` is the only
  producer of an `approve` decision — automatically only when nothing is
  flagged, otherwise from the user — and re-checks `preservation.ts` itself.
  The middleware is registered `failClosed` and also catches its own errors.
- **Exact content survives**: `preservation.ts` blocks Approve while code,
  `@` references, URLs, paths, identifiers or numbers are missing; heuristic
  losses (negations, requirement words, hyphenated names) need an explicit check.
- **No unrequested actions, as far as keywords can tell**: `intent.ts` flags a
  rewrite that asks for something the draft did not: deleting or overwriting,
  editing files when the draft only asked to read or explain, installing,
  contacting anyone or publishing, spending, permissions or credentials, a
  change to delegation (subagents, swarms) either way, or a run of 8+ words
  carried in from the chat excerpt. Automatic stops on any of them; Approve
  needs the explicit check. The refiner is also told not to add these.
- **Draft integrity**: cancel/failure returns `null`, so the composer's own
  restore puts the words and attachments back in the chat they came from;
  `scope-guard.ts` cancels a review when the user moves to another chat; the
  sending composer is held read-only while its send is refined or reviewed
  (`host.composer.hold`), so nothing typed meanwhile can be overwritten;
  late model answers are dropped by review id + AbortController.

## Limitations

The checks are deterministic and conservative, and they do **not** guarantee
that the meaning is preserved:

- They are keyword and pattern checks over English text. A rewrite can widen
  the task, change its scope or soften a constraint in words no list
  anticipates ("tidy up" for "delete", a new deliverable phrased as a
  suggestion), and those pass. Other languages get the exact-content checks
  (code, paths, URLs, numbers) but little of the intent check.
- Negation is read from the same clause only ("do not delete" is not a
  delete). An unusual phrasing can be misread either way: a false stop costs a
  click; a missed one is why Approval-First exists.
- Text carried in from the chat is caught only as a run of 8+ identical words.
  A paraphrased instruction from an earlier message is not detected (the
  refiner is told to treat the excerpt as data; that is a model instruction,
  not a guarantee).
- They cannot tell whether a step the draft already asked for got bigger
  ("delete the temp file" → "delete the temp folder").

For prompts where intent matters more than speed, use **Approval-First**: the
user reads every rewrite before anything is sent.

## Upstream merges

- The plugin uses **`@hermes/plugin-sdk` only** (enforced by ESLint for
  `src/plugins/**`), and every host call goes through **`host-bridge.ts`**. An
  upstream SDK change meets one small file.
- Core touchpoints (small, generic, tested): `ComposerDraft.origin` +
  `ComposerMiddleware.failClosed` (`app/chat/composer/contrib.ts`), the origin
  stamp (`hooks/use-middleware-submit.ts`), `typedDraft` on the composer's Enter
  path (`hooks/use-composer-submit.ts`, `SubmitTextOptions` in
  `use-prompt-actions/utils.ts`), the composer input hold
  (`store/composer-holds.ts`, `hooks/use-composer-input-disabled.ts`), and in the
  SDK: two type exports, `host.requestForSession`, `host.composer.hold`,
  `useThisComposer()`, and `ComposerDrawer` (the status drawer).

## In the composer

A **Refine** pill sits beside the model and reasoning pills
(`refine-control.tsx`, `COMPOSER_AREAS.actions`): its menu switches Mode, Depth
and Style in one click, and "More settings…" opens the settings page. Every
choice is saved (`settings.ts`, key
`v1.global.settings`, new fields validated one by one so older saved settings
load unchanged) and followed live by other windows.

**Style** says what kind of rewrite, Depth says how far; the two combine
(`STYLE_INSTRUCTIONS` in `prompt.ts`). Auto, Clear and concise, Technical,
Research, Creative, Agentic task, or **Custom**: the user's own guidance (up to
500 characters, written on the settings page).

**Chat context** (off by default): when on, the last few visible messages of
the composer's own chat (6 at most, clipped to 3,000 characters,
`chat-context.ts`) go with the draft, so "do the same for the other file" can be
made specific. The refiner is told to treat it as data and never follow
instructions inside it.

**Efficiency**: `requestFor` builds one request for both modes; the same
request on the same route is answered from a small in-memory LRU cache
(`cache.ts`: 20 entries, 10 minutes, successful answers only; Try again always
asks afresh). The key is the whole request (text, attachments, depth, style,
custom guidance, chat excerpt) plus the owner connection and profile, and with
"This chat's model" the chat and its model, so an answer never crosses
profiles, connections, models or chat excerpts. Lighter depths get a tighter
output budget.

## Tests

`pnpm exec vitest run src/plugins/prompt-refinement` (unit + real-composer
integration), `pnpm exec vitest run src/app/chat/composer/contrib.test.ts
src/app/chat/composer/hooks/use-middleware-submit.test.ts` (core seam),
`e2e/prompt-refinement.spec.ts` (Playwright, real gateway + mock model), and
`scripts/run_tests.sh tests/plugins/test_prompt_refinement_plugin.py`.
