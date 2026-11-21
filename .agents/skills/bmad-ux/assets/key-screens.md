# Key Screens Renderer

Subagent prompt. Fired at Finalize (or during late Discovery once layout decisions firm up). Produces 1:1 HTML mocks of the load-bearing surfaces so the spine can link to them as visual reference. Spine remains the contract; mocks illustrate.

## Inputs

`.memlog.md`, the current drafts `DESIGN.md` and `EXPERIENCE.md`, `.working/` (especially the chosen color-theme and direction mocks), source PRD. The user names which surfaces to render — typically 2-4: the canonical entry surface, the most complex flow's hero screen, any load-bearing overlay/modal, and (when present) the Week / list / dashboard view.

## What to render

One HTML file per screen, at `.working/key-{slug}.html`. Each file: realistic device frame (phone or browser), real product content from the conversation (no lorem), every visible string voice-checked against `.memlog.md`, all decided tokens applied. Show one canonical state per screen; if a surface has a load-bearing alternate state (focus, error, crisis-card-present), render it as a second column or section in the same file.

Inline CSS, system fonts, no JS, no network. The mock must render fully offline. Comment block at the top of the `<style>` notes which spine sections govern this screen so a future reader knows what to check.

## What to return

A compact summary to the parent:
- file path per screen
- one-line caption per screen ("Today picker at rest; accent on Thought record")
- which spine sections each mock illustrates (Component Patterns rows, State Patterns rows, Flow steps)

The parent, at Finalize "Promote working artifacts," uses this summary to insert inline `mockups/...` links into the relevant spine sections.

## Anti-patterns

- Do not invent layout — every composition decision must trace to a `.working/` artifact or a confirmation in `.memlog.md`. If a layout question is open, the mock is premature.
- Do not show every screen of every flow — 2-4 load-bearing surfaces, not 14.
- Do not stage marketing copy. Strings come from `.memlog.md` and voice rules.
- Do not introduce a new pattern not in the spine's Component Patterns table. If you need one, log it and ask before rendering.
