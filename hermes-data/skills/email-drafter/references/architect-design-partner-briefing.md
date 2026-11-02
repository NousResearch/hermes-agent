# Architectural design-partner briefing (what DRAAS expects from an architect)

Validated 2026-09-25 (Roshini + Gowri). This is the reusable **engagement model**
NDR wants stated to any senior architect / design firm he approaches to lead the
creative process — and the email shape that carries it.

## Trigger

NDR asks for a short briefing document/email "to the architect" explaining what
we expect from them and how the creative process should run — e.g. *"create a
briefing document for an architect… 150–250 words max in an email to Roshni,
include Gowri."* Often follows a long generative-ideation session (see
cross-link below), because he offers to share that run as the sample.

## Recipients (header-verified)

- **To:** Roshini Ranka `rnr@draas.com`
- **Cc:** Gowri Singh `gsingh@draas.com` — the header-verified work address.
  The contacts sheet's `gowrisingh1341@gmail.com` / `gowrisingh72@yahoo.com`
  are NOT the draas.com work identity. Headers win.

Length: **150–250 words** of prose. Blocks + a sign-off, not an essay.
HTML body must be **full width** — see `html-width-and-number-approval.md`
(no `max-width`, no centered narrow table; content spans the client pane).

## The two-phase model — approved language

**The Creative Phase.** Lead with the correction NDR made (see below): this phase
is run JOINTLY with the design team from the very beginning — not only after a
brief is written.

> We want to run this phase jointly with your creative team from the very
> beginning — not only after a brief is written. That means going through
> reference material together, whether Pinterest, Instagram, or curated sets of
> images and videos, and extracting the ideas, elements and aspects we like and
> want brought in, and those we do not.
>
> From there we build a mood board out of AI-generated images that respond to the
> project, its surroundings and the brief. We then iterate on those generated
> images — material, pattern, structure, form, artistic touch — round after
> round, until the direction converges on one clear picture we are confident we
> want to execute. That picture is a visualization and an agreed direction, not
> an executable architectural model.

**The Detailing Phase.**

> Detailed drawings — bylaw compliance, buildability, floor plans, BOQ and
> costing — can be handled by in-house or local architects. We would like the
> senior partners to review these as checks and balances: correctness, accuracy,
> whether the constraints have been honoured, and whether the agreed materials
> and composition still hold.
>
> Drafting and modelling can go to a specialist team. AI-assisted modelling is a
> phase-two exploration, as those tools mature.

Close with: landscape follows the same creative process; and we will share a
sample run (already executed end to end) so the team can see the vision.

## The correction that matters (2026-09-25)

The first draft framed the creative phase as *"once a basic brief and a set of
reference images are in place, we work in an iterative loop…"* — i.e. WE supply
references, YOU generate. NDR rewrote it. Two substantive changes:

1. **Reference harvesting is JOINT work, not an input we hand over.** The
   architect's creative team helps trawl Pinterest / Instagram / curated
   image+video sets and extract what we like and dislike. Ideation is a shared
   activity from day one, not a response to a finished brief.
2. **The output of the loop is a visualization and an agreed direction — not an
   executable architectural model.** Say this explicitly; it is the line that
   separates Phase 1 from Phase 2 and stops anyone treating the mood-board image
   as a construction document.

Also dropped from the original draft: the reference-"basic brief" framing and
the phrase "translates that imagery into architectural elements" as the team's
deliverable (it made the architect the sole author rather than a partner).
Retained: the detailing phase, the checks-and-balances role of senior partners,
and the landscape line.

## Delivery

Draft-only, always — never send. Verify by `drafts().list()` **and** reading the
stored draft back via `format='full'` (see the HTML-body verification pitfall in
SKILL.md — a `html=true` bridge draft stores a **base64** body inside
`multipart/alternative`, so string-checking the raw MIME reports everything
false). Delete any superseded draft so exactly one exists.

## Cross-link

The "sample run we have already executed end to end" is the Tapovana Design-DNA
programme — `reference-driven-image-variation` (and upstream
`architectural-design-dna-analyzer` / `architectural-image-organizer`). That
skill is the executable version of the Creative Phase described here; when the
architect asks "show me", the offer is those run folders.