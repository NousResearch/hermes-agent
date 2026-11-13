# The SCALE framework — on-brand volume without a designer (and without the "Canva look")

Canva's job in this stack is **on-brand design at scale**. The failure mode is the recognizable template post;
the win is a brand system that produces volume. Each letter is a gate; most users fail at S (skip the Brand Kit)
and never reach A (the automation layer).

## S — Start from the Brand Kit
Before generating anything, lock the **Brand Kit**: colors, fonts, logos, and (on paid tiers) the brand voice.
Every Magic Studio feature reads it — this is the difference between ~20% and ~80% on-brand *first drafts*, and
Canva's own data ties consistent Brand Kit use to measurably better brand recognition (attribute). Pull the
system from `brand-profile` + `design-and-templates`; on Teams tiers, lock templates + approvals so scale can't
drift off-brand.

## C — Customize past the template
Templates are **scaffolding for hierarchy and spacing — not the finished design.** The "Canva look" = untouched
layout + default fonts + recognizable stock + the template's palette. The de-templating moves: **replace every
stock element** (your photos, your product shots, brand-generated imagery), **change the layout bones** (move,
merge, delete blocks), **apply Brand Kit type + color over the defaults**, and **strip the template decorations.**
If a stranger can name the template, keep going.

## A — Automate the repeatable
The layer almost nobody uses, and where the real leverage is:
- **Bulk Create:** design ONE master template with data fields → connect a CSV/Excel (text, prices, image URLs)
  → generate every variant in one run (60 tips = 60 graphics). QA overflow/line-breaks before export.
- **Magic Resize:** one master → every platform size in a click; fix collisions per size.
- **Magic Switch:** one asset → other formats + 150+ languages (QA line breaks, CTAs, protected brand words).
- Via the **Connect APIs / MCP connector**, the agent can drive create/resize/adapt directly where connected.

## L — Layer in AI, keep it editable
Use the AI as **material, not the final say**: Dream Lab / AI 2.0 for imagery and drafts (editable objects >
flat images; Magic Layers unlocks flat AI images into layers), Magic Write as a **second-pass editor** (the
voice comes from `voice-builder`), Magic Eraser/Expand/Grab for cleanup. Honor the trade-off: dedicated
generators (→ `image-prompt` router) beat Dream Lab on pure quality; Canva wins on workflow. **AI-disclosure**
where generated visuals require it; never fake real people/brands.

## E — Export right + hand to publish
Correct format per destination (PNG/JPG for images, MP4 for motion, correct dimensions per platform; transparent
PNG where needed), legibility floor held (contrast, sizes, ≤2 fonts, safe zones), alt text prepared — then the
finished file goes to `scheduling-and-queue` → **WoopSocial publishes** (Canva renders; it does not publish in
this stack — see the Canva-scheduler note in scope). The human approves every visual before it ships.
