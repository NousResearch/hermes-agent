# Prompt Anatomy (model-agnostic)

The transferable craft: how to describe an image so any modern model gets close. Tool-specific tricks
(Ideogram's quotes rule, Nano Banana's references, Veo's audio) live in the tool skills — this is the
shared foundation underneath them.

## Describe, don't incant

Modern image models reason over **natural language**. Drop the 2023-era token spam — "4k, masterpiece,
trending on artstation, ultra-detailed, best quality" adds nothing. **Specific description** is what
moves quality. "A weak prompt names a thing; a strong prompt directs a picture."

## The anatomy (cover what matters for the shot)

- **Subject** — who/what, specifically ("a matte-black ceramic mug," not "a mug"). The focal point.
- **Composition / framing** — shot type and arrangement ("close-up, centered, lots of negative space
  on the left"; "flat lay, top-down"). Where things sit and how much room they get.
- **Lighting** — direction, quality, time ("soft window light from the left"; "harsh midday";
  "golden hour"). Lighting carries most of the mood.
- **Colour / palette** — the brand palette or a named scheme ("muted earth tones, one terminal-green
  accent").
- **Style / medium** — "clean flat vector," "editorial photography," "3D render," "watercolour."
- **Mood / atmosphere** — the feeling ("calm, optimistic"; "energetic, high-contrast").
- **Detail / focus** — what's sharp vs soft ("shallow depth of field on the product").
- **Constraints** — **aspect ratio**, **negative space for text overlay**, and what to **avoid**.

A workable order: *[Subject + adjectives], [composition], [lighting], [colour], [style], [mood].
[Aspect ratio + constraints].* Then add any tool-specific moves in the tool skill.

## Specificity beats length

Long isn't the goal; **specific** is. Every word should change the picture. Cut adjectives that don't
direct anything. If a detail matters (a colour, a layout, a piece of text), state it precisely; if it
doesn't, leave it open and let the model choose.

## Iterate, don't one-shot

Generate, read what you got, change **one thing** at a time ("same, but warmer light / more negative
space / tighter crop"). Most models give variations — make a few, pick, refine. (Iterate cheap, then
finalize — costs/quality tiers are per-tool.)

## Design for the overlay (if text is added later)

If a headline/caption goes on top in a design tool, **compose for it now**: leave a deliberate
low-detail region, keep contrast high where the text will sit, and don't fill the whole frame. Plan the
text's home before you generate.

## Weak → strong (the move)

**Weak:** "cool office, 4k, masterpiece, ultra detailed"
**Strong:** "A calm, minimal co-working corner: a single laptop on a light oak desk by a window, soft
morning light from the left, muted neutrals with one small green plant, clean editorial photography,
shallow depth of field, generous empty wall space on the right for a headline. 16:9."

Carry the brand in (`brand-profile.md`), set the aspect ratio, then route to a tool
(`choosing-the-tool.md`).
