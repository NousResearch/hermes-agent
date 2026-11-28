# Examples — brief → prompt → tool

Worked end-to-end flows showing the brief, the universal prompt, the tool routing, and honest gates.
Brand context: a SaaS social scheduler (clean, design-led, mono + terminal-green accent). Your output
pulls the real brand from `brand-profile.md`.

---

## Full flow: a post needs a visual

**Need:** "I need a visual for a post about batching a week of content."
**Brief:** stop-scroll hero for IG (4:5), conveying *"a whole week, done in one calm sitting,"* in the
clean mono brand look.
**Universal prompt:** "A calm, minimal flat-lay: a single laptop and a coffee on a light desk, a tidy
weekly calendar visible on screen, soft morning light, muted neutrals with one terminal-green accent,
clean editorial style, generous empty space top-left. 4:5."
**Route:** photoreal-ish lifestyle with no critical text → `nano-banana` (or a photoreal model if it
must look like a real photo). **Hand off** to that mini-skill for model-specific craft.

## Routing a batch of needs

- "Quote graphic with our tagline" → **`ideogram` or `nano-banana`** (both strong at text — pick by
  tooling).
- "Photoreal headshot-style portrait" → **`nano-banana`** (or Midjourney; not Ideogram — weak at real faces).
- "Same product, six campaign scenes" → **`nano-banana`** references or **`flux`** multi-reference.
- "Swap the background, keep the product identical" → **`nano-banana`** or **`flux`** (Kontext edit).
- "Our logo as a clean vector" → **Recraft** (vector-native).
- "A 6-second moving hook" → **`veo-3`**.

## Weak → strong (universal)

**Weak:** "cool office, 4k, masterpiece, ultra detailed"
**Strong:** "A minimal co-working corner: one laptop on light oak by a window, soft left light, muted
neutrals + a small green plant, editorial photography, shallow depth of field, empty wall on the right
for a headline. 16:9."

## The "don't generate" gate

**Need:** "Generate a realistic photo of my actual cafe and staff."
**Answer:** use a **real photo** of the actual cafe and team — it's authentic, it's *them*, and AI
can't (and shouldn't) fabricate your real place/people. AI generation is for what doesn't exist or must
match a specific look; this isn't that.

## Designing for an overlay

**Need:** a background with a headline added later in Canva.
**Prompt move:** "…deliberate negative space in the lower third, low detail and even tone there, high
contrast so white text reads on a phone. No text in the image." (Then the headline goes on in the
design tool.)

---

## Honest cross-cutting scope (say this)

- **Should it even be AI?** A real photo/screenshot/chart/UGC often beats generic AI — gate first.
- **Disclose AI** per platform/region (Meta "Made with AI", EU AI Act); **watermark behaviour varies by
  tool** (SynthID on Google tools; none enforced on Ideogram) — don't rely on it for disclosure.
- **No real people / no IP** — no real identifiable individuals, copyrighted characters, or trademarked
  logos; original directions only.
- **Verify** in-image text and any data before publishing.
- **Accessibility** — contrast + legibility on mobile; leave room for overlays.
- **Router, not generator** — this skill briefs/describes/routes; tool skills do model craft;
  `tools/integrations/*` connect; WoopSocial publishes via Media upload.

---

## What the examples share

- **Brief first** (job + the one thing + format + brand), then a **specific natural-language prompt.**
- **Routed to the right tool** with an honest read, then **handed off.**
- **Gated** for "real asset beats AI," **designed for overlays**, and **disclosed/IP-safe/verified.**
