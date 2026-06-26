# Canonical Brand Design Document template + brand-system map

This reference defines the standard Brand Design Document artifact class used by coach-agent brand ingestion. Product-facing language should call this the **standard Brand Design Document**, **canonical Brand Design Document template**, or **brand system map**. Do not reference the internal originating exemplar by name unless the active brand being documented is actually that brand.

## Core idea
Brand ingestion must convert source truth into a premium visual document plus a deterministic “paint-by-number” system map. The human PDF teaches the operator what the brand feels like; the machine-readable map tells coach agents exactly where to place each brand ingredient.

- **Design tokens** = reusable ingredients: logo, icon, palette, type, spacing, chrome, imagery, voice.
- **Component slots** = deterministic placements: cover logo slot, masthead icon slot, section divider rule, recap-card border, footer line, slide title treatment.
- **Brand system map** = one JSON sidecar combining tokens + component slots + source/provenance + downstream application rules.

## Required deliverables
Every premium Brand Design Document produced through this skill should include:
1. `brand-design-doc.pdf` — final human-facing truth artifact.
2. `brand-design-doc.html` — composed visual source used for PDF export.
3. `brand-system-map.json` — machine-readable tokens, component slots, provenance, and downstream rules.

Use brand-specific prefixes/slugs in filenames when needed, but preserve the artifact roles above.

## Canonical visual page contract
The visual PDF/HTML must follow the same page architecture and premium rhythm every time. Future docs should swap brand-specific content into this architecture, not invent a new document structure.

Required page/section sequence:
1. **Cover / Title Page**
   - Large premium title treatment.
   - Brand name and context.
   - Primary identity mark or wordmark placed in the canonical cover identity slot.
   - A refined field using primary/secondary color tokens and/or approved imagery.
2. **Brand Snapshot**
   - Concise brand posture.
   - Source status: source-backed, inferred, needs approval.
   - Intended downstream use: premium lesson recap, slide/template workflow, branded documents.
   - Quick token summary: identity, color, type, voice.
3. **Identity**
   - Visible primary logo/wordmark/icon treatment.
   - Exact source/provenance for each mark.
   - Clear usage notes and fallback behavior.
   - Third-party/uncertain marks excluded from approved identity surfaces.
4. **Color System**
   - Named swatches with visible fills and hex values.
   - Hierarchy: primary, secondary, accent, neutrals, backgrounds, text.
   - Usage guidance tied to component slots.
5. **Typography**
   - Heading/display direction with examples.
   - Body/supporting type direction with examples.
   - Substitution guidance when exact fonts are unavailable.
   - Slide/recap title hierarchy mapping.
6. **Visual Style**
   - Layout posture, spacing, rule/chrome behavior, card behavior, image style, texture/motif notes.
   - Do/don't examples when source confidence is mixed.
7. **Documents / Application**
   - Specific application rules for premium lesson recap packets.
   - Specific application rules for slide templates.
   - Header/footer behavior, section dividers, recap-card chrome, page rhythm, title treatment.
8. **Source Notes**
   - Source URLs/files/assets.
   - Source-backed vs inferred vs needs-approval separation.
   - Permission/temporary-use caveats where relevant.

## Brand-system map JSON shape
The sidecar should be valid JSON and use this top-level shape:

```json
{
  "schema_version": "brand-system-map.v1",
  "brand": {
    "name": "Example Brand",
    "slug": "example-brand",
    "context": "coach | club | academy | event | temporary-venue",
    "status": "approved | temporary | needs_approval"
  },
  "tokens": {
    "identity": {
      "primary_logo": {"path": "assets/logo@4x.png", "source": "...", "status": "source_backed"},
      "wordmark": {"path": "assets/wordmark@4x.png", "source": "...", "status": "source_backed"},
      "icon": {"path": "assets/icon@4x.png", "source": "...", "status": "source_backed"},
      "fallback": "Use The System identity only if no approved brand mark exists."
    },
    "color": {
      "primary": {"hex": "#123456", "usage": "cover field, masthead, primary rules"},
      "secondary": {"hex": "#234567", "usage": "support panels, secondary dividers"},
      "accent": {"hex": "#C9A86A", "usage": "small rules, callouts, badges"},
      "background": {"hex": "#F7F4EC", "usage": "page background or soft panels"},
      "text": {"hex": "#111111", "usage": "body copy"}
    },
    "typography": {
      "heading": {"family": "...", "fallback": "...", "usage": "cover and section titles"},
      "body": {"family": "...", "fallback": "...", "usage": "body and captions"},
      "caption": {"family": "...", "fallback": "...", "usage": "source notes and metadata"}
    },
    "layout": {
      "page_rhythm": "premium, spacious, structured",
      "border_radius": "...",
      "rule_weight": "...",
      "spacing_scale": ["..."],
      "chrome": "..."
    },
    "imagery": {
      "style": "...",
      "allowed_sources": ["..."],
      "avoid": ["third-party logos", "uncleared sponsor marks"]
    },
    "voice": {
      "tone": "...",
      "avoid": ["generic AI prose", "overly salesy claims"]
    }
  },
  "components": {
    "brand_doc_cover": {
      "logo_slot": "tokens.identity.primary_logo",
      "background": "tokens.color.primary",
      "accent_rule": "tokens.color.accent",
      "title_type": "tokens.typography.heading"
    },
    "brand_snapshot_card": {
      "label_color": "tokens.color.accent",
      "body_type": "tokens.typography.body"
    },
    "identity_panel": {
      "primary_mark": "tokens.identity.primary_logo",
      "support_mark": "tokens.identity.icon"
    },
    "color_swatch_grid": {
      "swatches": ["tokens.color.primary", "tokens.color.secondary", "tokens.color.accent", "tokens.color.background", "tokens.color.text"]
    },
    "lesson_recap_header": {
      "logo_slot": "tokens.identity.primary_logo",
      "rule_color": "tokens.color.primary",
      "metadata_type": "tokens.typography.caption"
    },
    "lesson_recap_section_divider": {
      "title_type": "tokens.typography.heading",
      "rule_color": "tokens.color.accent"
    },
    "lesson_recap_card": {
      "border_color": "tokens.color.secondary",
      "accent_color": "tokens.color.accent",
      "body_type": "tokens.typography.body"
    },
    "lesson_recap_footer": {
      "logo_or_icon": "tokens.identity.icon",
      "text_color": "tokens.color.text",
      "rule_color": "tokens.color.secondary"
    },
    "slide_title": {
      "logo_slot": "tokens.identity.icon",
      "title_type": "tokens.typography.heading",
      "accent_rule": "tokens.color.accent"
    },
    "slide_content_card": {
      "border_color": "tokens.color.secondary",
      "heading_type": "tokens.typography.heading",
      "body_type": "tokens.typography.body"
    }
  },
  "provenance": {
    "source_urls": [],
    "source_files": [],
    "source_backed": [],
    "inferred": [],
    "needs_approval": []
  },
  "downstream_rules": {
    "premium_lesson_recap": ["..."],
    "slide_template": ["..."],
    "branded_documents": ["..."]
  },
  "qa": {
    "rendered_pdf_checked": false,
    "identity_visible": false,
    "swatches_visible": false,
    "token_map_valid_json": false
  }
}
```

## Agent behavior rules
- Build the visual PDF first-class; do not reduce the work to a JSON exercise.
- Build the JSON sidecar as the deterministic handoff for coach agents.
- Use source-backed tokens where available; mark inferred values explicitly.
- If a token is missing, provide a safe fallback rule instead of letting downstream agents improvise.
- The visual doc must never expose the internal exemplar name unless that is the active brand.
- In operator/internal notes, it is acceptable to say the standard was derived from a prior accepted exemplar, but product copy should use only standard/canonical template language.

## NO-GO conditions
A Brand Design Document is incomplete if:
- it omits `brand-system-map.json` for an agent-facing brand-ingestion job;
- it changes the page architecture without explicit operator approval;
- it presents brand prose without component slots;
- it shows swatches or identity in HTML but not in the exported PDF;
- it uses third-party/uncertain marks as approved identity;
- it references the internal exemplar brand in product-facing language when documenting another brand.
