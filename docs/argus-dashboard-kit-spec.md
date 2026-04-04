# Argus Dashboard Kit Spec

A concept/spec for lightweight, thesis-adjacent research dashboards in Argus.

---

## Purpose

Argus dashboards are not meant to be full-blown web applications or generalized BI products. They are lightweight research artifacts that sit alongside a thesis note and help surface the most important evidence, visuals, and point-in-time context.

The goal is to standardize the workflow and design philosophy without over-standardizing the output.

This spec defines:
- the role dashboards play inside Argus
- the recommended project/folder shape
- the lifecycle of a dashboard
- the design philosophy Argus should follow
- a lightweight publishing philosophy suitable for private hosting

---

## Core Idea

Argus should support a shared dashboard kit plus per-dashboard instances.

That means:
- one reusable dashboard kit provides the visual language, rendering conventions, payload expectations, and workflow guidance
- each dashboard remains a self-contained project instance tied to a single company, token, theme, or thesis
- individual dashboards can be ephemeral, weird, or highly specific without needing to conform to a rigid component checklist

This is deliberately a middle ground between:
- one-off bespoke dashboard projects with no shared structure
- a heavy production dashboard framework that is too expensive to build and maintain for ephemeral thesis work

---

## Product Philosophy

Argus dashboards should be treated as research artifacts, not SaaS apps.

They should optimize for:
- speed of creation
- clarity of thought
- point-in-time usefulness
- ease of refresh when worthwhile
- low maintenance burden
- portability

They should not optimize for:
- maximal abstraction
- backend complexity
- generic productization too early
- polished enterprise software workflows

A dashboard that is useful once and never refreshed again can still be a success.

---

## Relationship to Argus Notes

Each dashboard should be a companion to an Argus research note, not a replacement for it.

Recommended split:
- the Argus note holds the longform thesis, narrative, judgment, and revision history
- the dashboard holds the visual summary, structured metrics, screenshots, charts, and compact evidence surface

A dashboard should always be able to point back to:
- its canonical thesis note
- the entity or theme it belongs to
- its last refresh date
- its source provenance

A thesis note should ideally point forward to:
- the dashboard slug/path or hosted URL
- the latest refresh date
- any important caveats about data freshness

---

## Dashboard Design Philosophy

Argus should standardize design philosophy, not a fixed list of required components.

### 1. Visual-first, not prose-first
Prefer:
- charts
- annotated screenshots
- diagrams
- logos/icons where helpful
- compact tables
- bullets

Avoid long explanatory text blocks unless they are necessary for nuance.

### 2. Compact and screen-efficient
Dashboards should use screen space aggressively and intelligently.

Prefer:
- dense, high-signal layouts
- above-the-fold prioritization of the most decision-relevant information
- mobile- and laptop-friendly layouts
- compact section spacing

Avoid airy marketing-page layouts or oversized UI chrome.

### 3. Chart-guideline-driven
Charts should follow Argus chart principles:
- high signal density
- compact presentation
- readability on phone and laptop screens
- clear labels and titles
- no decorative charting that does not advance the thesis

Charts should earn their screen space.

### 4. Prefer images and bullets over text walls
If the same point can be conveyed with:
- a screenshot plus 2 bullets
- a chart plus 3 bullets
- a compact comparison table

that should usually be preferred over 2-4 prose paragraphs.

### 5. Source-cited and epistemically clear
Dashboards should make provenance legible.

Distinguish clearly between:
- fetched data
- manually entered values
- derived metrics
- interpretive thesis statements

Whenever practical, cite:
- provider or source name
- date/time of refresh
- relevant URL or reference location

### 6. Snapshot-friendly
A dashboard should still be useful if it is never refreshed again.

This implies:
- the dashboard should make sense as a point-in-time artifact
- freshness metadata matters
- brittle refresh logic is acceptable if the artifact is still valuable without continuous maintenance

### 7. Flexible composition over rigid templates
Argus should not require every dashboard to contain the same sections.

Some dashboards may be:
- chart-heavy
- image-heavy
- metric-heavy
- timeline-heavy
- comparison-heavy

The kit should provide consistency of quality and visual grammar, not sameness of page shape.

### 8. Reuse patterns instead of inventing a new design system every time
New dashboards should inherit from the shared Argus dashboard kit rather than creating a fresh design language per project.

The dashboard should feel like an Argus artifact, not an unrelated microsite.

### 9. Responsive and visually validated
When refactoring or polishing a dashboard, visual review matters.

The preferred workflow is:
- use screenshots/references as source of truth
- match spacing, hierarchy, layout, and responsive behavior closely
- iterate visually until the dashboard actually looks right on desktop and mobile

### 10. Simplicity under ambiguity
If a design decision is ambiguous, prefer the simplest implementation that preserves the thesis intent and visual direction.

Do not introduce complexity merely because the tooling allows it.

---

## Frontend Refactor Guidance

When using Codex or another coding agent to improve dashboard presentation, Argus should follow the spirit of OpenAI's Codex frontend-design guidance:

- use screenshots and design references as the source of truth
- reuse the existing Argus dashboard kit and visual language instead of inventing a parallel design system
- match spacing, hierarchy, layout, and responsiveness closely
- if details are ambiguous, choose the simplest implementation that preserves the overall direction
- validate visually across screen sizes and iterate until it looks correct

Reference:
- https://developers.openai.com/codex/use-cases/frontend-designs

This guidance should be treated as a refactor playbook, not as a mandate to build elaborate frontends.

---

## Shared Kit vs. Dashboard Instance

### Shared dashboard kit
The shared kit should provide:
- a visual language
- typography and spacing defaults
- chart usage conventions
- source/provenance conventions
- screenshot/image treatment conventions
- responsive behavior conventions
- a standard payload contract
- a rendering shell
- optional example layouts

The shared kit should not enforce a single page schema.

### Per-dashboard instance
Each dashboard instance should remain self-contained and thesis-specific.

Each instance may include:
- local data files
- local refresh scripts
- local overrides or one-off layout choices
- asset-specific visualizations
- snapshots and exports

This preserves the current strength of the workflow: dashboards are cheap to spin up and easy to abandon when they are no longer useful.

---

## Recommended Folder Shape

A good default shape for each dashboard instance is:

```text
Dashboards/
  _kit/
    ...shared shell, styles, helpers, docs...
  <slug>/
    manifest.json
    README.md
    data/
    scripts/
    src/
    dist/
    assets/
```

### Folder intent

`_kit/`
- shared Argus dashboard kit
- common styles, shell, examples, conventions

`<slug>/manifest.json`
- dashboard metadata and linkage

`<slug>/data/`
- raw or semi-processed inputs
- query definitions
- exported JSON/CSV snapshots

`<slug>/scripts/`
- refresh/build scripts
- data shaping utilities

`<slug>/src/`
- source template files for the dashboard

`<slug>/dist/`
- built static output intended for viewing or publishing

`<slug>/assets/`
- images, screenshots, icons, static media

This structure should be treated as a recommendation, not a hard rule.

---

## Minimal Manifest Concept

Each dashboard should have a tiny manifest that lets Argus reason about it programmatically.

Suggested fields:
- `slug`
- `title`
- `entity_type` (`company`, `token`, `theme`, `other`)
- `entity_name`
- `ticker_or_symbol`
- `thesis_note_path`
- `status` (`draft`, `snapshot`, `living`, `archived`)
- `last_refreshed_at`
- `data_sources`
- `refresh_command`
- `publish_target`
- `hosted_url`
- `notes`

The manifest should be descriptive metadata, not a replacement for the thesis note.

---

## Standard Payload Philosophy

The most important standardization is the payload shape, not the page shape.

Argus should ideally emit a canonical payload that is easy for the shared kit to render. The exact schema can evolve, but it should cover:
- metadata
- freshness/provenance
- metrics
- charts/data series
- compact thesis claims
- source references
- refresh metadata

A canonical payload keeps the frontend simple and makes dashboard generation much easier for Hermes/Codex.

The payload should support a mix of:
- frozen point-in-time numbers
- derived metrics
- links to supporting screenshots/assets
- short-form thesis statements

---

## Dashboard Lifecycle

Dashboards should have explicit lifecycle states.

### 1. Draft
A dashboard in active creation or thesis formation.

Characteristics:
- incomplete data
- rough layout
- many manual values are acceptable

### 2. Snapshot
A point-in-time dashboard that is useful as an artifact even if never refreshed.

Characteristics:
- static/frozen payload
- enough provenance to remain interpretable later
- refresh path may exist, but is not required to be robust

### 3. Living
A dashboard that is worth revisiting and refreshing repeatedly.

Characteristics:
- refresh script is more reliable
- data sources are more regularized
- may justify incremental standardization over time

### 4. Archived
A dashboard kept for historical reference but not expected to be refreshed.

Characteristics:
- final snapshot retained
- hosted URL may be removed or frozen
- thesis note still points to it for historical context

This lifecycle lets Argus avoid overbuilding every dashboard while still allowing promotion of the best ones.

---

## Publish Philosophy

Publishing should be optional and secondary to local usefulness.

A dashboard should first succeed as:
- a local artifact
- a private research tool
- a thesis companion

Only then should it optionally be published.

### Default publishing principles
- static first
- private by default where possible
- custom-domain friendly
- low-ops
- easy to abandon

### Avoid by default
- public-by-default hosting for sensitive or proprietary research
- backend-heavy architectures for mostly static artifacts
- bespoke auth systems if a managed edge-auth solution is sufficient

---

## Hosting Recommendation Direction

For private hosted dashboards, the most natural direction is:
- static hosting on a custom domain
- access protected behind an authentication layer

A strong default candidate is:
- Cloudflare Pages + Cloudflare Access

Why this direction fits:
- preserves the static-first model
- supports custom domains cleanly
- provides auth gating without requiring Argus to own a full auth stack
- keeps operational overhead low for ephemeral dashboards

GitHub Pages is convenient for public prototypes but is not the right default for private Argus research artifacts.

---

## What Argus Should Eventually Be Able to Do

At a capability level, Argus should eventually support workflows like:
- create a dashboard scaffold for a company/token/theme
- link a dashboard to an Argus note
- build a first-pass static dashboard from a canonical payload
- refresh a dashboard when sources are still valid
- mark a dashboard as snapshot/living/archived
- publish a dashboard to a private target when desired
- preserve local-first usability even without publishing

This is a workflow capability, not necessarily a single monolithic feature.

---

## Success Criteria

This dashboard kit approach is successful if it leads to:
- faster dashboard creation
- more coherent Argus visual artifacts
- better linkage between thesis notes and dashboards
- easier refresh/reuse when a dashboard becomes important
- less incentive to overengineer ephemeral work
- a clearer path from one-off dashboard to reusable dashboard when warranted

---

## Non-Goals

This spec does not propose:
- a mandatory fixed dashboard component list
- a full backend dashboard platform
- enterprise analytics infrastructure
- strict schema lock-in from day one
- mandatory publishing for every dashboard

The aim is to create a reusable Argus pattern without destroying the speed and flexibility that make these dashboards useful in the first place.
