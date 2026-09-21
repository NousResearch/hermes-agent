---
name: site-teardown
description: "Use when reverse engineering a website into a build blueprint: HTML structure, CSS design system, JS interactions, animation techniques, assets, and section-by-section rebuild plan."
version: 1.0.0
author: KENSEI
license: MIT
metadata:
  hermes:
    tags: [design, teardown, reverse-engineering, ui, ux, animation, frontend]
    related_skills: [dogfood, plenishd-design-tokens, coachos-design-tokens]
adoption_status: permanent
---

# Site Teardown

## Overview

Reverse engineer a website into a complete build blueprint: tech stack, HTML structure, design system, interaction model, animation code, assets needed, and section-by-section implementation plan.

Use this for design inspiration, UI reconstruction, app landing-page analysis, and understanding how impressive effects actually work.

## When to Use

Activate when Sahil asks to:

- Tear down a site.
- Reverse engineer or deconstruct a website.
- Clone or recreate a website's design or animations.
- Understand a site's tech stack, UI patterns, animation libraries, or interaction techniques.
- Turn a URL into a build blueprint.
- Analyse raw HTML and explain how to recreate it.

Trigger examples:

- "Tear down this site: <url>"
- "How did they build this website?"
- "Reverse engineer this page"
- "Break down this design"
- "Website blueprint for <url>"
- "I want this kind of landing page for Plenishd"

## Inputs

Accept either:

1. A URL.
2. Raw pasted HTML.
3. A local HTML/CSS/JS bundle.

Raw HTML is best when available because web extraction can summarise or omit details.

## Pipeline

### Step 1: Get the HTML

If raw HTML is provided, use it directly.

If only a URL is provided, fetch/extract the page and preserve:

- element hierarchy
- class names
- data attributes
- image/video paths
- SVGs
- forms
- navigation
- meta tags
- inline styles/scripts
- script and stylesheet URLs

Tell Sahil that raw source via browser View Source gives the most complete teardown when extraction is incomplete.

### Step 2: Find CSS and JS paths

From the HTML, identify app-owned CSS and JS.

Keep:

- `/dist/`, `/build/`, `/assets/`, `/static/`
- `main.js`, `app.js`, `bundle.js`, `index.js`
- `main.css`, `app.css`, `style.css`, `styles.css`
- modulepreload files

Skip third-party noise unless it reveals stack:

- Google Tag Manager / Analytics
- cookie banners
- captcha
- social widgets
- CDN libraries
- WordPress core scripts

Construct full URLs from relative paths.

### Step 3: Extract JavaScript interactions

Fetch main JS files and extract:

- DOM manipulation
- event listeners
- scroll effects
- mouse/touch interactions
- click handlers
- class toggles
- IntersectionObserver usage
- CSS custom property updates
- GSAP / ScrollTrigger timelines
- Lenis / Locomotive smooth scroll
- Barba / Swup transitions
- sliders, carousels, draggable logic
- parallax
- image sequences
- custom cursors
- preloaders
- accordions and menus
- easing values and timings

Build the extraction prompt around class names from the HTML, especially stems like:

`hero`, `animate`, `scroll`, `slide`, `reveal`, `split`, `parallax`, `hover`, `active`, `open`, `toggle`, `frame`, `video`, `zoom`, `marquee`, `cursor`, `loader`, `transition`.

### Step 4: Extract CSS design system

Fetch CSS and extract:

- colours and variables
- `@font-face` declarations
- font assignments
- keyframes
- custom properties
- media queries and breakpoints
- transforms and transitions
- fixed/sticky elements
- z-index system
- spacing scale
- radius values
- shadows
- gradients
- blend/filter effects
- responsive typography
- pseudo-element styling
- layout patterns
- masks and clip-paths
- grain/noise texture implementation

Use class stems from the HTML to target relevant rules.

### Step 5: Assemble teardown document

Save to:

`research/YYYY-MM-DD-{site-slug}-teardown.md`

If no `research/` folder exists, create it if allowed or save in the current working directory.

## Output Template

```markdown
# Site Teardown: {Site Name}

**URL:** {url}
**Built by:** {agency/developer if identifiable}
**Platform:** {confirmed/inferred}
**Date analysed:** {YYYY-MM-DD}

## Tech Stack

| Technology | Evidence | Purpose |
|---|---|---|
| {library} | {script/code/meta evidence} | {what it does} |

## Design System

### Colours
| Name/Usage | Value |
|---|---|
| Primary background | #000000 |

### Typography
| Role | Font Family | Weight | Letter-spacing | Sizes |
|---|---|---|---|---|
| Headings | {font} | {weight} | {spacing} | {size range} |

### Spacing System
{spacing approach}

### Responsive Approach
{breakpoints and strategy}

## Effects Breakdown

| Effect | Implementation | Complexity | Cloneable? |
|---|---|---|---|
| {effect} | {specific implementation} | Low/Med/High | Yes/Partially/Hard |

## Implementation Details

### {Effect #1}
{how it works, code snippets, key insight}

## Assets Needed to Recreate

1. {asset type} — {description and source/generation idea}

## Build Plan

### Recommended Stack
- Framework: {why}
- Styling: {why}
- Animation: {why}

### Packages
```bash
npm install {packages}
```

### Section-by-Section Build Order

**Section 1: {name}**
- contents
- interactions
- implementation approach

## Notes
- gotchas
- licensing concerns
- performance considerations
```

## Quality Bar

Be specific, not vague.

Good: `GSAP ScrollTrigger with scrub:true pinned to .hero, scaling from 1 to 0.3`.
Bad: `Uses scroll animations`.

Distinguish confirmed from inferred.

Group effects by page section so Coding Lead can build from top to bottom.

Be practical about assets: count them, describe them, and suggest how to source or generate them.

## Common Patterns to Watch For

- Image sequences on scroll/mouse.
- SplitText character reveals.
- Parallax layers.
- Scroll-scrubbed animations.
- CSS custom property animations.
- Smooth scrolling via Lenis or Locomotive.
- Page transitions via Barba or Swup.
- Noise/grain overlays.
- Custom cursors.
- Marquee loops.
- Magnetic buttons.
- Reveal-on-scroll.

## Verification Checklist

- [ ] HTML structure captured.
- [ ] Main CSS and JS identified.
- [ ] Third-party scripts filtered.
- [ ] Design tokens extracted.
- [ ] Interactions and animations explained with implementation detail.
- [ ] Confirmed vs inferred facts labelled.
- [ ] Assets listed.
- [ ] Build plan is detailed enough for Coding Lead to execute.

## Local Reference

Original imported source exists at:

`/home/kensei/TeardownSkill.md`
