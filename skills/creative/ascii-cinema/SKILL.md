---
name: ascii-cinema
description: Build browser-playable ASCII cinema explainers.
version: 1.0.0
author: Mustafa Sarac (@mrsarac), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ascii, cinema, html, animation, storytelling, explainers, creative]
    category: creative
    related_skills: [ascii-art, ascii-video, claude-design, excalidraw]
---

# ASCII Cinema Skill

Build a self-contained browser explainer whose narrative is rendered as animated
ASCII scenes. Use this for interactive HTML storytelling, not static text art or
rendered video files.

## When to Use

Use this skill when the user wants one or more of these:

- an ASCII cinema, terminal story, or browser-playable ASCII explainer;
- a living focal subject rather than a static diagram;
- scene-by-scene progression with explanatory captions;
- scene-aware backgrounds that reinforce each concept;
- a shareable microsite, launch asset, or interactive teaching artifact;
- a page that remains useful as both a live experience and still screenshots.

Choose a neighboring skill instead when:

- a banner, logo, or single frame is enough: use `ascii-art`;
- the deliverable is MP4, GIF, or an image sequence: use `ascii-video`;
- ASCII is only a small decorative detail in a broader page: use
  `claude-design`;
- a static relationship diagram communicates the idea better: use
  `excalidraw`.

## Prerequisites

- A writable workspace for the final `.html` artifact.
- A modern browser with JavaScript enabled.
- The Hermes file tools: `read_file`, `search_files`, `write_file`, and `patch`.
- The Hermes browser tools: `browser_navigate`, `browser_console`, and
  `browser_vision`.
- Optional Node.js access through `terminal` for inline JavaScript syntax
  validation.

No framework, package install, network dependency, or build step is required for
the default artifact.

## How to Run

1. Inspect the target workspace with `search_files` and read relevant copy,
   design tokens, and existing HTML with `read_file`.
2. Define the audience, message, transformation, and ordered scene arc before
   changing files.
3. Read `references/scene-grammar.md` and map every scene to a teaching intent,
   focal motion, background language, caption, and duration.
4. Create one self-contained HTML file with `write_file`. Keep CSS, scene data,
   and runtime JavaScript inline unless the user explicitly requests a larger
   application structure.
5. Improve small, isolated sections with `patch`; do not rewrite unrelated
   content merely to polish the cinema.
6. Start a loopback-only local preview with `terminal` when direct file
   navigation is unavailable.
7. Open the artifact with `browser_navigate`, inspect runtime failures with
   `browser_console`, and review the composition with `browser_vision`.
8. Verify controls, keyboard use, narrow screens, reduced motion, and full-glyph
   visibility before delivery.

Completion means the real artifact has been opened and exercised in a browser,
not merely written to disk.

## Quick Reference

| Concern | Required behavior |
|---|---|
| Output | One self-contained HTML file with inline CSS and inline JavaScript; no build step |
| Story | Every scene advances one clear idea |
| Focal subject | Visibly alive through restrained motion |
| Background | Changes with the scene's teaching intent |
| Caption | Explains why the scene matters |
| Controls | Scene selection plus play/pause and restart |
| Keyboard | Native buttons; Enter and Space activation |
| Mobile | No body overflow; complete glyph rows remain visible |
| Motion | Reduced-motion mode disables autoplay but keeps manual control |
| Network | No external asset, font, or fetch dependency by default |
| Verification | Static checks plus real browser interaction and visual review |

A useful scene record contains:

```js
{
  id: "inspect-state",
  short: "Inspect",
  title: "Inspect before acting.",
  narration: "Read the real state before choosing a change.",
  mode: "inspect",
  duration: 6500,
  accent: "#67e8f9"
}
```

## Procedure

### 1. Establish the narrative contract

Write down:

- audience;
- initial misconception or tension;
- final understanding or action;
- ordered scene list;
- one sentence each scene must teach.

Prefer five to eight scenes. Combine scenes that teach the same idea. The arc is
ready when removing or reordering a scene would make the explanation weaker.

### 2. Assign scene grammar

For every scene choose:

- one background motif;
- one focal-subject behavior;
- one concise title;
- one explanatory caption;
- one accent or mood;
- one duration.

Use `references/scene-grammar.md` for motif patterns. A scene is ready when its
motion and background communicate the same concept as its caption.

### 3. Build the semantic shell

Use semantic sections and native controls:

- `<main>` for the experience;
- one heading hierarchy;
- `<pre aria-live="polite" aria-atomic="true">` for the cinematic frame when each
  complete scene change should be announced;
- `<button>` elements for playback and scene selection;
- a visible caption and progress indicator;
- an inline data-URI favicon to avoid a preview-only 404.

Keep the first paint meaningful. The initial scene, caption, and controls must
exist before animation begins.

### 4. Separate data, rendering, and control state

Keep these layers distinct inside the inline script:

1. scene records;
2. pure scene-to-frame rendering functions;
3. DOM update functions;
4. playback state and timer ownership;
5. event handlers.

Clamp or wrap scene indexes in one function. Maintain at most one active timer.
Restart must clear the old timer before returning to the first scene.

### 5. Make motion deliberate

Use two or three restrained motion cues for the focal subject, such as blink,
bob, pulse, scan, trail, or orbit. Do not animate every region at once.

Honor `prefers-reduced-motion: reduce` in both CSS and JavaScript:

- disable autoplay and nonessential transitions;
- show the control state as paused when no timer exists;
- preserve scene buttons, restart, and manual next/previous behavior.

Reduced motion removes involuntary motion; it does not remove access to content.

### 6. Protect narrow screens

A container can fit while its final monospace glyph is clipped. Verify both:

- document-level overflow; and
- the rendered width of the longest complete ASCII row.

Use a responsive grid-density or font-size rule when the full row exceeds its
frame. Preserve character aspect ratio and meaning rather than shrinking text
until it becomes unreadable. Put intentionally wide comparisons inside an
explicit horizontal-scroll region; never make the page body scroll sideways.

### 7. Add restrained controls

Include only controls that improve comprehension:

- play/pause;
- restart;
- direct scene selection;
- optional previous/next when direct selection is not enough.

The active scene needs more than color alone: use text, shape, `aria-current`, or
another programmatic state. Focus indicators must remain visible.

### 8. Verify the source contract

Before browser review, check:

- one document title and one viewport declaration;
- no duplicate element IDs;
- no external scripts, stylesheets, fonts, media, or data calls by default;
- exactly the intended inline runtime;
- valid inline JavaScript syntax;
- labeled controls and reduced-motion handling;
- no embedded credentials, tokens, personal data, or private operational paths.

Use `terminal` to run `node --check` on extracted inline JavaScript when Node.js
is available. Static checks are a preflight, not browser proof.

### 9. Exercise the real browser

Use `browser_navigate` to open the artifact. Then:

- read `browser_console` for uncaught exceptions and warnings;
- select the first, middle, and last scenes;
- test play/pause, restart, rapid toggling, and wrap boundaries;
- activate controls by keyboard;
- inspect desktop, tablet, mobile, and narrow-mobile layouts;
- emulate reduced motion;
- confirm body overflow is zero and complete glyph rows are visible;
- use `browser_vision` to judge hierarchy, caption readability, and whether the
  focal subject remains visually dominant.

When a check fails, reproduce it deterministically, fix the smallest responsible
layer, and rerun the failed case plus the complete matrix.

### 10. Deliver evidence

Report:

- the absolute artifact path;
- scenes and controls exercised;
- tested viewports and motion mode;
- console/page/network result;
- any intentionally scrollable region;
- remaining limitations.

Do not label the artifact complete from screenshots alone. Keep the source and
reproducible checks available until the user accepts the result.

## Pitfalls

1. **One background for every scene.** The experience becomes decorative.
   Change the ambient grammar with the concept.
2. **Static focal subject.** The cinema loses its living quality. Add restrained,
   scene-relevant motion.
3. **Controls competing with the frame.** Remove secondary controls and metadata.
4. **Labels instead of teaching captions.** Explain why the scene matters, not
   only what it is called.
5. **More than one timer.** Rapid toggles accelerate playback and leak work.
   Centralize timer ownership and clear before starting.
6. **False pause state in reduced motion.** Derive the label from actual timer
   state, not an autoplay assumption.
7. **Mobile container fits but glyphs clip.** Measure the complete row, then
   lower grid density or font size within a readable bound.
8. **Screenshot-only confidence.** A still cannot prove keyboard control,
   autoplay state, timer safety, or runtime cleanliness.
9. **External convenience assets.** A font or icon request breaks offline use
   and creates avoidable network noise. Inline or use system resources.
10. **Generic terminal styling.** ASCII is the medium, not the concept. Let the
    story determine palette, motion, and composition.

## Verification

- [ ] Frontmatter is parseable and the description is at most 60 characters.
- [ ] Output is a self-contained HTML artifact with a useful first paint.
- [ ] Scene data, rendering, UI updates, and playback state are separated.
- [ ] Every scene changes content and background intent.
- [ ] The focal subject has restrained, meaningful motion.
- [ ] Captions teach the narrative rather than repeat scene labels.
- [ ] Play/pause, restart, direct selection, and boundary behavior work.
- [ ] Native buttons work with pointer, Enter, and Space.
- [ ] No duplicate IDs or inline JavaScript syntax errors exist.
- [ ] No unexpected console, page, or network errors occur.
- [ ] Desktop, tablet, mobile, and narrow-mobile layouts were exercised.
- [ ] Document-level horizontal overflow is absent.
- [ ] The longest rendered ASCII row is fully visible or intentionally scrollable.
- [ ] Reduced motion disables autoplay while preserving manual navigation.
- [ ] No secret, private path, or unnecessary external dependency is embedded.
- [ ] Visual review confirms hierarchy, contrast, caption readability, and focus.
- [ ] Final report names the artifact path, real checks, and remaining limits.
