---
name: stitch-generation
description: AI-assisted UI design using Google Stitch from specifications and sketches
---

# Stitch UI Generation

**Goal:** Generate production-quality UI designs using Google Stitch AI

**Your Role:** Guide the user through preparing inputs and creating a Stitch generation dialog

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## OVERVIEW

Google Stitch transforms text prompts, sketches, and reference images into responsive interfaces.

**Input Formula:**
```
Visual Reference + Sketch + Specification = Stitch Generation
```

**Output:** UI designs exportable to Figma or HTML/CSS

---

## WHEN TO USE

**Use Stitch when:**
- New page with detailed specification ready
- Have a visual reference (existing design or screenshot)
- Have a sketch showing layout structure
- Want rapid visual design iteration

**Skip when:**
- Building design system components (use tokens instead)
- Minor updates to existing designs
- No specification exists yet (write spec first)

---

## PREREQUISITES

1. **Specification exists** for the screen(s) to generate
2. **Visual reference available** (screenshot or approved design)
3. **Sketch available** showing layout structure

---

## WORKFLOW

### Step 1: Create Generation Log

Create a Stitch generation log in the agent experiences folder.

**Location:** `{output_folder}/_progress/agent-experiences/{YYYY-MM-DD}-stitch-{feature}.md`

### Step 2: Pre-Generation Questions

For each potential screen, decide:

| Question | Columns |
|----------|---------|
| Generate in Stitch? | Screen, Has Code?, Has Sketch?, Generate?, Why |
| What reference? | Screen, Reference, Source |

### Step 3: Gather Inputs

| Input | Action |
|-------|--------|
| **Visual Reference** | Take screenshot OR locate existing design |
| **Sketch** | Locate in spec's `Sketches/` folder |
| **Prompt** | Prepare using template below |

### Step 3a: Prepare the Prompt

**DO NOT paste raw specifications into Stitch.** Use the prompt template instead.

**Template:** `templates/stitch-prompt.template.md`

Include: Project Context, Design System, Component Styles, Screen Content (ONE language, no Object IDs), Current State Notes.

### Step 4: Generate in Stitch

1. Go to [stitch.withgoogle.com](https://stitch.withgoogle.com)
2. Upload visual reference and sketch images
3. Paste prepared prompt
4. Generate 2-3 variants
5. Select best result

**Settings:** Standard Mode (quick) or Pro Mode (higher fidelity). Viewport: Mobile 390px or Desktop 1440px.

### Step 5: Review Against Spec

| Check | Pass? |
|-------|-------|
| Content/copy matches spec | |
| Layout follows sketch | |
| Visual style matches reference | |
| All key elements present | |

If issues: Re-prompt with specific corrections or edit in Stitch.

### Step 6: Export & Store

| Format | When | Destination |
|--------|------|-------------|
| **Figma** | Team collaboration | Figma project |
| **HTML/CSS** | Code reference | `{spec-folder}/Visual-Design/` |
| **Screenshot** | Documentation | `{spec-folder}/Visual-Design/` |

**Naming:** `{screen-name}-stitch-v{#}.{ext}`

### Step 7: Update Specification

Add Visual Design section to specification referencing the Stitch output.

---

## STITCH CAPABILITIES & LIMITS

**Does well:** Single screen generation, style matching, responsive layouts, clean HTML/CSS export, Figma-compatible output.

**Limitations:** Best with 2-3 screens at a time, layouts can be generic, no built-in design system awareness, free tier limits (350 standard + 200 pro/month).

---

## PROMPT TIPS

Effective prompts include: App type, Context, Visual direction, Key elements, Actual content/text, Mood/tone, Viewport.

**Template:** See `templates/stitch-prompt.template.md` for complete structure and example.

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action (implementation or further iteration)
