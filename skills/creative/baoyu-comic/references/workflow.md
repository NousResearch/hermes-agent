# Complete Workflow

Full workflow for generating knowledge comics.

## Progress Checklist

Copy and track progress:

```
Comic Progress:
- [ ] Step 1: Setup & Analyze
  - [ ] 1.1 Analyze content
  - [ ] 1.2 Check existing ⚠️ REQUIRED
- [ ] Step 2: Confirmation - Style & options ⚠️ REQUIRED
- [ ] Step 3: Generate storyboard + characters
- [ ] Step 4: Review outline (conditional)
- [ ] Step 5: Generate prompts
- [ ] Step 6: Review prompts (conditional)
- [ ] Step 7: Generate images
  - [ ] 7.1 Character sheet (if needed)
  - [ ] 7.2 Generate pages
- [ ] Step 8: Completion report
```

## Flow Diagram

```
Input → Analyze → [Check Existing?] → [Confirm: Style + Reviews] → Storyboard → [Review Outline?] → Prompts → [Review Prompts?] → Images → Complete
```

---

## Step 1: Setup & Analyze

### 1.1 Analyze Content → `analysis.md`

Read source content, save it if needed, and perform deep analysis.

**Actions**:
1. **Save source content** (if not already a file):
   - If user provides a file path: use as-is
   - If user pastes content: save to `source-{slug}.md` in the target directory using `write_file`, where `{slug}` is the kebab-case topic slug used for the output directory
   - **Backup rule**: If `source-{slug}.md` already exists, rename it to `source-{slug}-backup-YYYYMMDD-HHMMSS.md` before writing
2. Read source content
3. **Deep analysis** following `analysis-framework.md`:
   - Target audience identification
   - Value proposition for readers
   - Core themes and narrative potential
   - Key figures and their story arcs
4. Detect source language
5. **Determine language**:
   - If user specified a language → use it
   - Else → use detected source language or user's conversation language
6. Determine recommended page count:
   - Short story: 5-8 pages
   - Medium complexity: 9-15 pages
   - Full biography: 16-25 pages
7. Analyze content signals for art/tone/layout recommendations
8. **Save to `analysis.md`** using `write_file`

**analysis.md Format**: YAML front matter (title, topic, time_span, source_language, user_language, aspect_ratio, recommended_page_count, recommended_art, recommended_tone) + sections for Target Audience, Value Proposition, Core Themes, Key Figures & Story Arcs, Content Signals, Recommended Approaches. See `analysis-framework.md` for full template.

### 1.2 Check Existing Content ⚠️ REQUIRED

**MUST execute before proceeding to Step 2.**

Check if the output directory exists (e.g., via `test -d "comic/{topic-slug}"`).

**If directory exists**, use `clarify`:

```
question: "Existing content found at comic/{topic-slug}. How to proceed?"
options:
  - "Regenerate storyboard — Keep images, regenerate storyboard and characters only"
  - "Regenerate images — Keep storyboard, regenerate images only"
  - "Backup and regenerate — Backup to {slug}-backup-{timestamp}, then regenerate all"
  - "Exit — Cancel, keep existing content unchanged"
```

Save result and handle accordingly:
- **Regenerate storyboard**: Skip to Step 3, preserve `prompts/` and images
- **Regenerate images**: Skip to Step 7, use existing prompts
- **Backup and regenerate**: Move directory, start fresh from Step 2
- **Exit**: End workflow immediately

---

## Step 2: Confirmation - Style & Options ⚠️

**Purpose**: Select visual style + decide whether to review outline before generation. **Do NOT skip.**

**Display summary first**:
- Content type + topic identified
- Key figures extracted
- Time span detected
- Recommended page count
- Language (detected or user-specified)
- **Recommended style**: [art] + [tone] (based on content signals)

**Use `clarify` one question at a time**, in priority order:

> **Timeout handling (CRITICAL)**: if `clarify` returns `"The user did not provide a response within the time limit. Use your best judgement..."`, that is a per-question default, NOT blanket consent. Continue to the next question in the sequence — do not bail out of Step 2. Then, in your next user-visible message, explicitly surface every default that was taken (e.g. `"Defaulted style → ohmsha, narrative focus → concept explanation, audience → developers (clarify timed out on all three). Say the word to redirect."`). An unreported default is indistinguishable to the user from "the agent never asked."

### Question 1: Visual Style

If a preset is recommended (see `auto-selection.md`), show it first:

```
question: "Which visual style for this comic?"
options:
  - "[preset name] preset (Recommended) — [preset description] with special rules"
  - "[recommended art] + [recommended tone] (Recommended) — Best match for your content"
  - "ligne-claire + neutral — Classic educational, Logicomix style"
  - "ohmsha preset — Educational manga with visual metaphors, gadgets, NO talking heads"
  - "Custom — Specify your own art + tone or preset"
```

**Preset vs Art+Tone**: Presets include special rules beyond art+tone. `ohmsha` = manga + neutral + visual metaphor rules + character roles + NO talking heads. Plain `manga + neutral` does NOT include these rules.

### Question 2: Narrative Focus

```
question: "What should the comic emphasize? (Pick the primary focus; mention others in a follow-up if needed)"
options:
  - "Biography/life story — Follow a person's journey through key life events"
  - "Concept explanation — Break down complex ideas visually"
  - "Historical event — Dramatize important historical moments"
  - "Tutorial/how-to — Step-by-step educational guide"
```

### Question 3: Target Audience

```
question: "Who is the primary reader?"
options:
  - "General readers — Broad appeal, accessible content"
  - "Students/learners — Educational focus, clear explanations"
  - "Industry professionals — Technical depth, domain knowledge"
  - "Children/young readers — Simplified language, engaging visuals"
```

### Question 4: Outline Review

```
question: "Do you want to review the outline before image generation?"
options:
  - "Yes, let me review (Recommended) — Review storyboard and characters before generating images"
  - "No, generate directly — Skip outline review, start generating immediately"
```

### Question 5: Prompt Review

```
question: "Review prompts before generating images?"
options:
  - "Yes, review prompts (Recommended) — Review image generation prompts before generating"
  - "No, skip prompt review — Proceed directly to image generation"
```

**After responses**:
1. Update `analysis.md` with user preferences
2. **Store `skip_outline_review`** flag based on Question 4 response
3. **Store `skip_prompt_review`** flag based on Question 5 response
4. → Step 3

---

## Step 3: Generate Storyboard + Characters

Create storyboard and character definitions using the confirmed style from Step 2.

**Loading Style References**:
- Art style: `art-styles/{art}.md`
- Tone: `tones/{tone}.md`
- If preset (ohmsha/wuxia/shoujo/concept-story/four-panel): also load `presets/{preset}.md`

**Generate**:

1. **Storyboard** (`storyboard.md`):
   - YAML front matter with art_style, tone, layout, aspect_ratio
   - Cover design
   - Each page: layout, panel breakdown, visual prompts
   - **Written in user's preferred language** (from Step 1)
   - Reference: `storyboard-template.md`
   - **If using preset**: Load and apply preset rules from `presets/`

2. **Character definitions** (`characters/characters.md`):
   - Visual specs matching the art style (in user's preferred language)
   - Include Reference Sheet Prompt for later image generation
   - Reference: `character-template.md`
   - **If using ohmsha preset**: Use default Doraemon characters (see below)

**Ohmsha Default Characters** (use these unless user specifies custom characters):

| Role | Character | Visual Description |
|------|-----------|-------------------|
| Student | 大雄 (Nobita) | Japanese boy, 10yo, round glasses, black hair parted in middle, yellow shirt, navy shorts |
| Mentor | 哆啦 A 梦 (Doraemon) | Round blue robot cat, big white eyes, red nose, whiskers, white belly with 4D pocket, golden bell, no ears |
| Challenge | 胖虎 (Gian) | Stocky boy, rough features, small eyes, orange shirt |
| Support | 静香 (Shizuka) | Cute girl, black short hair, pink dress, gentle expression |

These are the canonical ohmsha-style characters. Do NOT create custom characters for ohmsha unless explicitly requested.

**After generation**:
- If `skip_outline_review` is true → Skip Step 4, go directly to Step 5
- If `skip_outline_review` is false → Continue to Step 4

---

## Step 4: Review Outline (Conditional)

**Skip this step** if user selected "No, generate directly" in Step 2.

**Purpose**: User reviews and confirms storyboard + characters before generation.

**Display**:
- Page count and structure
- Art style + Tone combination
- Page-by-page summary (Cover → P1 → P2...)
- Character list with brief descriptions

**Use `clarify`**:

```
question: "Ready to generate images with this outline?"
options:
  - "Yes, proceed (Recommended) — Generate character sheet and comic pages"
  - "Edit storyboard first — I'll modify storyboard.md before continuing"
  - "Edit characters first — I'll modify characters/characters.md before continuing"
  - "Edit both — I'll modify both files before continuing"
```

**After response**:
1. If user wants to edit → Wait for user to finish editing, then ask again
2. If user confirms → Continue to Step 5

---

## Step 5: Generate Prompts

Create image generation prompts for all pages.

**Style Reference Loading**:
- Read `art-styles/{art}.md` for rendering guidelines
- Read `tones/{tone}.md` for mood/color adjustments
- If preset: Read `presets/{preset}.md` for special rules

**For each page (cover + pages)**:
1. Create prompt following art style + tone guidelines
2. **Embed character descriptions** inline (copy relevant traits from `characters/characters.md`) — `image_generate` is prompt-only, so the prompt text is the sole vehicle for character consistency
3. Save to `prompts/NN-{cover|page}-[slug].md` using `write_file`
   - **Backup rule**: If prompt file exists, rename to `prompts/NN-{cover|page}-[slug]-backup-YYYYMMDD-HHMMSS.md`

**Prompt File Format**:
```markdown
# Page NN: [Title]

## Visual Style
Art: [art style] | Tone: [tone] | Layout: [layout type]

## Character Reference (embedded inline — maintain exact traits below)
- [Character A]: [detailed visual traits from characters/characters.md]
- [Character B]: [detailed visual traits from characters/characters.md]

## Panel Breakdown
[From storyboard.md - panel descriptions, actions, dialogue]

## Generation Prompt
[Combined prompt passed to image_generate]
```

**After generation**:
- If `skip_prompt_review` is true → Skip Step 6, go directly to Step 7
- If `skip_prompt_review` is false → Continue to Step 6

---

## Step 6: Review Prompts (Conditional)

**Skip this step** if user selected "No, skip prompt review" in Step 2.

**Purpose**: User reviews and confirms prompts before image generation.

**Display prompt summary table**:

| Page | Title | Key Elements |
|------|-------|--------------|
| Cover | [title] | [main visual] |
| P1 | [title] | [key elements] |
| ... | ... | ... |

**Use `clarify`**:

```
question: "Ready to generate images with these prompts?"
options:
  - "Yes, proceed (Recommended) — Generate all comic page images"
  - "Edit prompts first — I'll modify prompts/*.md before continuing"
  - "Regenerate prompts — Regenerate all prompts with different approach"
```

**After response**:
1. If user wants to edit → Wait for user to finish editing, then ask again
2. If user wants to regenerate → Go back to Step 5
3. If user confirms → Continue to Step 7

---

## Step 7: Generate Images

With confirmed prompts from Step 5/6, use the `image_generate` tool. The tool accepts only `prompt` and `aspect_ratio` (`landscape` | `portrait` | `square`) and **returns a URL** — it does not accept reference images and does not write local files. Every invocation must be followed by a download step.

**Aspect ratio mapping** — map the storyboard's `aspect_ratio` to the tool's enum:

| Storyboard ratio | `image_generate` format |
|------------------|-------------------------|
| `3:4`, `9:16`, `2:3` | `portrait` |
| `4:3`, `16:9`, `3:2` | `landscape` |
| `1:1` | `square` |

**Download procedure** (run after every successful `image_generate` call):

1. Extract the `url` field from the tool result
2. Fetch it to disk, e.g. `curl -fsSL "<url>" -o comic/{slug}/<target>.png`
3. Verify the file is non-empty (`test -s <target>.png`); on failure, retry the generation once

### 7.1 Generate Character Reference Sheet (conditional)

Character sheet is recommended for multi-page comics with recurring characters, but **NOT required** for all presets.

**When to generate**:

| Condition | Action |
|-----------|--------|
| Multi-page comic with detailed/recurring characters | Generate character sheet (recommended) |
| Preset with simplified characters (e.g., four-panel minimalist) | Skip — prompt descriptions are sufficient |
| Single-page comic | Skip unless characters are complex |

**When generating**:
1. Use Reference Sheet Prompt from `characters/characters.md`
2. **Backup rule**: If `characters/characters.png` exists, rename to `characters/characters-backup-YYYYMMDD-HHMMSS.png`
3. Call `image_generate` with `landscape` format
4. Download the returned URL → save to `characters/characters.png`

**Important**: the downloaded sheet is a **human-facing review artifact** (so the user can visually verify character design) and a reference for later regenerations or manual prompt edits. It does **not** drive Step 7.2 — page prompts were already written in Step 5 from the text descriptions in `characters/characters.md`. `image_generate` cannot accept images as visual input, so the text is the sole cross-page consistency mechanism.

### 7.2 Generate Comic Pages

**Before generating any page**:
1. Confirm each prompt file exists at `prompts/NN-{cover|page}-[slug].md`
2. Confirm that each prompt has character descriptions embedded inline (see Step 5). `image_generate` is prompt-only, so the prompt text is the sole consistency mechanism.

**Page Generation Strategy**: every page prompt must embed character descriptions (sourced from `characters/characters.md`) inline. This is done during Step 5, uniformly whether or not the PNG sheet was produced in 7.1 — the PNG is only a review/regeneration aid, never a generation input.

**Example embedded prompt** (`prompts/01-page-xxx.md`):

```markdown
# Page 01: [Title]

## Character Reference (embedded inline — maintain consistency)
- 大雄：Japanese boy, round glasses, yellow shirt, navy shorts, worried expression...
- 哆啦 A 梦：Round blue robot cat, white belly, red nose, golden bell, 4D pocket...

## Page Content
[Original page prompt body — panels, dialogue, visual metaphors]
```

**For each page (cover + pages)**:
1. Read prompt from `prompts/NN-{cover|page}-[slug].md`
2. **Backup rule**: If image file exists, rename to `NN-{cover|page}-[slug]-backup-YYYYMMDD-HHMMSS.png`
3. Call `image_generate` with the prompt text and mapped aspect ratio
4. Download the returned URL → save to `NN-{cover|page}-[slug].png`
5. Report progress after each generation: "Generated X/N: [page title]"

---

## Step 8: Completion Report

```
Comic Complete!
Title: [title] | Art: [art] | Tone: [tone] | Pages: [count] | Aspect: [ratio] | Language: [lang]
Location: [path]
✓ source-{slug}.md (if content was pasted)
✓ analysis.md
✓ characters.png (if generated)
✓ 00-cover-[slug].png ... NN-page-[slug].png
```

---

## Page Modification

| Action | Steps |
|--------|-------|
| **Edit** | Update prompt → Regenerate image → Download new PNG |
| **Add** | Create prompt at position → Generate image → Download PNG → Renumber subsequent (NN+1) → Update storyboard |
| **Delete** | Remove files → Renumber subsequent (NN-1) → Update storyboard |

**File naming**: `NN-{cover|page}-[slug].png` (e.g., `03-page-enigma-machine.png`)
- Slugs: kebab-case, unique, derived from content
- Renumbering: Update NN prefix only, slugs unchanged
