# Complete Workflow

Full workflow for generating knowledge comics.

## Progress Checklist

Copy and track progress:

```
Comic Progress:
- [ ] Step 1: Setup & Analyze
  - [ ] 1.1 Load preferences
  - [ ] 1.2 Analyze content
  - [ ] 1.3 Check existing ⚠️ REQUIRED
- [ ] Step 2: Confirmation 1 - Style & options ⚠️ REQUIRED
- [ ] Step 3: Generate storyboard + characters
- [ ] Step 4: Review outline (conditional)
- [ ] Step 5: Generate prompts
- [ ] Step 6: Review prompts (conditional)
- [ ] Step 7: Generate images
  - [ ] 7.1 Character sheet (if needed)
  - [ ] 7.2 Generate pages
- [ ] Step 8: Merge to PDF
- [ ] Step 9: Completion report
```

## Flow Diagram

```
Input → Preferences → Analyze → [Check Existing?] → [Confirm 1: Style + Reviews] → Storyboard → [Review Outline?] → Prompts → [Review Prompts?] → Images → PDF → Complete
```

---

## Step 1: Setup & Analyze

### 1.1 Load Preferences (EXTEND.md)

Check EXTEND.md existence (priority order):

```bash
# macOS, Linux, WSL, Git Bash
test -f .baoyu-skills/baoyu-comic/EXTEND.md && echo "project"
test -f "${XDG_CONFIG_HOME:-$HOME/.config}/baoyu-skills/baoyu-comic/EXTEND.md" && echo "xdg"
test -f "$HOME/.baoyu-skills/baoyu-comic/EXTEND.md" && echo "user"
```

```powershell
# PowerShell (Windows)
if (Test-Path .baoyu-skills/baoyu-comic/EXTEND.md) { "project" }
$xdg = if ($env:XDG_CONFIG_HOME) { $env:XDG_CONFIG_HOME } else { "$HOME/.config" }
if (Test-Path "$xdg/baoyu-skills/baoyu-comic/EXTEND.md") { "xdg" }
if (Test-Path "$HOME/.baoyu-skills/baoyu-comic/EXTEND.md") { "user" }
```

| Path | Location |
|------|----------|
| `.baoyu-skills/baoyu-comic/EXTEND.md` | Project directory |
| `$HOME/.baoyu-skills/baoyu-comic/EXTEND.md` | User home |

**When EXTEND.md Found** → Read, parse, **output summary to user**:

```
📋 Loaded preferences from [full path]
├─ Watermark: [enabled/disabled] [content if enabled]
├─ Art Style: [style name or "auto-select"]
├─ Tone: [tone name or "auto-select"]
├─ Layout: [layout or "auto-select"]
├─ Language: [language or "auto-detect"]
└─ Character presets: [count] defined
```

**MUST output this summary** so user knows their current configuration. Do not skip or silently load.

**When EXTEND.md Not Found** → First-time setup:

1. Inform user: "No preferences found. Let's set up your defaults."
2. Use AskUserQuestion to collect preferences (see `config/first-time-setup.md`)
3. Create EXTEND.md at user-chosen location
4. Confirm: "✓ Preferences saved to [path]"

**EXTEND.md Supports**: Watermark | Preferred art/tone/layout | Custom style definitions | Character presets | Language preference

Schema: `config/preferences-schema.md`

**Important**: Once EXTEND.md exists, watermark, language, and style defaults are NOT asked again in Confirmation 1 or 2. These are session-persistent settings.

### 1.2 Analyze Content → `analysis.md`

Read source content, save it if needed, and perform deep analysis.

**Actions**:
1. **Save source content** (if not already a file):
   - If user provides a file path: use as-is
   - If user pastes content: save to `source.md` in target directory
   - **Backup rule**: If `source.md` exists, rename to `source-backup-YYYYMMDD-HHMMSS.md`
2. Read source content
3. **Deep analysis** following `analysis-framework.md`:
   - Target audience identification
   - Value proposition for readers
   - Core themes and narrative potential
   - Key figures and their story arcs
4. Detect source language
5. **Determine language**:
   - If EXTEND.md has `language` → use it
   - Else if `--lang` option provided → use it
   - Else → use detected source language
6. Determine recommended page count:
   - Short story: 5-8 pages
   - Medium complexity: 9-15 pages
   - Full biography: 16-25 pages
7. Analyze content signals for art/tone/layout recommendations
8. **Save to `analysis.md`**

**analysis.md Format**: YAML front matter (title, topic, time_span, source_language, user_language, aspect_ratio, recommended_page_count, recommended_art, recommended_tone) + sections for Target Audience, Value Proposition, Core Themes, Key Figures & Story Arcs, Content Signals, Recommended Approaches. See `analysis-framework.md` for full template.

### 1.3 Check Existing Content ⚠️ REQUIRED

**MUST execute before proceeding to Step 2.**

Use Bash to check if output directory exists:

```bash
test -d "comic/{topic-slug}" && echo "exists"
```

**If directory exists**, use AskUserQuestion:

```
header: "Existing"
question: "Existing content found. How to proceed?"
options:
  - label: "Regenerate storyboard"
    description: "Keep images, regenerate storyboard and characters only"
  - label: "Regenerate images"
    description: "Keep storyboard, regenerate images only"
  - label: "Backup and regenerate"
    description: "Backup to {slug}-backup-{timestamp}, then regenerate all"
  - label: "Exit"
    description: "Cancel, keep existing content unchanged"
```

Save result and handle accordingly:
- **Regenerate storyboard**: Skip to Step 3, preserve `prompts/` and images
- **Regenerate images**: Skip to Step 7, use existing prompts
- **Backup and regenerate**: Move directory, start fresh from Step 2
- **Exit**: End workflow immediately

---

## Step 2: Confirmation 1 - Style & Options ⚠️

**Purpose**: Select visual style + decide whether to review outline before generation. **Do NOT skip.**

**Note**: Watermark and language already configured in EXTEND.md (Step 1).

**Display summary**:
- Content type + topic identified
- Key figures extracted
- Time span detected
- Recommended page count
- Language: [from EXTEND.md or detected]
- **Recommended style**: [art] + [tone] (based on content signals)

**Use AskUserQuestion** for:

### Question 1: Visual Style

If a preset is recommended (see `auto-selection.md`), show it first:

```
header: "Style"
question: "Which visual style for this comic?"
options:
  - label: "[preset name] preset (Recommended)"       # If preset recommended
    description: "[preset description] - includes special rules"
  - label: "[recommended art] + [recommended tone] (Recommended)"  # If no preset
    description: "Best match for your content based on analysis"
  - label: "ligne-claire + neutral"
    description: "Classic educational, Logicomix style"
  - label: "ohmsha preset"
    description: "Educational manga with visual metaphors, gadgets, NO talking heads"
  - label: "Custom"
    description: "Specify your own art + tone or preset"
```

**Preset vs Art+Tone**: Presets include special rules beyond art+tone. `ohmsha` = manga + neutral + visual metaphor rules + character roles + NO talking heads. Plain `manga + neutral` does NOT include these rules.

### Question 2: Narrative Focus (multiSelect: true)

```
header: "Focus"
question: "What should the comic emphasize? (Select all that apply)"
options:
  - label: "Biography/life story"
    description: "Follow a person's journey through key life events"
  - label: "Concept explanation"
    description: "Break down complex ideas visually"
  - label: "Historical event"
    description: "Dramatize important historical moments"
  - label: "Tutorial/how-to"
    description: "Step-by-step educational guide"
```

### Question 3: Target Audience

```
header: "Audience"
question: "Who is the primary reader?"
options:
  - label: "General readers"
    description: "Broad appeal, accessible content"
  - label: "Students/learners"
    description: "Educational focus, clear explanations"
  - label: "Industry professionals"
    description: "Technical depth, domain knowledge"
  - label: "Children/young readers"
    description: "Simplified language, engaging visuals"
```

### Question 4: Outline Review

```
header: "Review"
question: "Do you want to review the outline before image generation?"
options:
  - label: "Yes, let me review (Recommended)"
    description: "Review storyboard and characters before generating images"
  - label: "No, generate directly"
    description: "Skip outline review, start generating immediately"
```

### Question 5: Prompt Review

```
header: "Prompts"
question: "Review prompts before generating images?"
options:
  - label: "Yes, review prompts (Recommended)"
    description: "Review image generation prompts before generating"
  - label: "No, skip prompt review"
    description: "Proceed directly to image generation"
```

**After response**:
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
- If preset (ohmsha/wuxia/shoujo): also load `presets/{preset}.md`

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

**Ohmsha Default Characters** (use these unless user specifies `--characters`):

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

**Use AskUserQuestion**:

```
header: "Confirm"
question: "Ready to generate images with this outline?"
options:
  - label: "Yes, proceed (Recommended)"
    description: "Generate character sheet and comic pages"
  - label: "Edit storyboard first"
    description: "I'll modify storyboard.md before continuing"
  - label: "Edit characters first"
    description: "I'll modify characters/characters.md before continuing"
  - label: "Edit both"
    description: "I'll modify both files before continuing"
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
2. Include character visual descriptions for consistency
3. Save to `prompts/NN-{cover|page}-[slug].md`
   - **Backup rule**: If prompt file exists, rename to `prompts/NN-{cover|page}-[slug]-backup-YYYYMMDD-HHMMSS.md`

**Prompt File Format**:
```markdown
# Page NN: [Title]

## Visual Style
Art: [art style] | Tone: [tone] | Layout: [layout type]

## Character Reference
[Character descriptions from characters/characters.md]

## Panel Breakdown
[From storyboard.md - panel descriptions, actions, dialogue]

## Generation Prompt
[Combined prompt for image generation skill]
```

**Watermark Application** (if enabled in preferences):
Add to each prompt:
```
Include a subtle watermark "[content]" positioned at [position]. The watermark should
be legible but not distracting from the comic panels and storytelling.
Ensure watermark does not overlap speech bubbles or key action.
```
Reference: `config/watermark-guide.md`

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

**Use AskUserQuestion**:

```
header: "Confirm"
question: "Ready to generate images with these prompts?"
options:
  - label: "Yes, proceed (Recommended)"
    description: "Generate all comic page images"
  - label: "Edit prompts first"
    description: "I'll modify prompts/*.md before continuing"
  - label: "Regenerate prompts"
    description: "Regenerate all prompts with different approach"
```

**After response**:
1. If user wants to edit → Wait for user to finish editing, then ask again
2. If user wants to regenerate → Go back to Step 5
3. If user confirms → Continue to Step 7

---

## Step 7: Generate Images

With confirmed prompts from Step 5/6:

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
3. Generate → `characters/characters.png`
4. **Compress** to reduce API payload size when used as `--ref`:
   - `sips -s format jpeg -s formatOptions 80 characters.png --out characters-compressed.jpg` (macOS)
   - Or: `pngquant --quality=65-80 characters.png -o characters-compressed.png`

### 7.2 Generate Comic Pages

**Before generating any page**:
1. Read the image generation skill's SKILL.md
2. Check if it supports reference image input (`--ref`, `--reference`, etc.)
3. Determine if character sheet exists
4. Choose the appropriate strategy below

**Page Generation Strategy**:

| Character Sheet | Skill Capability | Strategy |
|-----------------|------------------|----------|
| Exists | Supports `--ref` | **A**: Pass character sheet as `--ref` with every page |
| Exists | No `--ref` support | **B**: Embed character descriptions in every prompt |
| Skipped | — | **C**: Prompt file contains all descriptions inline |

**Strategy A: Using `--ref` parameter** (e.g., baoyu-imagine)

- Read the chosen image generation skill's `SKILL.md`
- Invoke that installed skill via its documented interface, not by calling its scripts directly
- For every page, use `prompts/01-page-xxx.md` as the prompt-file input
- Save output to `01-page-xxx.png`
- Use aspect ratio from storyboard (default `3:4`, preset may override)
- Pass `characters/characters.png` (or compressed version) as `--ref`

**`--ref` failure recovery**:
If generation fails when using `--ref`:
1. **Compress/convert** reference image:
   - `sips -s format jpeg -s formatOptions 70 characters.png --out characters-compressed.jpg`
   - Or reduce resolution: `sips -Z 1024 characters.png --out characters-small.png`
2. **Retry** with compressed/converted image as `--ref`
3. **If still fails**: Fall back to **Strategy C** — generate WITHOUT `--ref`, with character descriptions embedded in prompt text

**Strategy B: Embedding character descriptions in prompt**

When skill does NOT support reference images, create combined prompt files:

```markdown
# prompts/01-page-xxx.md (with embedded character reference)

## Character Reference (maintain consistency)
[Copy relevant sections from characters/characters.md here]
- 大雄：Japanese boy, round glasses, yellow shirt, navy shorts...
- 哆啦 A 梦：Round blue robot cat, white belly, red nose, golden bell...

## Page Content
[Original page prompt here]
```

**Strategy C: Prompt-only (no character sheet)**

When character sheet was skipped or `--ref` failed:
- Prompt file already contains all character descriptions inline
- No `--ref` parameter needed
- Rely on detailed text descriptions for character consistency

**For each page (cover + pages)**:
1. Read prompt from `prompts/NN-{cover|page}-[slug].md`
2. **Backup rule**: If image file exists, rename to `NN-{cover|page}-[slug]-backup-YYYYMMDD-HHMMSS.png`
3. Generate image using Strategy A, B, or C
4. Save to `NN-{cover|page}-[slug].png`
5. Report progress after each generation: "Generated X/N: [page title]"

**Session Management**:
If image generation skill supports `--sessionId`:
1. Generate unique session ID: `comic-{topic-slug}-{timestamp}`
2. Use same session ID for all pages
3. Ensures visual consistency across generated images

---

## Step 8: Merge to PDF

After all images generated:

```bash
${BUN_X} {baseDir}/scripts/merge-to-pdf.ts <comic-dir>
```

Creates `{topic-slug}.pdf` with all pages as full-page images.

---

## Step 9: Completion Report

```
Comic Complete!
Title: [title] | Art: [art] | Tone: [tone] | Pages: [count] | Aspect: [ratio] | Language: [lang]
Watermark: [enabled/disabled]
Location: [path]
✓ analysis.md
✓ characters.png (if generated)
✓ 00-cover-[slug].png ... NN-page-[slug].png
✓ {topic-slug}.pdf
```

---

## Page Modification

| Action | Steps |
|--------|-------|
| **Edit** | Update prompt → Regenerate image → Regenerate PDF |
| **Add** | Create prompt at position → Generate image → Renumber subsequent (NN+1) → Update storyboard → Regenerate PDF |
| **Delete** | Remove files → Renumber subsequent (NN-1) → Update storyboard → Regenerate PDF |

**File naming**: `NN-{cover|page}-[slug].png` (e.g., `03-page-enigma-machine.png`)
- Slugs: kebab-case, unique, derived from content
- Renumbering: Update NN prefix only, slugs unchanged
