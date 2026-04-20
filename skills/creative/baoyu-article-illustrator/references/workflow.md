# Detailed Workflow Procedures

## Step 1: Detect Reference Images

Check if the user provided reference images. Handle based on input type:

| Input Type | Action |
|------------|--------|
| Image file path provided | Copy to `{output-dir}/references/` → reference it by description in prompts |
| Image in conversation (no path) | Ask user (via `clarify`) for a file path or a description |
| User can't provide path | Extract style/palette verbally → append to prompts (no `references:` frontmatter) |

**CRITICAL**: Only add a `references:` field to prompt frontmatter if files are ACTUALLY SAVED to the `references/` subdirectory.

**If user provides a file path**:
1. Copy to `{output-dir}/references/NN-ref-{slug}.png` using `write_file`
2. Create description: `{output-dir}/references/NN-ref-{slug}.md`
3. Verify files exist (via `read_file`) before proceeding

**If user can't provide a path** (extracted verbally):
1. Analyze the image visually, extract: colors, style, composition
2. Create `{output-dir}/references/extracted-style.md` with extracted info
3. Do NOT add `references:` to prompt frontmatter
4. Instead, append extracted style/colors directly to prompt text

**Description File Format** (only when file saved):
```yaml
---
ref_id: NN
filename: NN-ref-{slug}.png
---
[User's description or auto-generated description]
```

---

## Step 2: Analyze

### 2.1 Determine Output Directory

| Input | Output Directory | Source-save path |
|-------|------------------|------------------|
| Article file path | `{article-dir}/imgs/` (default) | — (read article via `read_file`) |
| Pasted content | `illustrations/{topic-slug}/` (cwd) | `source-{slug}.{ext}` (save via `write_file`) |

If the user explicitly asked for a different layout (e.g., images in the article's folder, or an `illustrations/` subdirectory), honor that.

### 2.2 Analyze Content

| Analysis | Description |
|----------|-------------|
| Content type | Technical / Tutorial / Methodology / Narrative |
| Illustration purpose | information / visualization / imagination |
| Core arguments | 2-5 main points to visualize |
| Visual opportunities | Positions where illustrations add value |
| Recommended type | Based on content signals and purpose |
| Recommended density | Based on length and complexity |

Save analysis to `{output-dir}/analysis.md` using `write_file`.

### 2.3 Extract Core Arguments

- Main thesis
- Key concepts reader needs
- Comparisons/contrasts
- Framework/model proposed

**CRITICAL**: If the article uses metaphors (e.g., "电锯切西瓜"), do NOT illustrate literally. Visualize the **underlying concept**.

### 2.4 Identify Positions

**Illustrate**:
- Core arguments (REQUIRED)
- Abstract concepts
- Data comparisons
- Processes, workflows

**Do NOT Illustrate**:
- Metaphors literally
- Decorative scenes
- Generic illustrations

### 2.5 Analyze Reference Images (if saved in Step 1)

For each reference image:

| Analysis | Description |
|----------|-------------|
| Visual characteristics | Style, colors, composition |
| Content/subject | What the reference depicts |
| Suitable positions | Which sections match this reference |
| Style match | Which illustration types/styles align |
| Usage recommendation | `direct` / `style` / `palette` |

| Usage | When to Use |
|-------|-------------|
| `direct` | Reference matches desired output closely |
| `style` | Extract visual style characteristics only |
| `palette` | Extract color scheme only |

Note: `image_generate` does not accept reference-image inputs. For `direct` usage, describe the reference in the prompt text (composition, subject, palette) rather than passing the file itself.

---

## Step 3: Confirm Settings

Use the `clarify` tool. Since `clarify` handles one question at a time, ask the most important question first. Skip any question the user already answered in their request.

### Q1: Preset or Type (highest priority)

Based on Step 2 content analysis, recommend a preset first (sets both type & style). Look up [style-presets.md](style-presets.md) "Content Type → Preset Recommendations" table.

- [Recommended preset] — [brief: type + style + why]
- [Alternative preset] — [brief]
- Or choose type manually: infographic / scene / flowchart / comparison / framework / timeline / mixed

**If user picks a preset → skip Q3** (type & style both resolved).
**If user picks a type → Q3 is required.**

### Q2: Density

- minimal (1-2) — Core concepts only
- balanced (3-5) — Major sections
- per-section — At least 1 per section/chapter (Recommended)
- rich (6+) — Comprehensive coverage

### Q3: Style (skip if preset chosen in Q1)

Present Core Styles first:

- [Best compatible core style] (Recommended)
- [Other compatible core style 1]
- [Other compatible core style 2]
- Other (see full Style Gallery)

**Core Styles** (simplified selection):

| Core Style | Maps To | Best For |
|------------|---------|----------|
| `minimal-flat` | notion | General, knowledge sharing, SaaS |
| `sci-fi` | blueprint | AI, frontier tech, system design |
| `hand-drawn` | sketch/warm | Relaxed, reflective, casual |
| `editorial` | editorial | Processes, data, journalism |
| `scene` | warm/watercolor | Narratives, emotional, lifestyle |
| `poster` | screen-print | Opinion, editorial, cultural, cinematic |

Style selection based on Type × Style compatibility matrix ([styles.md](styles.md)).
**In Step 5**, read `styles/<style>.md` for visual elements and rendering rules.

### Q4: Palette (optional)

If the preset did not specify a palette, offer:

- Default (use style's built-in colors) (Recommended)
- `macaron` — soft pastel blocks on warm cream
- `warm` — warm earth tones, no cool colors
- `neon` — vibrant neon on dark backgrounds

**Skip if**: preset already resolved palette, or user specified a palette in the request.

See Palette Gallery in [styles.md](styles.md#palette-gallery) and full specs in `palettes/<palette>.md`.

### Q5: Image Text Language (only when ambiguous)

If the article language is different from the user's conversational language, ask which to use:
- Article language (match article content) (Recommended)
- User's conversational language

**Skip if**: languages match, or the user already specified in the request.

### Display Reference Usage (if references saved in Step 1)

When presenting the outline preview to the user, show reference assignments:

```
Reference Images:
| Ref | Filename | Recommended Usage |
|-----|----------|-------------------|
| 01 | 01-ref-diagram.png | direct → Illustration 1, 3 |
| 02 | 02-ref-chart.png | palette → Illustration 2 |
```

---

## Step 4: Generate Outline

Save as `{output-dir}/outline.md` using `write_file`:

```yaml
---
type: infographic
density: balanced
style: blueprint
image_count: 4
references:                    # Only if references provided
  - ref_id: 01
    filename: 01-ref-diagram.png
    description: "Technical diagram showing system architecture"
  - ref_id: 02
    filename: 02-ref-chart.png
    description: "Color chart with brand palette"
---

## Illustration 1

**Position**: [section] / [paragraph]
**Purpose**: [why this helps]
**Visual Content**: [what to show]
**Type Application**: [how type applies]
**References**: [01]                    # Optional: list ref_ids used
**Reference Usage**: direct             # direct | style | palette
**Filename**: 01-infographic-concept-name.png

## Illustration 2
...
```

**Backup rule**: If `outline.md` exists, rename to `outline-backup-YYYYMMDD-HHMMSS.md` before writing.

**Requirements**:
- Each position justified by content needs
- Type applied consistently
- Style reflected in descriptions
- Count matches density
- References assigned based on Step 2.5 analysis

---

## Step 5: Generate Prompts

**BLOCKING**: Every illustration must have a saved prompt file before any image is generated.

For each illustration in the outline:

1. **Create prompt file**: `{output-dir}/prompts/NN-{type}-{slug}.md` via `write_file`
2. **Include YAML frontmatter**:
   ```yaml
   ---
   illustration_id: 01
   type: infographic
   style: custom-flat-vector
   ---
   ```
3. **Load style specs**: Read `styles/<style>.md` (via `read_file`) for visual elements, style rules, and rendering instructions
4. **Load palette specs** (if palette specified): Read `palettes/<palette>.md` for colors and background. Palette colors **replace** the style's default Color Palette. If no palette specified, use the style's built-in colors.
5. **Follow type-specific template** from [prompt-construction.md](prompt-construction.md), using rendering from style + colors from palette (or style default)
6. **Prompt quality requirements** (all REQUIRED):
   - `Layout`: Describe overall composition (grid / radial / hierarchical / left-right / top-down)
   - `ZONES`: Describe each visual area with specific content, not vague descriptions
   - `LABELS`: Use **actual numbers, terms, metrics, quotes from the article** — NOT generic placeholders
   - `COLORS`: Specify hex codes from palette (or style default) with semantic meaning
   - `STYLE`: Describe line treatment, texture, mood, character rendering per style rules
   - `ASPECT`: Specify ratio (e.g., `16:9`)
7. **Apply defaults**: composition requirements, character rendering, text guidelines
8. **Backup rule**: If a prompt file exists, rename to `prompts/NN-{type}-{slug}-backup-YYYYMMDD-HHMMSS.md`

**CRITICAL - References in Frontmatter**:
- Only add `references` field if files ACTUALLY EXIST in `{output-dir}/references/` directory
- If style/palette was extracted verbally (no file), append info to prompt BODY instead
- Before writing frontmatter, verify the reference file exists

### 5.1 Process References (if references saved in Step 1)

Since `image_generate` doesn't accept reference-image inputs, convert every reference to a textual description and append it to the prompt body:

| Usage | Action |
|-------|--------|
| `direct` | Describe the reference (composition, subject, style, palette) in the prompt body |
| `style` | Append style traits to prompt: "Style: clean lines, gradient backgrounds..." |
| `palette` | Append extracted colors to prompt: "Colors: #E8756D coral, #7ECFC0 mint..." |

---

## Step 6: Generate Images

For each prompt file:

1. Read the prompt file (via `read_file`) and extract the assembled prompt
2. Map the prompt's `ASPECT` to `image_generate`'s format: `16:9` → `landscape`, `9:16` → `portrait`, `1:1` → `square`. Custom ratios → nearest named aspect.
3. Call `image_generate` with the prompt text
4. **Backup rule**: If an existing image file is present, rename to `NN-{type}-{slug}-backup-YYYYMMDD-HHMMSS.png` before writing
5. Save the resulting image to `{output-dir}/NN-{type}-{slug}.png`
6. On failure, retry once, then log and continue. After each generation, report "Generated X/N".

---

## Step 7: Finalize

### 7.1 Update Article

Insert after the corresponding paragraph, using the path relative to the article file:

| Input | Insert Path |
|-------|-------------|
| Article file path (default `imgs-subdir`) | `![description](imgs/NN-{type}-{slug}.png)` |
| Article file path (images alongside) | `![description](NN-{type}-{slug}.png)` |
| Article file path (`illustrations/` subdirectory) | `![description](illustrations/NN-{type}-{slug}.png)` |
| Pasted content | `![description](illustrations/{topic-slug}/NN-{type}-{slug}.png)` (relative to cwd) |

Alt text: concise description in the article's language.

### 7.2 Output Summary

```
Article Illustration Complete!

Article: [path]
Type: [type] | Density: [level] | Style: [style]
Location: [directory]
Images: X/N generated

Positions:
- 01-xxx.png → After "[Section]"
- 02-yyy.png → After "[Section]"

[If failures]
Failed:
- NN-zzz.png: [reason]
```
