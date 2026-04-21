---
name: baoyu-comic
description: Knowledge comic creator supporting multiple art styles and tones. Creates original educational comics with detailed panel layouts and sequential image generation. Use when user asks to create "知识漫画", "教育漫画", "biography comic", "tutorial comic", or "Logicomix-style comic".
version: 1.56.1
author: 宝玉 (JimLiu)
license: MIT
metadata:
  hermes:
    tags: [comic, knowledge-comic, creative, image-generation]
    homepage: https://github.com/JimLiu/baoyu-skills#baoyu-comic
---

# Knowledge Comic Creator

Adapted from [baoyu-comic](https://github.com/JimLiu/baoyu-skills) for Hermes Agent's tool ecosystem.

Create original knowledge comics with flexible art style × tone combinations.

## When to Use

Trigger this skill when the user asks to create a knowledge/educational comic, biography comic, tutorial comic, or uses terms like "知识漫画", "教育漫画", or "Logicomix-style". The user provides content (text, file path, URL, or topic) and optionally specifies art style, tone, layout, aspect ratio, or language.

## Reference Images

Users may supply reference images to guide art style, palette, scene composition, or subject. This is **separate from** the auto-generated character sheet (Step 7.1) — both can coexist: user refs guide the look, the character sheet anchors recurring character identity.

**Intake**: Accept file paths when the user provides them (or pastes images in conversation).
- File path(s) → copy to `refs/NN-ref-{slug}.{ext}` alongside the comic output
- Pasted image with no path → ask the user for the path via `clarify`, or extract style traits verbally as a text fallback
- No reference → skip this section

**Usage modes** (per reference):

| Usage | Effect |
|-------|--------|
| `direct` | Pass the file to `image_generate` as a reference image on every page (or selected pages) |
| `style` | Extract style traits (line treatment, texture, mood) and append to every page's prompt body |
| `palette` | Extract hex colors and append to every page's prompt body |

**Record in each page's prompt frontmatter** when refs exist:

```yaml
references:
  - ref_id: 01
    filename: 01-ref-scene.png
    usage: direct
```

**At generation time**:
- Verify each referenced file exists on disk
- If `usage: direct` AND `image_generate` accepts multiple reference images → pass both the character sheet (Step 7.2) and the user refs; compress images first per Step 7.1's guidance to avoid payload failures
- If only one ref slot is available → prefer the character sheet for pages with recurring characters; embed user-ref traits in the prompt body instead
- For `style`/`palette` usage → embed extracted traits in every page's prompt text

## Options

### Visual Dimensions

| Option | Values | Description |
|--------|--------|-------------|
| Art | ligne-claire (default), manga, realistic, ink-brush, chalk, minimalist | Art style / rendering technique |
| Tone | neutral (default), warm, dramatic, romantic, energetic, vintage, action | Mood / atmosphere |
| Layout | standard (default), cinematic, dense, splash, mixed, webtoon, four-panel | Panel arrangement |
| Aspect | 3:4 (default, portrait), 4:3 (landscape), 16:9 (widescreen) | Page aspect ratio |
| Language | auto (default), zh, en, ja, etc. | Output language |
| Refs | File paths | Reference images applied to every page for style / palette / scene guidance. See [Reference Images](#reference-images) above. |

### Partial Workflow Options

| Option | Description |
|--------|-------------|
| Storyboard only | Generate storyboard only, skip prompts and images |
| Prompts only | Generate storyboard + prompts, skip images |
| Images only | Generate images from existing prompts directory |
| Regenerate N | Regenerate specific page(s) only (e.g., `3` or `2,5,8`) |

Details: [references/partial-workflows.md](references/partial-workflows.md)

### Art, Tone & Preset Catalogue

- **Art styles** (6): `ligne-claire`, `manga`, `realistic`, `ink-brush`, `chalk`, `minimalist`. Full definitions at `references/art-styles/<style>.md`.
- **Tones** (7): `neutral`, `warm`, `dramatic`, `romantic`, `energetic`, `vintage`, `action`. Full definitions at `references/tones/<tone>.md`.
- **Presets** (5) with special rules beyond plain art+tone:

  | Preset | Equivalent | Hook |
  |--------|-----------|------|
  | `ohmsha` | manga + neutral | Visual metaphors, no talking heads, gadget reveals |
  | `wuxia` | ink-brush + action | Qi effects, combat visuals, atmospheric |
  | `shoujo` | manga + romantic | Decorative elements, eye details, romantic beats |
  | `concept-story` | manga + warm | Visual symbol system, growth arc, dialogue+action balance |
  | `four-panel` | minimalist + neutral + four-panel layout | 起承转合 structure, B&W + spot color, stick-figure characters |

  Full rules at `references/presets/<preset>.md` — load the file when a preset is picked.

- **Compatibility matrix** and **content-signal → preset** table live in [references/auto-selection.md](references/auto-selection.md). Read it before recommending combinations in Step 2.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/merge-to-pdf.ts` | Merge comic pages into PDF (runs with `bun`) |

Resolve `{baseDir}` as this SKILL.md's directory; script path is `{baseDir}/scripts/merge-to-pdf.ts`.

## File Structure

Output directory: `comic/{topic-slug}/`
- Slug: 2-4 words kebab-case from topic (e.g., `alan-turing-bio`)
- Conflict: append timestamp (e.g., `turing-story-20260118-143052`)

**Contents**:
| File | Description |
|------|-------------|
| `source-{slug}.{ext}` | Source files |
| `analysis.md` | Content analysis |
| `storyboard.md` | Storyboard with panel breakdown |
| `characters/characters.md` | Character definitions |
| `characters/characters.png` | Character reference sheet |
| `prompts/NN-{cover\|page}-[slug].md` | Generation prompts |
| `NN-{cover\|page}-[slug].png` | Generated images |
| `{topic-slug}.pdf` | Final merged PDF |

## Language Handling

**Detection Priority**:
1. User-specified language (explicit option)
2. User's conversation language
3. Source content language

**Rule**: Use user's input language for ALL interactions:
- Storyboard outlines and scene descriptions
- Image generation prompts
- User selection options and confirmations
- Progress updates, questions, errors, summaries

Technical terms remain in English.

## Workflow

### Progress Checklist

```
Comic Progress:
- [ ] Step 1: Setup & Analyze
  - [ ] 1.1 Analyze content
  - [ ] 1.2 Check existing directory
- [ ] Step 2: Confirmation - Style & options ⚠️ REQUIRED
- [ ] Step 3: Generate storyboard + characters
- [ ] Step 4: Review outline (conditional)
- [ ] Step 5: Generate prompts
- [ ] Step 6: Review prompts (conditional)
- [ ] Step 7: Generate images
  - [ ] 7.1 Generate character sheet (if needed) → characters/characters.png
  - [ ] 7.2 Generate pages (with character ref if sheet exists)
- [ ] Step 8: Merge to PDF
- [ ] Step 9: Completion report
```

### Flow

```
Input → Analyze → [Check Existing?] → [Confirm: Style + Reviews] → Storyboard → [Review?] → Prompts → [Review?] → Images → PDF → Complete
```

### Step Summary

| Step | Action | Key Output |
|------|--------|------------|
| 1.1 | Analyze content | `analysis.md` |
| 1.2 | Check existing directory | Handle conflicts |
| 2 | Confirm style, focus, audience, reviews | User preferences |
| 3 | Generate storyboard + characters | `storyboard.md`, `characters/` |
| 4 | Review outline (if requested) | User approval |
| 5 | Generate prompts | `prompts/*.md` |
| 6 | Review prompts (if requested) | User approval |
| 7.1 | Generate character sheet (if needed) | `characters/characters.png` |
| 7.2 | Generate pages (with character ref if available) | `*.png` files |
| 8 | Merge to PDF | `{slug}.pdf` |
| 9 | Completion report | Summary |

### User Questions

Use the `clarify` tool to confirm options. Since `clarify` handles one question at a time, ask the most important question first and proceed sequentially. See [references/workflow.md](references/workflow.md) for the full Step 2 question set.

### Step 7: Image Generation

Use Hermes' built-in `image_generate` tool for all image rendering.

**Prompt file requirement (hard)**: write each image's full, final prompt to a standalone file under `prompts/` (naming: `NN-{type}-[slug].md`) BEFORE calling `image_generate`. The prompt file is the reproducibility record.

**Aspect ratio mapping** — `image_generate` supports `landscape`, `portrait`, and `square`. Map as follows:

| Storyboard ratio | `image_generate` format |
|------------------|-------------------------|
| `3:4`, `9:16`, `2:3` | `portrait` |
| `4:3`, `16:9`, `3:2` | `landscape` |
| `1:1` | `square` |

**7.1 Character sheet** — generate it (to `characters/characters.png`, aspect `landscape`) when the comic is multi-page with recurring characters. Skip for simple presets (e.g., four-panel minimalist) or single-page comics. Compress to JPEG before using as a reference (`sips -s format jpeg -s formatOptions 80 …` on macOS, `pngquant --quality=65-80 …` on Linux) to avoid payload failures. The prompt file at `characters/characters.md` must exist before invoking `image_generate`.

**7.2 Pages** — each page's prompt MUST already be at `prompts/NN-{cover|page}-[slug].md` before invoking `image_generate`. Strategy depends on the character sheet:

| Character sheet | `image_generate` reference support | Strategy |
|-----------------|------------------------------------|----------|
| Exists | Supported | Pass sheet as reference image on every page |
| Exists | Not supported | Prepend character descriptions to every prompt file |
| Skipped | — | All descriptions inline in prompt |

**Backup rule**: existing `prompts/…md` and `…png` files → rename with `-backup-YYYYMMDD-HHMMSS` suffix (use `write_file` / standard shell rename) before regenerating. Aspect ratio from storyboard (default `3:4`; preset may override).

**Reference failure recovery**: compress sheet → retry → still fails → drop the reference and embed character descriptions in the prompt text.

Full step-by-step workflow (analysis, storyboard, review gates, regeneration variants): [references/workflow.md](references/workflow.md).

## References

**Core Templates**:
- [analysis-framework.md](references/analysis-framework.md) - Deep content analysis
- [character-template.md](references/character-template.md) - Character definition format
- [storyboard-template.md](references/storyboard-template.md) - Storyboard structure
- [ohmsha-guide.md](references/ohmsha-guide.md) - Ohmsha manga specifics

**Style Definitions**:
- `references/art-styles/` - Art styles (ligne-claire, manga, realistic, ink-brush, chalk, minimalist)
- `references/tones/` - Tones (neutral, warm, dramatic, romantic, energetic, vintage, action)
- `references/presets/` - Presets with special rules (ohmsha, wuxia, shoujo, concept-story, four-panel)
- `references/layouts/` - Layouts (standard, cinematic, dense, splash, mixed, webtoon, four-panel)

**Workflow**:
- [workflow.md](references/workflow.md) - Full workflow details
- [auto-selection.md](references/auto-selection.md) - Content signal analysis
- [partial-workflows.md](references/partial-workflows.md) - Partial workflow options

## Page Modification

| Action | Steps |
|--------|-------|
| **Edit** | **Update prompt file FIRST** → regenerate image → regenerate PDF |
| **Add** | Create prompt at position → generate with character ref → renumber subsequent → update storyboard → regenerate PDF |
| **Delete** | Remove files → renumber subsequent → update storyboard → regenerate PDF |

**IMPORTANT**: When updating pages, ALWAYS update the prompt file (`prompts/NN-{cover|page}-[slug].md`) FIRST before regenerating. This ensures changes are documented and reproducible.

## Pitfalls

- Image generation: 10-30 seconds per page; auto-retry once on failure
- Use stylized alternatives for sensitive public figures
- **Step 2 confirmation required** - do not skip
- **Steps 4/6 conditional** - only if user requested in Step 2
- **Step 7.1 character sheet** - recommended for multi-page comics, optional for simple presets
- **Step 7.2 character reference** - pass sheet as reference if it exists; compress/convert on failure; fall back to prompt-only
- **Strip secrets** — scan source content for API keys, tokens, or credentials before writing any output file
