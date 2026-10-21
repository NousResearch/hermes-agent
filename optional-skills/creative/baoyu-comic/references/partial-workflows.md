# Partial Workflows

Options to run specific parts of the workflow. Trigger these via natural language (e.g., "just the storyboard", "regenerate page 3").

## Options Summary

| Option | Steps Executed | Output |
|--------|----------------|--------|
| Storyboard only | 1-3 | `storyboard.md` + `characters/` |
| Prompts only | 1-5 | + `prompts/*.md` |
| Images only | 7-8 | + images |
| Regenerate N | 7 (partial) | Specific page(s) |

---

## Storyboard-only

Generate storyboard and characters without prompts or images.

**User cue**: "storyboard only", "just the outline", "don't generate images yet".

**Workflow**: Steps 1-3 only (stop after storyboard + characters)

**Output**:
- `analysis.md`
- `storyboard.md`
- `characters/characters.md`

**Use case**: Review and edit the storyboard before generating images. Useful for:
- Getting feedback on the narrative structure
- Making manual adjustments to panel layouts
- Defining custom characters

---

## Prompts-only

Generate storyboard, characters, and prompts without images.

**User cue**: "prompts only", "write the prompts but don't generate yet".

**Workflow**: Steps 1-5 (generate prompts, skip images)

**Output**:
- `analysis.md`
- `storyboard.md`
- `characters/characters.md`
- `prompts/*.md`

**Use case**: Review and edit prompts before image generation. Useful for:
- Fine-tuning image generation prompts
- Ensuring visual consistency before committing to generation
- Making style adjustments at the prompt level

---

## Images-only

Generate images from existing prompts (starts at Step 7).

**User cue**: "generate images from existing prompts", "run the images now" (pointing at an existing `comic/topic-slug/` directory).

**Workflow**: Skip to Step 7, then 8

**Prerequisites** (must exist in directory):
- `prompts/` directory with page prompt files
- `storyboard.md` with style information
- `characters/characters.md` with character definitions

**Output**:
- `characters/characters.png` (if not exists)
- `NN-{cover|page}-[slug].png` images

**Use case**: Re-generate images after editing prompts. Useful for:
- Recovering from failed image generation
- Trying different image generation settings
- Regenerating after manual prompt edits

---

## Regenerate

Regenerate specific pages only.

**User cue**: "regenerate page 3", "redo pages 2, 5, 8", "regenerate the cover".

**Workflow**:
1. Read existing prompts for specified pages
2. Regenerate images only for those pages via `image_generate`
3. Download each returned URL and overwrite the existing PNG

**Prerequisites** (must exist):
- `prompts/NN-{cover|page}-[slug].md` for specified pages
- `characters/characters.md` (for agent-side consistency checks, if it was used originally)

**Output**:
- Regenerated `NN-{cover|page}-[slug].png` for specified pages

**Use case**: Fix specific pages without regenerating entire comic. Useful for:
- Fixing a single problematic page
- Iterating on specific visuals
- Regenerating pages after prompt edits

**Page numbering**:
- `0` = Cover page
- `1-N` = Content pages
