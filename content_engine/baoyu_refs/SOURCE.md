# Baoyu Reference Files — Source Attribution

## Upstream
The dimensional prompt method comes from
[JimLiu/baoyu-skills](https://github.com/JimLiu/baoyu-skills) (v1.57.0).

## On-box skill (source of truth at runtime)
The `baoyu-article-illustrator` skill is installed on this machine at
`~/.hermes/skills/creative/baoyu-article-illustrator/`. Its `SKILL.md`
frontmatter declares `license: MIT`. The 23 style files and 4 palette
files under `references/` are the upstream references verbatim and are
re-read at runtime by `baoyu_loader.py` (falling back to the vendored
copies in `baoyu_refs/` if the on-box skill dir is absent).

No upstream binaries are vendored. The vendored markdown refs are
attribution copies of the on-box MIT content; running against the
on-box refs is the preferred path.

## Sourcing
The vendored files in `baoyu_refs/skills/baoyu-article-illustrator/
references/` are verbatim copies of the on-box skill's references. The
on-box skill's own provenance is the upstream project.

## Files
- `references/styles/*.md` (23 files): per-style design aesthetic,
  colour palette table, visual elements, do/don't rules, best-for uses
- `references/palettes/*.md` (4 files): per-palette colour roles with
  hex codes and semantic constraints
- `references/styles.md`, `style-presets.md`, `prompt-construction.md`:
  the dimensional method, preset map (type × style × palette), and
  prompt construction rules

## Updating
The on-box skill is the source of truth. To pull upstream changes,
diff against the on-box refs first, then mirror if they have changed:

```bash
# Compare on-box vs vendored for a single style.
diff ~/.hermes/skills/creative/baoyu-article-illustrator/references/styles/editorial.md \
     baoyu_refs/skills/baoyu-article-illustrator/references/styles/editorial.md

# Mirror after a confirmed licence update.
cp ~/.hermes/skills/creative/baoyu-article-illustrator/references/styles/*.md \
   content_engine/baoyu_refs/skills/baoyu-article-illustrator/references/styles/
```

## Style and palette reference (canonical, from upstream)

### Types (7)
infographic, comparison, framework, flowchart, timeline, scene, hero

### Available styles (23)
blueprint, chalkboard, editorial, elegant, fantasy-animation, flat-doodle,
flat, ink-notes, intuition-machine, minimal, nature, notion, pixel-art,
playful, retro, scientific, screen-print, sketch, sketch-notes,
vector-illustration, vintage, warm, watercolor

### Available palettes (4)
macaron, mono-ink, neon, warm

## Preset recommendations (excerpt — full table in style-presets.md)

| Preset | Type | Style | Palette | Best for |
|---|---|---|---|---|
| `tech-explainer` | infographic | blueprint | — | API docs, technical deep-dives |
| `system-design` | framework | blueprint | — | Architecture, system design |
| `ink-notes-compare` | comparison | ink-notes | mono-ink | Before/After, OS-style comparisons |
| `ink-notes-flow` | flowchart | ink-notes | mono-ink | Process explainers, hand-drawn walkthroughs |
| `ink-notes-framework` | framework | ink-notes | mono-ink | System analogies, command-centre diagrams |
| `warm-knowledge` | infographic | vector-illustration | warm | Product showcases, team intros |
| `opinion-piece` | scene | screen-print | — | Op-eds, commentary |
| `cinematic` | scene | screen-print | — | Dramatic narratives, cultural essays |
| `storytelling` | scene | warm | — | Personal essays, reflections |
| `lifestyle` | scene | watercolor | — | Travel, wellness, lifestyle, creative |
