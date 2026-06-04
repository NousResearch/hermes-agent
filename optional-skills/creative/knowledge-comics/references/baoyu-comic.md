# baoyu-comic — Knowledge Comic Creator

> Full skill at `/home/<username>/.hermes/skills/.archive/creative/baoyu-comic/`

## Summary

Create original knowledge comics with flexible art style × tone combinations. Supports multi-page sequential narrative, character consistency, reference image trait extraction, and review gates.

## When to Use

Trigger when user asks for a knowledge/educational comic, biography comic, tutorial comic, or uses terms like "知识漫画", "教育漫画", or "Logicomix-style".

## Key Options

| Option | Values | Default |
|--------|--------|---------|
| Art | ligne-claire, manga, realistic, ink-brush, chalk, minimalist | ligne-claire |
| Tone | neutral, warm, dramatic, romantic, energetic, vintage, action | neutral |
| Layout | standard, cinematic, dense, splash, mixed, webtoon, four-panel | standard |
| Aspect | 3:4 (portrait), 4:3 (landscape), 16:9 (widescreen) | 3:4 |
| Language | auto, zh, en, ja, etc. | auto |
| Refs | File paths for style/palette/scene extraction | — |

**Presets**: `ohmsha` (manga+neutral), `wuxia` (ink-brush+action), `shoujo` (manga+romantic), `concept-story` (manga+warm), `four-panel` (minimalist+neutral)

## Workflow (8 steps)

1. **Step 1**: Analyze content → `analysis.md`, `source-{slug}.md`
2. **Step 2**: Confirm style, focus, audience, review preferences (REQUIRED; use `clarify`)
3. **Step 3**: Generate storyboard + characters → `storyboard.md`, `characters/`
4. **Step 4**: Review outline (if requested in Step 2)
5. **Step 5**: Generate prompts → `prompts/NN-{cover|page}-[slug].md`
6. **Step 6**: Review prompts (if requested in Step 2)
7. **Step 7**: Generate images — `image_generate` → download via `curl -o /absolute/path.png`
   - 7.1: Character sheet (if multi-page with recurring characters) → `characters/characters.png`
   - 7.2: Pages with character descriptions embedded from `characters/characters.md`
8. **Step 8**: Completion report

## Critical Notes

- **`image_generate` is prompt-only**: Always download the returned URL to an **absolute path** using `curl -fsSL "<url>" -o /abs/path/to/comic/<slug>/NN-page-<slug>.png`
- **Character consistency**: Driven by text descriptions in `characters/characters.md` embedded inline in every page prompt — NOT by the PNG sheet (which is a review artifact only)
- **Reference images**: Extract style/palette/scene traits as text; do NOT pass images to `image_generate`
- **Timeout handling**: If `clarify` times out, treat as default for *that question only* and surface it visibly
- **Strip secrets**: Scan source content for API keys, tokens, credentials before writing output

## Output Structure

```
comic/{topic-slug}/
├── source-{slug}.md
├── analysis.md
├── storyboard.md
├── characters/characters.md
├── characters/characters.png  (if 7.1 run)
├── prompts/NN-{cover|page}-[slug].md
├── NN-{cover|page}-[slug].png
└── refs/NN-ref-{slug}.{ext}  (optional, user-supplied)
```

## Pitfalls

- 10-30 seconds per page; auto-retry once on failure
- **Always download** the URL — downstream tooling expects local PNGs, not ephemeral URLs
- Use **absolute paths** for `curl -o` — never rely on shell CWD persistence
- Style: use stylized alternatives for sensitive public figures
- Steps 4/6 are conditional on user preference set in Step 2
- Step 2 confirmation is required — do not skip

## Reference Files (archived)

- `references/analysis-framework.md` — Deep content analysis
- `references/character-template.md` — Character definition format
- `references/storyboard-template.md` — Storyboard structure
- `references/ohmsha-guide.md` — Ohmsha manga specifics
- `references/art-styles/` — 6 art style definitions
- `references/tones/` — 7 tone definitions
- `references/presets/` — Preset special rules
- `references/layouts/` — 7 layout definitions
- `references/workflow.md` — Full workflow details
- `references/auto-selection.md` — Content signal → preset/option mapping
- `references/partial-workflows.md` — Partial workflow options
