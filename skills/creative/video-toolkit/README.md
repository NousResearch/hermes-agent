# Video Toolkit

Production pipeline for programmatic video using [Remotion](https://remotion.dev) (React). Create announcement videos, explainers, product demos, social clips, data stories, and music-synced content entirely from code.

## Quick Start

```bash
# Check prerequisites
bash skills/creative/video-toolkit/scripts/setup.sh

# Start from a template
cp -r skills/creative/video-toolkit/templates/announcement ./my-video
cd my-video
npm install

# Edit src/Root.tsx to customize content
# Preview in browser
npx remotion studio

# Render to MP4
npm run render
```

## Templates

| Template | Duration | Description |
|----------|----------|-------------|
| `announcement` | 30s | Product/feature announcement — title, features, CTA |
| `explainer` | 60s | Technical explainer — problem, solution, result |

## Modes

| Mode | Best for |
|------|----------|
| **Announcement** | Product launches, feature releases, version updates |
| **Explainer** | Architecture overviews, how-it-works, protocol explanations |
| **Demo** | Product demos, tool walkthroughs, onboarding |
| **Data story** | Benchmarks, quarterly reports, before/after comparisons |
| **Social clip** | Twitter/X, Instagram, Discord announcements (15-30s) |
| **Music video** | Audio-reactive visuals, lyric videos, visualizers |

## Stack

- **Remotion 4.x** — React-based video framework
- **Node.js 18+** — runtime
- **ffmpeg** — post-processing (optional)
- **ElevenLabs / Qwen3-TTS** — voiceover generation (optional)

## References

Detailed documentation in `references/`:

- `remotion-core.md` — Compositions, hooks, layout, rendering pipeline
- `animations.md` — interpolate, spring, easing, stagger patterns
- `text-and-typography.md` — Kinetic typography, typewriter, font loading
- `transitions.md` — Scene transitions, overlays, custom effects
- `audio.md` — Audio visualization, voiceover/TTS, music sync
- `visual-components.md` — Backgrounds, overlays, layout components
- `data-visualization.md` — Charts, counters, stat comparisons
- `recording-and-demos.md` — Browser recording, mockups, terminal demos
- `rendering-and-output.md` — CLI rendering, quality presets, formats
- `troubleshooting.md` — Common errors and fixes
