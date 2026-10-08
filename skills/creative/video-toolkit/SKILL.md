---
name: video-toolkit
description: "Production pipeline for programmatic video using Remotion (React). Creates announcement videos, explainers, product demos, social clips, data stories, and music-synced content entirely from code. Covers: kinetic typography, motion graphics, audio visualization, voiceover/TTS integration, browser demo recording, chart animations, scene transitions, and headless rendering. Use when users request: making a video, announcement video, demo video, explainer video, product launch video, social media clip, animated presentation, motion graphics, kinetic text, or any video content that can be described and generated programmatically."
---

# Video Toolkit — Remotion Production Pipeline

## Creative Standard

This is video production. React is the medium; cinema is the standard.

**Before writing a single component**, articulate the creative concept. What is the mood? What story does this video tell? What makes THIS video memorable? The user's prompt is a creative brief — interpret it with directorial ambition, not literal transcription.

**First-render excellence is non-negotiable.** The output must look professional without revision rounds. If it looks like "AI-generated motion graphics" — generic gradients, default fonts, predictable timing — it is wrong. Rethink the creative concept before shipping.

**Go beyond safe defaults.** The reference catalogs provide a vocabulary of techniques. For every project, combine, modify, and invent. Stock transitions and template typography are a starting point. The viewer should see something they haven't seen before.

**Be proactively creative.** Include at least one visual moment the user didn't ask for but will appreciate — a transition effect, a data visualization, a typographic flourish that elevates the whole piece.

**Timing is everything.** Video lives and dies by rhythm. Every cut, every text reveal, every transition should land on a beat or a breath. A technically perfect video with bad timing feels amateur.

**Cohesive aesthetic.** All scenes must feel connected by a unifying visual language — shared color temperature, consistent typography, related motion vocabulary. A video where every scene uses a different font and color palette is an aesthetic failure.

## Modes

| Mode | Input | Output | Best for |
|------|-------|--------|----------|
| **Announcement** | Product/feature details | 30-60s launch/release video | Product launches, feature announcements, release videos |
| **Explainer** | Technical concept | 60-120s walkthrough | Architecture overviews, how-it-works, protocol explanations |
| **Demo** | App/tool to showcase | 30-90s screen recording + motion graphics | Product demos, tool walkthroughs, onboarding |
| **Data story** | Metrics/stats | 30-60s animated data visualization | Quarterly reports, benchmarks, before/after comparisons |
| **Social clip** | Key message | 15-30s vertical or square clip | Twitter/X, Instagram, Discord announcements |
| **Music video** | Audio track + visual concept | Duration of track | Audio-reactive visuals, lyric videos, visualizers |

## Stack

React/TypeScript project per video. Renders to MP4 headlessly via CLI.

| Layer | Tool | Purpose |
|-------|------|---------|
| Core | Remotion 4.x | React-based video framework — compositions, timeline, rendering |
| Animation | `interpolate()`, `spring()` | Frame-driven animation (CSS animations are FORBIDDEN) |
| Typography | Google Fonts, `@remotion/google-fonts` | Web font loading for kinetic text |
| Transitions | `@remotion/transitions` | Scene-to-scene effects (fade, slide, wipe, custom) |
| Audio | `@remotion/media-utils` | Audio visualization, waveform data, duration detection |
| TTS | ElevenLabs API / Qwen3-TTS (local) | AI voiceover generation |
| Recording | Playwright (via Hermes browser tool) | Browser demo capture as video assets |
| Charts | D3.js / custom React | Animated data visualizations |
| 3D | React Three Fiber (optional) | 3D scenes, product renders, globe visualizations |
| Rendering | Remotion CLI | Headless MP4/WebM/GIF output |
| Video I/O | ffmpeg | Post-processing, format conversion, audio muxing |

## Pipeline

Every mode follows the same 6-stage pipeline:

```
CONCEPT → STORYBOARD → SCENES → AUDIO → RENDER → REVIEW
```

1. **CONCEPT** — Creative vision: mood, story, color world, typography, what makes it unique
2. **STORYBOARD** — Scene breakdown with timing, transitions, and content per scene
3. **SCENES** — Write React components for each scene. Each scene is a self-contained composition
4. **AUDIO** — Generate voiceover (TTS), source music, sync timing to audio beats/pauses
5. **RENDER** — Preview frames (`npx remotion still`), then full render (`npx remotion render`)
6. **REVIEW** — Watch output, check timing, verify creative vision, iterate if needed

## Creative Direction

### Typography Styles

| Style | Technique | Best for | Reference |
|-------|-----------|----------|-----------|
| **Kinetic reveal** | Word-by-word or letter-by-letter with spring physics | Headlines, key messages | `references/text-and-typography.md` |
| **Typewriter** | Character-by-character with blinking cursor | Code snippets, terminal output | `references/text-and-typography.md` |
| **Slide-up stack** | Lines entering from bottom, stacking vertically | Feature lists, bullet points | `references/text-and-typography.md` |
| **Scale burst** | Text scales from 0 to full size with overshoot | Single-word emphasis, titles | `references/animations.md` |
| **Highlight sweep** | Colored highlight bar sweeps behind text | Key terms, callouts | `references/text-and-typography.md` |
| **Split reveal** | Text splits from center or slides in from edges | Dramatic openings, transitions | `references/text-and-typography.md` |

### Color Strategies

| Strategy | Description | Best for |
|----------|-------------|----------|
| **Dark tech** | Dark background (#0a0a0f), bright accents (cyan, purple, electric blue) | Dev tools, technical products |
| **Clean corporate** | White/light gray background, brand colors for accents | Enterprise, professional |
| **Gradient wash** | Full-screen animated gradient as background | Social clips, modern feel |
| **Monochrome + accent** | Grayscale with a single highlight color | Elegant, focused messaging |
| **Neon on dark** | Black background with glowing neon colors | Gaming, creative tools |
| **Earth + warmth** | Deep browns, amber, warm whites | Human-centric, community |

### Motion Vocabulary

| Motion | When to use | Physics |
|--------|-------------|---------|
| **Spring entrance** | Element appearing for the first time | `spring({ damping: 12, mass: 0.5 })` |
| **Fade in** | Subtle appearance, backgrounds | `interpolate(frame, [0, 20], [0, 1])` |
| **Slide from edge** | Panels, sidebars, secondary content | Spring + translateX/Y |
| **Scale from zero** | Emphasis, dramatic reveals | Spring + scale transform |
| **Stagger** | Lists, grids, multiple items | Delay each by N frames |
| **Exit up/fade** | Leaving the scene | Reverse of entrance, faster |

### Per-Scene Variation

Never use identical styling across all scenes. For each scene:
- **Different dominant color** (or shift the accent within the palette)
- **Different text animation** (kinetic reveal for headlines, typewriter for code, slide for lists)
- **Different layout** (centered, left-aligned, split-screen, full-bleed)
- **Vary transition type** between scenes (don't use the same fade everywhere)
- **Match energy to content** (fast cuts for features, slow reveals for emotional moments)

### Project-Specific Invention

For every project, create at least one of:
- A custom color palette matching the brand/mood
- A custom transition (combine built-in transitions with overlays)
- A custom typographic treatment for the key message
- A visual motif that repeats across scenes (a shape, a pattern, a color accent)
- A moment of visual surprise the user didn't request

## Workflow

### Step 1: Creative Vision

Before any code, articulate:

- **Mood/atmosphere**: Professional? Energetic? Elegant? Technical? Playful?
- **Visual story**: What's the arc? Build excitement → reveal → CTA? Problem → solution → result?
- **Color world**: Brand colors? Dark tech? Warm gradients? What's the dominant hue?
- **Typography**: Sans-serif modern? Monospace technical? Display dramatic?
- **Timing**: Fast-paced (15fps cuts)? Measured (3s per scene)? Music-synced?
- **What makes THIS different**: What's the one thing that makes this video memorable?

### Step 2: Project Setup

Start from a template or initialize fresh:

```bash
# From template (recommended)
cp -r skills/creative/video-toolkit/templates/announcement ./my-video
cd my-video && npm install

# Or fresh init
npx create-video@latest --template blank
```

Project structure:
```
my-video/
├── src/
│   ├── Root.tsx              # Register all compositions
│   ├── scenes/               # One component per scene
│   │   ├── TitleScene.tsx
│   │   ├── FeatureScene.tsx
│   │   └── CTAScene.tsx
│   └── components/           # Shared visual components
│       ├── AnimatedBg.tsx
│       └── KineticText.tsx
├── public/                   # Static assets (audio, images, fonts)
├── remotion.config.ts        # Remotion configuration
└── package.json
```

### Step 3: Build Scenes

Each scene is a React component driven by `useCurrentFrame()`:

```tsx
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

export const TitleScene: React.FC<{ title: string }> = ({ title }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scale = spring({ frame, fps, config: { damping: 12 } });
  const opacity = interpolate(frame, [0, fps * 0.5], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{
      background: "linear-gradient(135deg, #0a0a1a, #1a0a2e)",
      justifyContent: "center",
      alignItems: "center",
    }}>
      <h1 style={{
        color: "white",
        fontSize: 80,
        fontFamily: "Inter",
        transform: `scale(${scale})`,
        opacity,
      }}>
        {title}
      </h1>
    </AbsoluteFill>
  );
};
```

**Critical rules:**
- ALL animation via `useCurrentFrame()` + `interpolate()`/`spring()` — CSS animations and Tailwind animation classes are **FORBIDDEN** (they don't render in Remotion's frame-by-frame pipeline)
- Use `<AbsoluteFill>` as scene container (full-frame positioning)
- Use `<Sequence>` for timed sub-elements within a scene
- Time everything in frames, convert seconds via `fps`: `const frameAt2s = 2 * fps`

### Step 4: Audio Integration

For voiceover, generate TTS audio first, then size the composition to match:

```tsx
// In Root.tsx — use calculateMetadata prop on Composition to set dynamic duration
const calculateMyMetadata: CalculateMetadataFunction<Props> = async ({ props }) => {
  const duration = await getAudioDurationInSeconds(staticFile("voiceover.mp3"));
  return { durationInFrames: Math.ceil(duration * 30) + 60 }; // +2s padding
};

<Composition
  id="MainVideo"
  component={MainVideo}
  calculateMetadata={calculateMyMetadata}
  fps={30}
  width={1920}
  height={1080}
/>
```

For music sync, use `useWindowedAudioData()` to drive visuals. See `references/audio.md`.

### Step 5: Preview and Render

```bash
# Preview a single frame at 3 seconds (frame 90 at 30fps)
npx remotion still src/index.ts MainVideo --frame=90 --output=preview.png

# Preview at multiple timestamps
for f in 0 90 180 270; do
  npx remotion still src/index.ts MainVideo --frame=$f --output=preview_$f.png
done

# Full render
npx remotion render src/index.ts MainVideo --output=output.mp4

# Draft quality (faster)
npx remotion render src/index.ts MainVideo --output=draft.mp4 --quality=50 --scale=0.5

# Production quality
npx remotion render src/index.ts MainVideo --output=final.mp4 --codec=h264 --crf=18
```

### Step 6: Quality Verification

- **Preview frames first**: always render stills at key timestamps before full render
- **Timing check**: do cuts land where they should? Does text have time to be read?
- **Readability**: can all text be read at playback speed? Minimum 2 seconds for short text, 4 for sentences
- **Audio sync**: do visual transitions align with audio beats/pauses?
- **Creative vision check**: does the output match the concept from Step 1? If it looks generic, go back
- **Platform check**: correct aspect ratio and duration for the target platform

## Critical Implementation Notes

### CSS Animations Are Forbidden

Remotion renders frame-by-frame. CSS transitions, `@keyframes`, and Tailwind animation classes (`animate-spin`, `animate-bounce`) produce blank or static output. ALL motion must be driven by `useCurrentFrame()` + `interpolate()` or `spring()`.

### Font Loading

Use `@remotion/google-fonts` for web fonts. Fonts must be loaded before the first frame renders:

```tsx
import { loadFont } from "@remotion/google-fonts/Inter";
const { fontFamily } = loadFont();
```

Never use `@import url()` in CSS — it doesn't work in Remotion's rendering context.

### Audio Must Be in public/

Audio files referenced via `staticFile()` must be in the `public/` directory. Remotion resolves `staticFile("voiceover.mp3")` to `public/voiceover.mp3`.

### Frame Rate and Duration

Default: 30fps. Set in the Composition registration:

```tsx
<Composition id="MainVideo" component={Main}
  durationInFrames={30 * 60} // 60 seconds
  fps={30}
  width={1920} height={1080}
/>
```

Common durations: 15s social clip = 450 frames, 30s announcement = 900, 60s explainer = 1800.

### Aspect Ratios

| Platform | Ratio | Resolution |
|----------|-------|------------|
| YouTube/general | 16:9 | 1920x1080 |
| Instagram Reels / TikTok | 9:16 | 1080x1920 |
| Twitter/X | 16:9 or 1:1 | 1920x1080 or 1080x1080 |
| Square (universal) | 1:1 | 1080x1080 |

## Hermes Tool Usage

The agent builds Remotion projects using Hermes's standard tools:

| Tool | Usage |
|------|-------|
| **terminal** | `npm install`, `npx remotion render`, `npx remotion still`, ffmpeg post-processing |
| **write_file** | Create/modify `.tsx` scene files, `Root.tsx`, `package.json` |
| **read_file** | Inspect existing scenes before modification, read rendered preview images |
| **patch** | Modify specific parts of scene components (preferred over full file rewrites) |
| **browser_tool** | Record demos via Playwright, capture screenshots for video assets |

**Workflow pattern:**
1. Use `write_file` to create the project structure from template
2. Use `terminal` to run `npm install`
3. Use `write_file` / `patch` to create/modify scene components
4. Use `terminal` to render preview stills: `npx remotion still ...`
5. Use `read_file` to inspect the preview image
6. Use `patch` to adjust scenes based on preview
7. Use `terminal` to render full video: `npx remotion render ...`

**Preview-first iteration:** Always render a still frame before a full render. Preview stills take 3-10 seconds; full renders take 2-10 minutes. Inspect the preview, adjust the code, re-preview. Only do a full render when previews look correct.

## Performance Targets

| Stage | Budget |
|-------|--------|
| Scene component render | 50-200ms/frame |
| Audio data processing | 5-20ms/frame |
| Full 30s video render | 2-5 minutes |
| Full 60s video render | 4-10 minutes |
| Preview still | 3-10 seconds |

## References

| File | Contents |
|------|----------|
| `references/remotion-core.md` | Composition registration, hooks (`useCurrentFrame`, `useVideoConfig`), `<Sequence>`, `<Series>`, `<AbsoluteFill>`, static files, the rendering pipeline |
| `references/animations.md` | `interpolate()`, `spring()`, easing curves, stagger patterns, entrance/exit/emphasis animations, loop patterns |
| `references/text-and-typography.md` | Kinetic typography (word reveal, typewriter, highlight sweep), Google Fonts loading, text measurement, responsive text, captions/subtitles |
| `references/transitions.md` | `TransitionSeries`, built-in transitions (fade, slide, wipe, flip), custom transitions (glitch, RGB split, pixelate), timing functions, overlays |
| `references/audio.md` | `useWindowedAudioData()`, audio visualization (spectrum, waveform, bass-reactive), audio import/trimming, voiceover generation (ElevenLabs + Qwen3-TTS), music sync patterns |
| `references/visual-components.md` | Animated backgrounds (gradient, particle, noise), vignette, film grain, split screen, picture-in-picture, progress bars, animated shapes |
| `references/data-visualization.md` | Animated counters, bar charts, pie charts, before/after comparisons, stat cards, D3.js integration |
| `references/recording-and-demos.md` | Playwright browser recording via Hermes browser tool, screen capture as video assets, embedding recordings in compositions |
| `references/rendering-and-output.md` | CLI rendering commands, quality presets (draft/preview/production), format options (MP4/WebM/GIF), `npx remotion still` for previews, Lambda/cloud rendering |
| `references/troubleshooting.md` | Common errors (CSS animations, font loading, audio sync), ffmpeg post-processing, memory limits, debugging frame-by-frame |
