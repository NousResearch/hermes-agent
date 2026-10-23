# Troubleshooting

## CSS Animations Don't Work

**Symptom:** Elements are static despite having CSS `transition`, `animation`, or Tailwind animation classes.

**Cause:** Remotion renders frame-by-frame. CSS animations run in real-time and don't advance between frames. They produce blank or frozen output.

**Fix:** Replace ALL CSS animation with `useCurrentFrame()` + `interpolate()` or `spring()`.

```tsx
// WRONG — will not animate
<div style={{ transition: "opacity 0.5s", opacity: isVisible ? 1 : 0 }}>

// WRONG — will not animate
<div className="animate-fadeIn">

// CORRECT
const opacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: "clamp" });
<div style={{ opacity }}>
```

This is the #1 most common error with Remotion.

## Font Not Loading

**Symptom:** Text renders in a default sans-serif instead of the specified font.

**Cause:** `@import url()` in CSS doesn't work in Remotion. Google Fonts must be loaded via the Remotion package.

**Fix:**
```bash
npx remotion add @remotion/google-fonts
```

```tsx
// WRONG
// @import url('https://fonts.googleapis.com/css2?family=Inter');

// CORRECT
import { loadFont } from "@remotion/google-fonts/Inter";
const { fontFamily } = loadFont();
```

## Audio Out of Sync

**Symptom:** Voiceover or music doesn't align with visual elements.

**Cause:** Usually the composition duration doesn't match the audio duration, or Sequences have wrong `from` values.

**Fix:** Use `calculateMetadata` to set duration from audio length:

```tsx
const duration = await getAudioDurationInSeconds(staticFile("voiceover.mp3"));
return { durationInFrames: Math.ceil(duration * fps) + 60 }; // +2s padding
```

For manual sync, preview at specific frames and compare audio position with visual state.

## Black / Blank Frames

**Symptom:** Some or all frames render as black.

**Causes and fixes:**

1. **Component returns null** — A condition causes the component to return nothing for some frames. Check all conditional rendering.

2. **Sequence timing off** — Content is inside a `<Sequence>` that hasn't started or has ended. Check `from` and `durationInFrames`.

3. **Z-index stacking** — A background `<AbsoluteFill>` is rendering on top of content. Check layer order (later JSX elements render on top).

4. **Opacity at 0** — An interpolation produces 0 opacity outside its range. Add `extrapolateRight: "clamp"` to prevent values going below 0.

## "Cannot find module" Errors

**Symptom:** Import errors when rendering.

**Fix:** Ensure all dependencies are installed:

```bash
npm install
npx remotion add @remotion/transitions    # if using transitions
npx remotion add @remotion/media-utils    # if using audio
npx remotion add @remotion/google-fonts   # if using fonts
npx remotion add @remotion/light-leaks    # if using light leak overlays
```

## Rendering is Slow

**Symptom:** Full render takes much longer than expected.

**Causes:**
1. **Complex DOM per frame** — Reduce the number of DOM nodes. Simplify particle systems, reduce data points in charts.
2. **Unoptimized images** — Large PNG files slow down every frame. Compress images before adding to `public/`.
3. **Re-renders** — Memoize expensive components with `React.memo()` and `useMemo()`.

**Quick fixes:**
```bash
# Draft render at half resolution
npx remotion render ... --scale=0.5 --quality=50

# Check which frames are slow (shows per-frame timing)
npx remotion render ... --log=verbose
```

## ffmpeg Issues

### "ffmpeg not found"

Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

### Audio Not Included in Output

Remotion includes audio by default. If audio is missing:
1. Check the audio file exists in `public/`
2. Check `<Audio src={staticFile("file.mp3")} />` is in the component tree
3. Check the audio Sequence timing — it might start after the composition ends

### Output File Too Large

```bash
# Re-encode with higher CRF (lower quality, smaller file)
ffmpeg -i output.mp4 -c:v libx264 -crf 28 -c:a aac -b:a 128k smaller.mp4

# Or render with higher CRF directly
npx remotion render ... --crf=28
```

## TypeScript Errors

### "Property does not exist on type"

Remotion is strongly typed. Define props interfaces for all components:

```tsx
interface SceneProps {
  title: string;
  subtitle?: string;
  accentColor?: string;
}

export const MyScene: React.FC<SceneProps> = ({ title, subtitle, accentColor = "#00d4ff" }) => {
  // ...
};
```

### "Module has no exported member"

Some Remotion imports changed between versions. Check the version:

```bash
npx remotion --version
```

Use the correct import paths for your version. v4.x is the current stable.
