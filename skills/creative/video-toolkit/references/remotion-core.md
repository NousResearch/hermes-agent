# Remotion Core

## Composition Registration

Every video starts with a Composition registered in `Root.tsx`:

```tsx
import { Composition } from "remotion";
import { MainVideo } from "./MainVideo";

export const RemotionRoot: React.FC = () => (
  <>
    <Composition
      id="MainVideo"
      component={MainVideo}
      durationInFrames={900}   // 30 seconds at 30fps
      fps={30}
      width={1920}
      height={1080}
      defaultProps={{
        title: "My Video",
      }}
    />
  </>
);
```

Multiple compositions can be registered for different versions (social, widescreen, preview).

## Core Hooks

### useCurrentFrame()

Returns the current frame number (0-indexed). This is the ONLY way to drive animation.

```tsx
const frame = useCurrentFrame(); // 0, 1, 2, ... durationInFrames-1
```

### useVideoConfig()

Returns composition metadata:

```tsx
const { fps, durationInFrames, width, height } = useVideoConfig();
const currentSecond = frame / fps;
```

## Layout Primitives

### AbsoluteFill

Full-frame container. Stack multiple for layered compositions:

```tsx
import { AbsoluteFill } from "remotion";

// Background layer
<AbsoluteFill style={{ backgroundColor: "#0a0a1a" }} />
// Content layer
<AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
  <h1>Title</h1>
</AbsoluteFill>
// Overlay layer
<AbsoluteFill style={{ background: "linear-gradient(transparent, rgba(0,0,0,0.5))" }} />
```

### Sequence

Time-offset a subtree. Children only render during their sequence window:

```tsx
import { Sequence } from "remotion";

// Title appears at 0s, subtitle at 1s, body at 2s
<Sequence from={0} durationInFrames={fps * 4}>
  <Title />
</Sequence>
<Sequence from={fps * 1} durationInFrames={fps * 3}>
  <Subtitle />
</Sequence>
<Sequence from={fps * 2}>
  <Body />
</Sequence>
```

Inside a `<Sequence>`, `useCurrentFrame()` resets to 0 at the sequence start. This is critical — animations within a sequence always start from frame 0 regardless of their position in the timeline.

### Series

Sequential arrangement without manual frame math:

```tsx
import { Series } from "remotion";

<Series>
  <Series.Sequence durationInFrames={90}>
    <SceneOne />
  </Series.Sequence>
  <Series.Sequence durationInFrames={120}>
    <SceneTwo />
  </Series.Sequence>
  <Series.Sequence durationInFrames={90}>
    <SceneThree />
  </Series.Sequence>
</Series>
```

## Static Files

Place assets in `public/`. Reference via `staticFile()`:

```tsx
import { staticFile, Audio, Img, Video } from "remotion";

<Audio src={staticFile("voiceover.mp3")} />
<Img src={staticFile("logo.png")} />
<Video src={staticFile("demo.mp4")} />
```

Never use relative imports for media files. Always `staticFile()`.

## Dynamic Metadata

Use `calculateMetadata` to set duration/dimensions dynamically (e.g., based on audio length):

```tsx
import { CalculateMetadataFunction } from "remotion";
import { getAudioDurationInSeconds } from "@remotion/media-utils";

export const calculateMyMetadata: CalculateMetadataFunction<Props> = async ({ props }) => {
  const duration = await getAudioDurationInSeconds(staticFile(props.audioFile));
  return {
    durationInFrames: Math.ceil(duration * 30) + 90, // +3s padding
    props,
  };
};

// In Root.tsx:
<Composition
  id="MainVideo"
  component={MainVideo}
  calculateMetadata={calculateMyMetadata}
  // durationInFrames is now dynamic
  fps={30}
  width={1920}
  height={1080}
/>
```

## Project Structure

```
my-video/
├── src/
│   ├── Root.tsx              # Composition registration
│   ├── MainVideo.tsx         # Top-level component (arranges scenes)
│   ├── scenes/               # One file per scene
│   │   ├── TitleScene.tsx
│   │   ├── FeatureScene.tsx
│   │   └── CTAScene.tsx
│   ├── components/           # Shared visual components
│   │   ├── AnimatedBg.tsx
│   │   └── KineticText.tsx
│   └── index.ts              # Entry point
├── public/                   # Static assets
│   ├── audio/
│   ├── images/
│   └── fonts/
├── remotion.config.ts
├── tsconfig.json
└── package.json
```

## Entry Point (index.ts)

```tsx
import { registerRoot } from "remotion";
import { RemotionRoot } from "./Root";

registerRoot(RemotionRoot);
```

## Remotion Config

```ts
// remotion.config.ts
import { Config } from "@remotion/cli/config";

Config.setVideoImageFormat("jpeg"); // faster preview than PNG
Config.setOverwriteOutput(true);    // don't prompt on re-render
```
