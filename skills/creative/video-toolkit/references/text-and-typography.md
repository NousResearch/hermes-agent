# Text and Typography

## Font Loading

Use `@remotion/google-fonts` for web fonts. Never use `@import url()` in CSS.

```bash
npx remotion add @remotion/google-fonts
```

```tsx
import { loadFont } from "@remotion/google-fonts/Inter";
const { fontFamily } = loadFont();

// Use in components:
<p style={{ fontFamily }}>Clean sans-serif text</p>
```

Common choices by mood:
| Mood | Font | Weight |
|------|------|--------|
| Modern/tech | Inter, Space Grotesk | 400, 700 |
| Editorial | Playfair Display, Fraunces | 400, 700 |
| Monospace/code | JetBrains Mono, Fira Code | 400, 500 |
| Bold display | Bebas Neue, Anton | 400 |
| Friendly | Nunito, Poppins | 400, 600 |

## Kinetic Typography — Word-by-Word Reveal

The signature motion graphics technique. Reveal text one word at a time with spring physics:

```tsx
const KineticText: React.FC<{ text: string; startFrame?: number }> = ({
  text,
  startFrame = 0,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const words = text.split(" ");
  const framesPerWord = Math.floor(fps * 0.12); // ~4 frames per word at 30fps

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 12, justifyContent: "center" }}>
      {words.map((word, i) => {
        const wordFrame = frame - startFrame - i * framesPerWord;
        const entrance = spring({
          frame: wordFrame,
          fps,
          config: { damping: 14, mass: 0.5, stiffness: 200 },
        });

        return (
          <span
            key={i}
            style={{
              opacity: entrance,
              transform: `translateY(${interpolate(entrance, [0, 1], [20, 0])}px)`,
              fontSize: 64,
              fontWeight: 700,
              color: "white",
            }}
          >
            {word}
          </span>
        );
      })}
    </div>
  );
};
```

## Typewriter Effect

Character-by-character reveal with blinking cursor. Use string slicing, never per-character opacity:

```tsx
const Typewriter: React.FC<{ text: string; charsPerFrame?: number }> = ({
  text,
  charsPerFrame = 0.5,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const charsShown = Math.min(Math.floor(frame * charsPerFrame), text.length);
  const showCursor = frame % Math.floor(fps * 0.5) < Math.floor(fps * 0.25);
  const isComplete = charsShown >= text.length;

  return (
    <span style={{ fontFamily: "JetBrains Mono", fontSize: 32, color: "#00ff88" }}>
      {text.slice(0, charsShown)}
      {(!isComplete || showCursor) && (
        <span style={{ opacity: showCursor ? 1 : 0 }}>|</span>
      )}
    </span>
  );
};
```

## Word Highlight

Animated highlighter pen sweeping behind a word:

```tsx
const HighlightWord: React.FC<{ children: string; color?: string; delay?: number }> = ({
  children,
  color = "#FFD700",
  delay = 0,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = interpolate(frame - delay, [0, fps * 0.3], [0, 100], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.33, 1, 0.68, 1),
  });

  return (
    <span style={{ position: "relative", display: "inline-block" }}>
      <span
        style={{
          position: "absolute",
          bottom: 2,
          left: -4,
          height: "35%",
          width: `${progress}%`,
          backgroundColor: color,
          opacity: 0.4,
          borderRadius: 4,
          zIndex: -1,
        }}
      />
      {children}
    </span>
  );
};
```

## Slide-Up Stack

Lines entering from bottom, stacking vertically. Good for feature lists:

```tsx
const SlideUpStack: React.FC<{ lines: string[] }> = ({ lines }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const staggerFrames = Math.floor(fps * 0.3);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {lines.map((line, i) => {
        const lineFrame = frame - i * staggerFrames;
        const y = interpolate(lineFrame, [0, 20], [40, 0], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
          easing: Easing.bezier(0.33, 1, 0.68, 1),
        });
        const opacity = interpolate(lineFrame, [0, 15], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });

        return (
          <div key={i} style={{ opacity, transform: `translateY(${y}px)`, fontSize: 36 }}>
            {line}
          </div>
        );
      })}
    </div>
  );
};
```

## Text Measurement

Measure text to fit within containers or create responsive layouts:

```tsx
import { measureText } from "@remotion/layout-utils";

const { width, height } = measureText({
  text: "Hello World",
  fontFamily: "Inter",
  fontSize: 64,
  fontWeight: "700",
});
```

For fitting text to a container width:

```tsx
import { fitText } from "@remotion/layout-utils";

const { fontSize } = fitText({
  text: "A very long title that needs to fit",
  withinWidth: 1600,
  fontFamily: "Inter",
  fontWeight: "700",
});
```

## Captions / Subtitles

For voiceover-synced captions, use `@remotion/captions`:

```bash
npx remotion add @remotion/captions
```

Import SRT or generate from TTS timestamps. Render as animated text synced to audio. See `references/audio.md` for voiceover generation and timing extraction.

## Readability Guidelines

| Content type | Minimum display time |
|-------------|---------------------|
| Single word / title | 1.5 seconds |
| Short phrase (3-5 words) | 2 seconds |
| Full sentence | 3-4 seconds |
| Paragraph / bullet list | 5-8 seconds |
| Code snippet | 6-10 seconds |

At 30fps: 1.5s = 45 frames, 2s = 60 frames, 4s = 120 frames.
