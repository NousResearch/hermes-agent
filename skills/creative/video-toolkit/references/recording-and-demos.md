# Recording and Demos

Capture browser interactions and screen recordings as video assets for Remotion compositions.

## Browser Demo Recording via Playwright

Hermes has a built-in browser tool. Use it to record browser interactions as video:

### Record a Web App Demo

```bash
# Start a screen recording via Playwright
# Hermes browser tool supports this natively
```

In a Hermes session, use the browser tool to:

1. Navigate to the target URL
2. Start video recording
3. Perform the demo interactions (clicks, typing, navigation)
4. Stop recording
5. Save the video file to `public/demos/`

### Convert Screen Recording to Remotion Asset

Once you have a recorded video file:

```tsx
import { Video, staticFile, Sequence } from "remotion";

// Embed the demo recording in a scene
const DemoScene: React.FC = () => (
  <AbsoluteFill style={{ backgroundColor: "#0a0a1a", padding: 40 }}>
    {/* Browser chrome mockup */}
    <div style={{
      borderRadius: 12,
      overflow: "hidden",
      border: "1px solid rgba(255,255,255,0.1)",
      boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
    }}>
      {/* Title bar */}
      <div style={{
        height: 36,
        backgroundColor: "#1e1e2e",
        display: "flex",
        alignItems: "center",
        padding: "0 12px",
        gap: 8,
      }}>
        <div style={{ width: 12, height: 12, borderRadius: "50%", backgroundColor: "#ff5f56" }} />
        <div style={{ width: 12, height: 12, borderRadius: "50%", backgroundColor: "#ffbd2e" }} />
        <div style={{ width: 12, height: 12, borderRadius: "50%", backgroundColor: "#27c93f" }} />
        <div style={{
          flex: 1, height: 24, borderRadius: 6,
          backgroundColor: "rgba(255,255,255,0.05)",
          marginLeft: 12, display: "flex", alignItems: "center",
          padding: "0 12px", fontSize: 12, color: "rgba(255,255,255,0.3)",
        }}>
          https://your-app.com
        </div>
      </div>
      {/* Video content */}
      <Video src={staticFile("demos/app-demo.mp4")} style={{ width: "100%" }} />
    </div>
  </AbsoluteFill>
);
```

### Zoom and Pan on Demo Video

Highlight specific parts of a recording by zooming in:

```tsx
const ZoomedDemo: React.FC<{
  src: string;
  zoomRegion: { x: number; y: number; scale: number };
  zoomAt: number; // frame to start zoom
  zoomDuration?: number; // frames
}> = ({ src, zoomRegion, zoomAt, zoomDuration = 30 }) => {
  const frame = useCurrentFrame();

  const zoomProgress = interpolate(frame, [zoomAt, zoomAt + zoomDuration], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.33, 1, 0.68, 1),
  });

  const scale = interpolate(zoomProgress, [0, 1], [1, zoomRegion.scale]);
  const x = interpolate(zoomProgress, [0, 1], [0, -zoomRegion.x]);
  const y = interpolate(zoomProgress, [0, 1], [0, -zoomRegion.y]);

  return (
    <div style={{ overflow: "hidden", width: "100%", height: "100%" }}>
      <Video
        src={staticFile(src)}
        style={{
          transform: `scale(${scale}) translate(${x}px, ${y}px)`,
          transformOrigin: "top left",
        }}
      />
    </div>
  );
};
```

## Screenshot Assets

Use Hermes browser tool to capture screenshots, then embed as images:

```tsx
import { Img, staticFile } from "remotion";

<Img src={staticFile("screenshots/feature-highlight.png")}
  style={{ width: "80%", borderRadius: 12, boxShadow: "0 20px 60px rgba(0,0,0,0.5)" }}
/>
```

## Phone / Device Mockups

Wrap screenshots in device frames:

```tsx
const PhoneMockup: React.FC<{ screenshot: string }> = ({ screenshot }) => (
  <div style={{
    width: 300, height: 650,
    borderRadius: 36,
    border: "4px solid #333",
    overflow: "hidden",
    backgroundColor: "#000",
    padding: 8,
    boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
  }}>
    <div style={{ borderRadius: 28, overflow: "hidden", width: "100%", height: "100%" }}>
      <Img src={staticFile(screenshot)} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
    </div>
  </div>
);
```

## Terminal / CLI Mockup

For recording CLI tool demos, embed terminal-style output:

```tsx
const TerminalMockup: React.FC<{
  lines: { text: string; delay: number }[];
}> = ({ lines }) => {
  const frame = useCurrentFrame();

  return (
    <div style={{
      backgroundColor: "#1a1a2e",
      borderRadius: 12,
      padding: 24,
      fontFamily: "JetBrains Mono",
      fontSize: 16,
      color: "#e0e0e0",
      lineHeight: 1.6,
      border: "1px solid rgba(255,255,255,0.1)",
    }}>
      {/* Terminal header */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <div style={{ width: 12, height: 12, borderRadius: "50%", backgroundColor: "#ff5f56" }} />
        <div style={{ width: 12, height: 12, borderRadius: "50%", backgroundColor: "#ffbd2e" }} />
        <div style={{ width: 12, height: 12, borderRadius: "50%", backgroundColor: "#27c93f" }} />
      </div>
      {/* Lines with typewriter effect */}
      {lines.map((line, i) => {
        const lineFrame = frame - line.delay;
        if (lineFrame < 0) return null;
        const charsShown = Math.min(Math.floor(lineFrame * 0.8), line.text.length);
        return (
          <div key={i}>
            <span style={{ color: "#888" }}>$ </span>
            {line.text.slice(0, charsShown)}
          </div>
        );
      })}
    </div>
  );
};
```
