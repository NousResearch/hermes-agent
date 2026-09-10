# Animations

**Critical rule:** ALL animation MUST be driven by `useCurrentFrame()`. CSS transitions, `@keyframes`, `transition:`, and Tailwind animation classes (`animate-spin`, `animate-bounce`, etc.) are **FORBIDDEN** — they produce blank or static frames in Remotion's renderer.

## interpolate()

Maps a frame number to any value range:

```tsx
import { interpolate, useCurrentFrame } from "remotion";

const frame = useCurrentFrame();

// Fade in over 20 frames
const opacity = interpolate(frame, [0, 20], [0, 1], {
  extrapolateRight: "clamp",
});

// Slide in from 100px left over 30 frames
const translateX = interpolate(frame, [0, 30], [-100, 0], {
  extrapolateRight: "clamp",
});
```

Always use `extrapolateRight: "clamp"` to prevent values from exceeding the target range after the animation completes.

### Easing

```tsx
import { Easing } from "remotion";

const scale = interpolate(frame, [0, 30], [0.5, 1], {
  extrapolateRight: "clamp",
  easing: Easing.bezier(0.25, 0.1, 0.25, 1), // ease-out
});
```

Common easings:
- `Easing.ease` — standard ease
- `Easing.bezier(0.33, 1, 0.68, 1)` — ease-out (most natural for entrances)
- `Easing.bezier(0.65, 0, 0.35, 1)` — ease-in-out
- `Easing.bezier(0.68, -0.6, 0.32, 1.6)` — overshoot bounce

### Time in Seconds

Always convert seconds to frames:

```tsx
const { fps } = useVideoConfig();
const opacity = interpolate(frame, [0, 0.5 * fps], [0, 1], {
  extrapolateRight: "clamp",
});
```

## spring()

Physics-based animation with overshoot and settle:

```tsx
import { spring, useCurrentFrame, useVideoConfig } from "remotion";

const frame = useCurrentFrame();
const { fps } = useVideoConfig();

const scale = spring({
  frame,
  fps,
  config: {
    damping: 12,   // higher = less bounce (default 10)
    mass: 0.5,     // lower = faster (default 1)
    stiffness: 200, // higher = snappier (default 100)
  },
});
// scale goes from 0 → ~1, with natural overshoot and settle
```

### Spring Presets

| Preset | Config | Character |
|--------|--------|-----------|
| **Snappy** | `{ damping: 15, mass: 0.4, stiffness: 250 }` | Fast, minimal bounce |
| **Bouncy** | `{ damping: 8, mass: 0.8, stiffness: 180 }` | Playful overshoot |
| **Heavy** | `{ damping: 12, mass: 2, stiffness: 120 }` | Slow, weighty |
| **Gentle** | `{ damping: 20, mass: 0.6, stiffness: 100 }` | Smooth, no overshoot |

### Delayed Spring

```tsx
const scale = spring({
  frame: frame - delayFrames, // negative frames produce 0
  fps,
  config: { damping: 12 },
});
```

## Stagger Pattern

Animate multiple items with sequential delays:

```tsx
const items = ["Feature A", "Feature B", "Feature C", "Feature D"];
const staggerDelay = 8; // frames between each item

{items.map((item, i) => {
  const entrance = spring({
    frame: frame - i * staggerDelay,
    fps,
    config: { damping: 12, mass: 0.5 },
  });

  return (
    <div key={i} style={{
      opacity: entrance,
      transform: `translateY(${interpolate(entrance, [0, 1], [30, 0])}px)`,
    }}>
      {item}
    </div>
  );
})}
```

## Common Animation Patterns

### Entrance Animations

```tsx
// Fade in
const opacity = interpolate(frame, [0, 20], [0, 1], { extrapolateRight: "clamp" });

// Slide up + fade
const y = interpolate(frame, [0, 25], [40, 0], { extrapolateRight: "clamp", easing: Easing.bezier(0.33, 1, 0.68, 1) });
const opacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: "clamp" });

// Scale in with spring
const scale = spring({ frame, fps, config: { damping: 12 } });

// Slide from left
const x = interpolate(frame, [0, 30], [-200, 0], { extrapolateRight: "clamp", easing: Easing.bezier(0.33, 1, 0.68, 1) });
```

### Exit Animations

Exits should be faster than entrances (shorter duration, no overshoot):

```tsx
const exitStart = durationInFrames - 15; // start exit 15 frames before scene ends
const exitOpacity = interpolate(frame, [exitStart, durationInFrames], [1, 0], {
  extrapolateLeft: "clamp",
  extrapolateRight: "clamp",
});
```

### Emphasis (Pulse, Glow)

```tsx
// Gentle pulse (loop)
const pulse = Math.sin(frame * 0.1) * 0.05 + 1; // oscillates 0.95 to 1.05
<div style={{ transform: `scale(${pulse})` }}>Highlighted text</div>

// Color glow cycle
const hue = (frame * 2) % 360;
<div style={{ textShadow: `0 0 20px hsl(${hue}, 80%, 60%)` }}>Glowing</div>
```

### Loop Pattern

```tsx
// Loop a 60-frame animation forever using modulo
const loopDuration = 60;
const loopedFrame = frame % loopDuration;
const rotation = interpolate(loopedFrame, [0, loopDuration], [0, 360]);
```
