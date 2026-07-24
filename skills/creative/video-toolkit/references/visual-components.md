# Visual Components

Reusable visual building blocks for scene backgrounds, overlays, and effects.

## Animated Backgrounds

### Gradient Animation

```tsx
const AnimatedGradient: React.FC<{
  colors?: [string, string, string];
  speed?: number;
}> = ({ colors = ["#0a0a2e", "#1a0a3e", "#0a1a3e"], speed = 0.5 }) => {
  const frame = useCurrentFrame();
  const angle = (frame * speed) % 360;

  return (
    <AbsoluteFill
      style={{
        background: `linear-gradient(${angle}deg, ${colors[0]}, ${colors[1]}, ${colors[2]})`,
      }}
    />
  );
};
```

### Floating Particles

```tsx
const Particles: React.FC<{ count?: number; color?: string }> = ({
  count = 30,
  color = "rgba(255,255,255,0.15)",
}) => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();

  // Generate deterministic particles from seed (same every render)
  const particles = useMemo(
    () =>
      Array.from({ length: count }).map((_, i) => ({
        x: ((i * 7919) % 1000) / 1000, // pseudo-random but deterministic
        y: ((i * 6271) % 1000) / 1000,
        size: 2 + ((i * 3571) % 6),
        speed: 0.3 + ((i * 4219) % 500) / 1000,
      })),
    [count]
  );

  return (
    <AbsoluteFill>
      {particles.map((p, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            left: p.x * width,
            top: ((p.y * height + frame * p.speed) % (height + 20)) - 10,
            width: p.size,
            height: p.size,
            borderRadius: "50%",
            backgroundColor: color,
          }}
        />
      ))}
    </AbsoluteFill>
  );
};
```

### Grid Pattern

```tsx
const GridBackground: React.FC<{
  cellSize?: number;
  color?: string;
  animated?: boolean;
}> = ({ cellSize = 40, color = "rgba(255,255,255,0.05)", animated = true }) => {
  const frame = useCurrentFrame();
  const offset = animated ? (frame * 0.5) % cellSize : 0;

  return (
    <AbsoluteFill
      style={{
        backgroundImage: `
          linear-gradient(${color} 1px, transparent 1px),
          linear-gradient(90deg, ${color} 1px, transparent 1px)
        `,
        backgroundSize: `${cellSize}px ${cellSize}px`,
        backgroundPosition: `${offset}px ${offset}px`,
      }}
    />
  );
};
```

## Overlays

### Vignette

Darkened edges for cinematic feel:

```tsx
const Vignette: React.FC<{ intensity?: number }> = ({ intensity = 0.6 }) => (
  <AbsoluteFill
    style={{
      background: `radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,${intensity}) 100%)`,
      pointerEvents: "none",
    }}
  />
);
```

### Film Grain

Subtle noise overlay for texture:

```tsx
const FilmGrain: React.FC<{ opacity?: number }> = ({ opacity = 0.04 }) => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill
      style={{
        opacity,
        pointerEvents: "none",
        mixBlendMode: "overlay",
        // SVG filter for noise
        filter: `url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.8' seed='${frame}' numOctaves='4'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>#n")`,
        width: "100%",
        height: "100%",
      }}
    />
  );
};
```

### Progress Bar

Animated progress indicator across the bottom:

```tsx
const ProgressBar: React.FC<{
  color?: string;
  height?: number;
}> = ({ color = "#00d4ff", height = 4 }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const progress = (frame / durationInFrames) * 100;

  return (
    <AbsoluteFill style={{ justifyContent: "flex-end" }}>
      <div
        style={{
          height,
          width: `${progress}%`,
          backgroundColor: color,
          boxShadow: `0 0 8px ${color}`,
        }}
      />
    </AbsoluteFill>
  );
};
```

## Layout Components

### Split Screen

Side-by-side comparison or content arrangement:

```tsx
const SplitScreen: React.FC<{
  left: React.ReactNode;
  right: React.ReactNode;
  dividerColor?: string;
}> = ({ left, right, dividerColor = "rgba(255,255,255,0.1)" }) => (
  <AbsoluteFill style={{ flexDirection: "row" }}>
    <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>{left}</div>
    <div style={{ width: 2, backgroundColor: dividerColor }} />
    <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>{right}</div>
  </AbsoluteFill>
);
```

### Centered Content Card

Floating card with shadow for focused content:

```tsx
const ContentCard: React.FC<{
  children: React.ReactNode;
  width?: number;
  bgColor?: string;
}> = ({ children, width = 800, bgColor = "rgba(255,255,255,0.05)" }) => (
  <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
    <div
      style={{
        width,
        padding: 48,
        backgroundColor: bgColor,
        borderRadius: 16,
        border: "1px solid rgba(255,255,255,0.1)",
        backdropFilter: "blur(20px)",
      }}
    >
      {children}
    </div>
  </AbsoluteFill>
);
```

## Animated Shapes

### Animated Circle

```tsx
const PulsingCircle: React.FC<{ color?: string; size?: number }> = ({
  color = "#00d4ff",
  size = 200,
}) => {
  const frame = useCurrentFrame();
  const scale = 1 + Math.sin(frame * 0.05) * 0.1;

  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        backgroundColor: color,
        opacity: 0.3,
        transform: `scale(${scale})`,
        filter: `blur(${size * 0.3}px)`,
      }}
    />
  );
};
```

### Decorative Lines

```tsx
const AnimatedLine: React.FC<{ direction?: "horizontal" | "vertical" }> = ({
  direction = "horizontal",
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const length = interpolate(frame, [0, fps * 0.5], [0, 100], {
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.33, 1, 0.68, 1),
  });

  const isH = direction === "horizontal";

  return (
    <div
      style={{
        [isH ? "width" : "height"]: `${length}%`,
        [isH ? "height" : "width"]: 2,
        background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)",
      }}
    />
  );
};
```
