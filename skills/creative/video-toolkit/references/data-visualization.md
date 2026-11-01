# Data Visualization

Animated charts, counters, and stat comparisons for data-driven video content.

## Animated Counter

Numbers counting up from 0 to a target value:

```tsx
const AnimatedCounter: React.FC<{
  target: number;
  duration?: number; // seconds
  prefix?: string;
  suffix?: string;
  decimals?: number;
}> = ({ target, duration = 2, prefix = "", suffix = "", decimals = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const value = interpolate(frame, [0, duration * fps], [0, target], {
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.33, 1, 0.68, 1),
  });

  return (
    <span style={{ fontVariantNumeric: "tabular-nums", fontFamily: "Inter", fontWeight: 700 }}>
      {prefix}{value.toFixed(decimals)}{suffix}
    </span>
  );
};

// Usage:
<AnimatedCounter target={99.7} suffix="%" decimals={1} />
<AnimatedCounter target={50000} prefix="$" />
<AnimatedCounter target={12} suffix="x faster" />
```

## Stat Card

A single metric with label, animated entrance:

```tsx
const StatCard: React.FC<{
  label: string;
  value: number;
  suffix?: string;
  delay?: number;
  color?: string;
}> = ({ label, value, suffix = "", delay = 0, color = "#00d4ff" }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const entrance = spring({
    frame: frame - delay,
    fps,
    config: { damping: 14, mass: 0.5 },
  });

  const countValue = interpolate(
    frame - delay,
    [0, fps * 1.5],
    [0, value],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <div style={{
      opacity: entrance,
      transform: `translateY(${interpolate(entrance, [0, 1], [30, 0])}px) scale(${entrance})`,
      textAlign: "center",
      padding: 24,
    }}>
      <div style={{ fontSize: 72, fontWeight: 800, color, fontVariantNumeric: "tabular-nums" }}>
        {Math.round(countValue)}{suffix}
      </div>
      <div style={{ fontSize: 24, color: "rgba(255,255,255,0.6)", marginTop: 8 }}>
        {label}
      </div>
    </div>
  );
};
```

## Bar Chart

Animated horizontal or vertical bars:

```tsx
const BarChart: React.FC<{
  data: { label: string; value: number; color?: string }[];
  maxValue?: number;
  direction?: "horizontal" | "vertical";
}> = ({ data, maxValue, direction = "horizontal" }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const max = maxValue ?? Math.max(...data.map((d) => d.value));

  return (
    <div style={{
      display: "flex",
      flexDirection: direction === "horizontal" ? "column" : "row",
      gap: 12,
      alignItems: direction === "horizontal" ? "stretch" : "flex-end",
      width: "100%",
      height: direction === "vertical" ? "100%" : "auto",
    }}>
      {data.map((item, i) => {
        const barProgress = interpolate(
          frame - i * 8, // stagger
          [0, fps * 0.8],
          [0, item.value / max],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.bezier(0.33, 1, 0.68, 1) }
        );
        const barColor = item.color ?? `hsl(${(i / data.length) * 200 + 200}, 70%, 55%)`;

        if (direction === "horizontal") {
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ width: 100, fontSize: 14, color: "rgba(255,255,255,0.7)", textAlign: "right" }}>
                {item.label}
              </div>
              <div style={{ flex: 1, height: 24, borderRadius: 4, overflow: "hidden", backgroundColor: "rgba(255,255,255,0.05)" }}>
                <div style={{ width: `${barProgress * 100}%`, height: "100%", backgroundColor: barColor, borderRadius: 4 }} />
              </div>
              <div style={{ width: 60, fontSize: 14, color: "rgba(255,255,255,0.7)" }}>
                {Math.round(barProgress * item.value)}
              </div>
            </div>
          );
        }

        return (
          <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-end", gap: 8 }}>
            <div style={{ width: "80%", height: `${barProgress * 100}%`, backgroundColor: barColor, borderRadius: "4px 4px 0 0" }} />
            <div style={{ fontSize: 12, color: "rgba(255,255,255,0.6)" }}>{item.label}</div>
          </div>
        );
      })}
    </div>
  );
};
```

## Before/After Comparison

Two stat sets with a visual divider:

```tsx
const BeforeAfter: React.FC<{
  before: { label: string; value: string }[];
  after: { label: string; value: string }[];
  beforeTitle?: string;
  afterTitle?: string;
}> = ({ before, after, beforeTitle = "Before", afterTitle = "After" }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const dividerProgress = interpolate(frame, [fps * 0.5, fps * 1], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ flexDirection: "row", padding: 80 }}>
      {/* Before */}
      <div style={{ flex: 1, opacity: 0.5 }}>
        <h3 style={{ color: "rgba(255,255,255,0.4)", fontSize: 24 }}>{beforeTitle}</h3>
        {before.map((item, i) => (
          <div key={i} style={{ marginTop: 16 }}>
            <div style={{ fontSize: 36, color: "rgba(255,255,255,0.5)" }}>{item.value}</div>
            <div style={{ fontSize: 16, color: "rgba(255,255,255,0.3)" }}>{item.label}</div>
          </div>
        ))}
      </div>

      {/* Divider */}
      <div style={{
        width: 2,
        backgroundColor: "rgba(255,255,255,0.2)",
        margin: "0 40px",
        transform: `scaleY(${dividerProgress})`,
        transformOrigin: "top",
      }} />

      {/* After */}
      <div style={{ flex: 1 }}>
        <h3 style={{ color: "#00d4ff", fontSize: 24 }}>{afterTitle}</h3>
        {after.map((item, i) => {
          const entrance = spring({
            frame: frame - fps * 1 - i * 10,
            fps,
            config: { damping: 12 },
          });
          return (
            <div key={i} style={{ marginTop: 16, opacity: entrance, transform: `translateX(${interpolate(entrance, [0, 1], [20, 0])}px)` }}>
              <div style={{ fontSize: 36, fontWeight: 700, color: "white" }}>{item.value}</div>
              <div style={{ fontSize: 16, color: "rgba(255,255,255,0.6)" }}>{item.label}</div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```

## Percentage Circle

Animated donut/ring showing a percentage:

```tsx
const PercentageCircle: React.FC<{
  value: number; // 0-100
  size?: number;
  color?: string;
  strokeWidth?: number;
}> = ({ value, size = 200, color = "#00d4ff", strokeWidth = 12 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = interpolate(frame, [0, fps * 1.5], [0, value / 100], {
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.33, 1, 0.68, 1),
  });

  const r = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - progress);

  return (
    <div style={{ width: size, height: size, position: "relative" }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth={strokeWidth} />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none"
          stroke={color} strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
        />
      </svg>
      <div style={{
        position: "absolute", inset: 0,
        display: "flex", justifyContent: "center", alignItems: "center",
        fontSize: size * 0.25, fontWeight: 700, color: "white",
      }}>
        {Math.round(progress * value)}%
      </div>
    </div>
  );
};
```
