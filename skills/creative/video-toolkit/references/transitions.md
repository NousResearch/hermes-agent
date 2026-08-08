# Transitions

## TransitionSeries

The primary scene arrangement component when transitions are needed between scenes:

```bash
npx remotion add @remotion/transitions
```

```tsx
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";

<TransitionSeries>
  <TransitionSeries.Sequence durationInFrames={90}>
    <TitleScene />
  </TransitionSeries.Sequence>

  <TransitionSeries.Transition
    presentation={fade()}
    timing={linearTiming({ durationInFrames: 20 })}
  />

  <TransitionSeries.Sequence durationInFrames={120}>
    <FeatureScene />
  </TransitionSeries.Sequence>

  <TransitionSeries.Transition
    presentation={slide({ direction: "from-left" })}
    timing={linearTiming({ durationInFrames: 25 })}
  />

  <TransitionSeries.Sequence durationInFrames={90}>
    <CTAScene />
  </TransitionSeries.Sequence>
</TransitionSeries>
```

**Important:** TransitionSeries shortens the timeline — during a transition, both scenes play simultaneously. A 20-frame fade means the video is 20 frames shorter than the sum of all sequence durations.

## Timing Functions

```tsx
import { linearTiming, springTiming } from "@remotion/transitions";

// Linear (constant speed)
timing={linearTiming({ durationInFrames: 20 })}

// Spring (physics-based, natural feel — recommended)
timing={springTiming({ config: { damping: 12 }, durationInFrames: 30 })}
```

Spring timing is almost always better than linear for scene transitions.

## Built-in Transitions

| Transition | Import | Options | Best for |
|-----------|--------|---------|----------|
| `fade()` | `@remotion/transitions/fade` | — | Universal, subtle |
| `slide()` | `@remotion/transitions/slide` | `direction`: from-left/right/top/bottom | Panel reveals, directional flow |
| `wipe()` | `@remotion/transitions/wipe` | `direction`, `angle` | Cinematic, energetic |
| `flip()` | `@remotion/transitions/flip` | `direction` | Playful, card-like |
| `clockWipe()` | `@remotion/transitions/clock-wipe` | — | Dramatic, radial |

```tsx
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";
import { wipe } from "@remotion/transitions/wipe";

// Slide from right
slide({ direction: "from-right" })

// Wipe at angle
wipe({ direction: "from-left" })
```

## Overlays (Non-Shortening)

Overlays render ON TOP of the cut point without shortening the timeline. Use for light leaks, flash effects, or decorative elements:

```bash
npx remotion add @remotion/light-leaks
```

```tsx
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { LightLeak } from "@remotion/light-leaks";

<TransitionSeries>
  <TransitionSeries.Sequence durationInFrames={90}>
    <SceneOne />
  </TransitionSeries.Sequence>

  <TransitionSeries.Overlay timing={linearTiming({ durationInFrames: 30 })}>
    <LightLeak seed={3} />
  </TransitionSeries.Overlay>

  <TransitionSeries.Sequence durationInFrames={120}>
    <SceneTwo />
  </TransitionSeries.Sequence>
</TransitionSeries>
```

## Custom Transitions

Build transitions as React components that receive `progress` (0 to 1):

```tsx
import { TransitionPresentation } from "@remotion/transitions";

// Glitch transition
const glitch = (): TransitionPresentation => {
  return {
    component: ({ progress, children }) => {
      const glitchAmount = Math.sin(progress * Math.PI) * 20;
      const rgbShift = Math.sin(progress * Math.PI) * 10;

      return (
        <AbsoluteFill>
          <div style={{
            width: "100%",
            height: "100%",
            filter: `hue-rotate(${glitchAmount * 5}deg)`,
            transform: `translateX(${Math.random() > 0.9 ? glitchAmount : 0}px)`,
          }}>
            {children}
          </div>
        </AbsoluteFill>
      );
    },
  };
};
```

### Custom Transition Ideas

| Transition | Description | Technique |
|-----------|-------------|-----------|
| **Glitch** | Digital distortion with RGB shift and horizontal tears | Random translateX + hue-rotate at peak progress |
| **Pixelate** | Mosaic dissolution — pixels grow larger then resolve | CSS `image-rendering: pixelated` + scale |
| **Blur zoom** | Scene blurs and zooms simultaneously | `filter: blur()` + `transform: scale()` |
| **Curtain** | Vertical strips reveal the next scene left to right | Clip-path with animated percentage |
| **Flash** | Bright white flash between scenes | White overlay with opacity peak at progress=0.5 |

## Transition Selection Guide

| Scenario | Recommended |
|----------|-------------|
| Default / safe | `fade()` with spring timing |
| Directional flow (left→right narrative) | `slide({ direction: "from-right" })` |
| Energy change (calm→intense) | `wipe()` or custom flash |
| Same topic, new angle | `fade()` 15-frame duration |
| Dramatic reveal | `clockWipe()` |
| Playful / casual | `flip()` |
| Technical / glitch aesthetic | Custom glitch transition |
| Cinematic | Light leak overlay + fade |

## Transition Timing Guidelines

- **Subtle transitions** (same topic): 10-15 frames (0.3-0.5s)
- **Standard scene change**: 20-25 frames (0.7-0.8s)
- **Dramatic reveal**: 30-40 frames (1.0-1.3s)
- **Music-synced**: align transition midpoint with the beat
