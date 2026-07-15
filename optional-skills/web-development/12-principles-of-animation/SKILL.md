---
name: 12-principles-of-animation
description: Audit animation code against Disney's 12 principles adapted for web. Use when reviewing motion, implementing animations, or checking animation quality. Outputs file:line findings.
license: MIT
metadata:
  author: raphael-salaja
  version: "2.0.0"
  source: /content/12-principles-of-animation/index.mdx
---

# 12 Principles of Animation

Review animation code for compliance with Disney's 12 principles adapted for web interfaces.

## How It Works

1. Read the specified files (or prompt user for files/pattern)
2. Check against all rules below
3. Output findings in `file:line` format

## Rule Categories

| Priority | Category | Prefix |
|----------|----------|--------|
| 1 | Timing | `timing-` |
| 2 | Easing | `easing-` |
| 3 | Physics | `physics-` |
| 4 | Staging | `staging-` |

## Rules

### Timing Rules

#### `timing-under-300ms`
User-initiated animations must complete within 300ms.

**Fail:**
```css
.button { transition: transform 400ms; }
```

**Pass:**
```css
.button { transition: transform 200ms; }
```

#### `timing-consistent`
Similar elements must use identical timing values.

**Fail:**
```css
.button-primary { transition: 200ms; }
.button-secondary { transition: 150ms; }
```

**Pass:**
```css
.button-primary { transition: 200ms; }
.button-secondary { transition: 200ms; }
```

#### `timing-no-entrance-context-menu`
Context menus should not animate on entrance (exit only).

**Fail:**
```tsx
<motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} />
```

**Pass:**
```tsx
<motion.div exit={{ opacity: 0 }} />
```

### Easing Rules

#### `easing-entrance-ease-out`
Entrances must use `ease-out` (arrive fast, settle gently).

**Fail:**
```css
.modal-enter { animation-timing-function: ease-in; }
```

**Pass:**
```css
.modal-enter { animation-timing-function: ease-out; }
```

#### `easing-exit-ease-in`
Exits must use `ease-in` (build momentum before departure).

**Fail:**
```css
.modal-exit { animation-timing-function: ease-out; }
```

**Pass:**
```css
.modal-exit { animation-timing-function: ease-in; }
```

#### `easing-no-linear-motion`
Linear easing should only be used for progress indicators, not motion.

**Fail:**
```css
.card { transition: transform 200ms linear; }
```

**Pass:**
```css
.progress-bar { transition: width 100ms linear; }
```

#### `easing-natural-decay`
Use exponential ramps, not linear, for natural decay.

**Fail:**
```ts
gain.gain.linearRampToValueAtTime(0, t + 0.05);
```

**Pass:**
```ts
gain.gain.exponentialRampToValueAtTime(0.001, t + 0.05);
```

### Physics Rules

#### `physics-active-state`
Interactive elements must have active/pressed state with scale transform.

**Fail:**
```css
.button:hover { background: var(--gray-3); }
/* Missing :active state */
```

**Pass:**
```css
.button:active { transform: scale(0.98); }
```

#### `physics-subtle-deformation`
Squash/stretch deformation must be subtle (0.95-1.05 range).

**Fail:**
```tsx
<motion.div whileTap={{ scale: 0.8 }} />
```

**Pass:**
```tsx
<motion.div whileTap={{ scale: 0.98 }} />
```

#### `physics-spring-for-overshoot`
Use springs (not easing) when overshoot-and-settle is needed.

**Fail:**
```tsx
<motion.div transition={{ duration: 0.3, ease: "easeOut" }} />
// When element should bounce/settle
```

**Pass:**
```tsx
<motion.div transition={{ type: "spring", stiffness: 500, damping: 30 }} />
```

#### `physics-no-excessive-stagger`
Stagger delays must not exceed 50ms per item.

**Fail:**
```tsx
transition={{ staggerChildren: 0.15 }}
```

**Pass:**
```tsx
transition={{ staggerChildren: 0.03 }}
```

### Staging Rules

#### `staging-one-focal-point`
Only one element should animate prominently at a time.

**Fail:**
```tsx
// Multiple elements with competing entrance animations
<motion.div animate={{ scale: 1.1 }} />
<motion.div animate={{ scale: 1.1 }} />
```

#### `staging-dim-background`
Modal/dialog backgrounds should dim to direct focus.

**Fail:**
```css
.overlay { background: transparent; }
```

**Pass:**
```css
.overlay { background: var(--black-a6); }
```

#### `staging-z-index-hierarchy`
Animated elements must respect z-index layering.

**Fail:**
```css
.tooltip { /* No z-index, may render behind other elements */ }
```

**Pass:**
```css
.tooltip { z-index: 50; }
```

## Output Format

When reviewing files, output findings as:

```
file:line - [rule-id] description of issue

Example:
components/modal/index.tsx:45 - [timing-under-300ms] Exit animation 400ms exceeds 300ms limit
components/button/styles.module.css:12 - [physics-active-state] Missing :active transform
```

## Summary Table

After findings, output a summary:

| Rule | Count | Severity |
|------|-------|----------|
| `timing-under-300ms` | 2 | HIGH |
| `physics-active-state` | 3 | MEDIUM |
| `easing-entrance-ease-out` | 1 | MEDIUM |

## References

- [The Illusion of Life: Disney Animation](https://www.amazon.com/Illusion-Life-Disney-Animation/dp/0786860707)
- [Apple WWDC23: Animate with Springs](https://developer.apple.com/videos/play/wwdc2023/10158)
