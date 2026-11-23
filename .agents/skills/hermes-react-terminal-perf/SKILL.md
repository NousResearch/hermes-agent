---
name: hermes-react-terminal-perf
description: Performance hygiene and rendering optimization for the Hermes Desktop/Webapp React, Vite, and xterm.js terminal renderer. Prevents unnecessary React re-renders around mounted terminal instances, coalesces ResizeObserver and MutationObserver bursts, avoids Vite barrel-import chunk bloat, and verifies builds via offline webapp build scripts.
---

Based on: [https://skills.sh/vercel-labs/agent-skills/vercel-react-best-practices](https://skills.sh/vercel-labs/agent-skills/vercel-react-best-practices) (Inspired only / License not verified)

# Hermes React & Terminal Performance

Performance hygiene guide and verification steps for the Hermes Desktop/Webapp React/TSX + Vite + xterm.js terminal architecture (`apps/desktop/`).

## When to Use
- When optimizing React components interacting with mounted xterm.js WebGL/DOM terminal instances.
- When stabilizing `PersistentTerminal` and `KeepAlivePanes` without causing accidental instance teardowns or memory leaks.
- When managing high-frequency layout observations (`ResizeObserver`, `MutationObserver`) in terminal container panels.
- When verifying production bundle output and chunk health.

---

## Actionable Steps

### 1. Protect xterm.js Instances from Unnecessary Re-renders
- Wrap terminal container wrappers and active session panels in `React.memo` with precise equality checks.
- Keep xterm `Terminal` and `Addon` (FitAddon, WebglAddon) references stable using `useRef` rather than state objects, preventing expensive terminal re-initializations on parent state updates.
- Ensure terminal DOM container refs do not trigger state cascading during mount/resize cycles.

### 2. Coalesce Layout Observer Bursts
- Batch rapid `ResizeObserver` and `MutationObserver` callbacks using `requestAnimationFrame` to prevent layout thrashing and CPU spikes during window resizing or panel splitting.
- Clean up all active observers in `useEffect` cleanup functions.

### 3. Eliminate Barrel-Import Chunk Bloat
- Avoid importing entire UI icon sets or large utility suites via barrel index files (`import { ... } from '@/components'`). Import directly from individual component files (`import { TerminalView } from '@/components/TerminalView'`).
- Keep Vite manual chunks lean to ensure fast cold starts and minimal bundle size.

### 4. Verify Build Integrity
- Run offline webapp builds using the project build script:
  ```bash
  node apps/desktop/scripts/build-webapp.mjs --no-install
  ```
- Check that output bundles conform to expected chunk limits and pass TypeScript compilation without errors.

---

## What this skill does NOT do
- Never applies Next.js or React Server Components (RSC) patterns to this Vite client-side SPA.
- Never patches xterm.js internal canvas rendering or upstream mouse selection workarounds.
- Never sacrifices accessibility (a11y), keyboard navigation, or core correctness for micro-optimizations.
