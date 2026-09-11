# Intro reveal

An idealized chat types a request, works through tools and a cube viewport,
streams a reply, expands into parallel agents, then closes on the brand.
The seven beats take 22 seconds, followed by a 900 ms dissolve. Reduced motion
shows the brand briefly. Sound defaults on and respects the haptics mute preference.
Fonts are the existing Collapse and JetBrains Mono faces.

Eligibility is `guestOnboardingEnabled && !firstRunSkipped && !hasSeenIntroReveal()`.
Electron sets the flag from `HERMES_GUEST_ONBOARDING=1` or `--guest-onboarding`.
The gate queues the guided chat on completion; the chat gate acknowledges the
free-tier notice as the cinematic starts. With the flag off, neither gate starts.

| Piece | Path |
| --- | --- |
| Main-window gate and conductor | `index.tsx` |
| Phase, seen-key ownership and IPC listeners | `../../store/intro-reveal.ts` |
| Overlay boot and surface | `intro-root.tsx`, `intro-reveal-surface.tsx` |
| Clock, score, cube and synthesized sound | `use-intro-clock.ts`, `timeline.ts`, `viewport-cube.ts`, `sound.ts` |
| Constellation, brand and text effects | `scenes/` |
| Native window and preload bridge | `../../../electron/intro-reveal-window.ts`, `../../../electron/preload.ts` |

The transparent native window (`?win=intro`) covers the primary display while
the main app hides. The overlay owns the clock because the hidden main renderer's
animation frames are throttled. The main renderer owns the phase and seen key.

The screen must ALWAYS come back. Four independent layers:

1. Normal completion: the overlay clock finishes → main renderer closes it.
2. Esc/click: local close with a 1.2s fallback that bypasses the main renderer.
3. Local deadman: the overlay force-closes itself `INTRO_DEADMAN_MS` after
   mount, even with all IPC dead.
4. Main-process watchdog (34s) destroys the window unconditionally.

Rehearsal with isolated state: see [Desktop Engineering Guide](../../../AGENTS.md#rehearsing-the-guided-onboarding).
