// Dev-only: eyeball the tour UI without asking the agent for one.
//
// Three triggers, all the same toggle: a floating button (bottom-left), the
// Ctrl+Shift+X chord, and a ⌘K row. The button exists because a chord can be
// swallowed before the renderer ever sees it — by a menu accelerator, a
// main-process before-input-event hook, or the OS — and a demo you can't
// reliably open is worse than no demo.
//
// The steps walk the Artifacts page: a route step, a late-mounting target,
// per-step narration, and the return trip on exit. That covers the whole
// feature on a surface every user has, which is what the agent does when
// asked "how does X work?".
//
// Installed only under `import.meta.env.DEV` (see contrib/wiring.tsx), so none
// of this ships in a production build.

import { PALETTE_AREA, type PaletteContribution } from '@/app/command-palette/contrib'
import { ARTIFACTS_ROUTE } from '@/app/routes'
import { registry } from '@/contrib/registry'
import { Compass } from '@/lib/icons'
import { isTourActive, startTour, stopTour, type TourStep } from '@/lib/tour'

const BUTTON_ID = '__hermes-tour-demo-button'

/**
 * The demo tour: the Artifacts page, start to finish.
 *
 * A realistic walkthrough rather than a style swatch — it navigates to a page,
 * explains that page's nav, then explains one item on it, and hands the app
 * back when it ends. That exercises the whole feature (route step, late-
 * mounting target, per-step narration, return trip) on a surface every user
 * has, which is what the agent does when asked "how does X work?".
 *
 * `data-tour` handles are used on purpose: they're the documented way to give
 * an element a durable name, and they survive re-renders and class changes.
 */
function demoSteps(): TourStep[] {
  return [
    {
      // Opens centered, wherever you happen to be: a tour says what it's about
      // to do before it moves you anywhere.
      text: "I'll take you to the Artifacts page and show you how it's organised. Use Next to follow along — you'll land back here at the end.",
      title: "Let's look at Artifacts"
    },
    {
      navigate: ARTIFACTS_ROUTE,
      selector: '[data-tour="page-tabs"]',
      text: 'Here we are. This row filters what the page lists, with a live count beside each one. We\'ll take them in turn.',
      title: 'Artifacts, filtered'
    },
    {
      selector: '[data-tour="tab-all"]',
      side: 'bottom',
      text: 'Everything the agent produced or touched, newest first.',
      title: 'All'
    },
    {
      selector: '[data-tour="tab-image"]',
      side: 'bottom',
      text: 'Screenshots, renders, and diagrams — these get a thumbnail in the grid.',
      title: 'Images'
    },
    {
      selector: '[data-tour="tab-file"]',
      side: 'bottom',
      text: 'Documents and data the agent wrote or read, listed by name and path.',
      title: 'Files'
    },
    {
      selector: '[data-tour="tab-link"]',
      side: 'bottom',
      text: 'Pages the agent opened or cited, so you can retrace where something came from.',
      title: 'Links'
    },
    {
      // Nothing pins this to the FIRST card; the selector matches whichever
      // card the grid renders first, so the step works on any library.
      selector: '[data-tour="artifact-card"]',
      side: 'right',
      text: 'Whichever filter you pick, each result is a card: a preview, its name, and where it came from. Opening one brings it up in the preview pane.',
      title: 'One artifact'
    },
    {
      text: 'Closing the tour returns you to the session you started from — a tour borrows the app, it never leaves you somewhere else.',
      title: 'And back'
    }
  ]
}

/** Guards against a double-fire (two installs, or a chord that repeats)
 *  turning one request into start-then-immediately-stop. */
let inFlight = false

/** Toggle the demo tour. Resolves once the tour is up (or torn down). */
export async function toggleTourDemo(): Promise<void> {
  if (inFlight) {
    return
  }

  inFlight = true

  try {
    // Ask the engine, never a local flag: Esc / the ✕ / an overlay click all
    // end a tour without telling us, and a stale "it's running" belief would
    // turn the next toggle into a no-op stop — the trigger would look dead.
    if (isTourActive()) {
      await stopTour()

      return
    }

    const result = await startTour(demoSteps())

    if (!result.success) {
      console.warn('[dev] tour demo could not start:', result.error)
    }
  } finally {
    inFlight = false
  }
}

/** The always-there trigger: a floating button, styled off app tokens. */
function mountButton(): () => void {
  document.getElementById(BUTTON_ID)?.remove()

  const button = document.createElement('button')

  button.id = BUTTON_ID
  button.type = 'button'
  button.textContent = 'Tour demo'
  button.title = 'Dev only — toggle the tour UI (Ctrl+Shift+X)'
  button.style.cssText = [
    'position:fixed',
    'left:12px',
    'bottom:12px',
    // Under the tour overlay (10000) so a running tour still dims it, but over
    // ordinary app chrome.
    'z-index:9999',
    'padding:4px 8px',
    'font:500 11px/1rem var(--font-sans, system-ui)',
    'color:var(--foreground, #fff)',
    'background:var(--ui-bg-quaternary, rgba(255,255,255,.1))',
    'border:1px solid var(--ui-stroke-secondary, rgba(255,255,255,.15))',
    'border-radius:2.5px',
    'cursor:pointer',
    'opacity:.75',
    // driver.js sets `.driver-active * { pointer-events: none }` while a tour
    // is up, which would make this button unable to STOP the tour it started.
    'pointer-events:auto'
  ].join(';')

  // Fire on pointerdown, not click: the tour mounts an overlay that swallows
  // pointer events, so waiting for the full click can drop the trailing half
  // of the gesture onto the fresh overlay.
  const onPointerDown = (event: PointerEvent) => {
    event.preventDefault()
    event.stopPropagation()
    void toggleTourDemo()
  }

  button.addEventListener('pointerdown', onPointerDown)
  document.body.appendChild(button)

  return () => {
    button.removeEventListener('pointerdown', onPointerDown)
    button.remove()
  }
}

// Ctrl+Shift+X on every platform (Ctrl, not Cmd, so it can't clash with a
// system Cmd chord). Matched on `code` so it's layout independent, and
// captured before the composer so a focused input can't swallow it.
function isTriggerChord(e: KeyboardEvent): boolean {
  return e.ctrlKey && e.shiftKey && !e.metaKey && !e.altKey && e.code === 'KeyX'
}

/**
 * Install the dev triggers: a floating button, a capture-phase hotkey
 * (Ctrl+Shift+X), a ⌘K palette entry, and a `window.__tourDemo()` console
 * hook. Returns a disposer.
 */
export function installTourDemo(): () => void {
  const onKeyDown = (e: KeyboardEvent) => {
    if (!isTriggerChord(e)) {
      return
    }

    e.preventDefault()
    e.stopPropagation()
    void toggleTourDemo()
  }

  window.addEventListener('keydown', onKeyDown, { capture: true })
  ;(window as unknown as { __tourDemo?: () => Promise<void> }).__tourDemo = toggleTourDemo

  const unmountButton = mountButton()

  const disposePalette = registry.register({
    id: 'dev.tourDemo',
    area: PALETTE_AREA,
    data: {
      id: 'dev.tourDemo',
      icon: Compass,
      keywords: ['tour', 'walkthrough', 'highlight', 'driver', 'dev', 'demo'],
      label: 'Dev: toggle tour demo',
      // keepOpen:false closes the palette, and that teardown can land a
      // stray pointer/focus event on the just-mounted overlay — defer past it.
      run: () => window.setTimeout(() => void toggleTourDemo(), 120)
    } satisfies PaletteContribution
  })

  console.info('[dev] tour demo ready — click the "Tour demo" button, press Ctrl+Shift+X, or run window.__tourDemo()')

  return () => {
    window.removeEventListener('keydown', onKeyDown, { capture: true })
    unmountButton()
    disposePalette()
    delete (window as unknown as { __tourDemo?: () => Promise<void> }).__tourDemo

    if (isTourActive()) {
      void stopTour()
    }
  }
}
