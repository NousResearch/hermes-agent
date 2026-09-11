/**
 * TOUR ENGINE — the one function that runs a tour action, on BOTH surfaces.
 *
 * The app surface calls it directly (imported driver.js factory + a module
 * holder + `document`). The preview surface injects `runTourEngine.toString()`
 * into the pane's <webview> alongside the driver.js IIFE and calls it with the
 * page's own globals. Because the same source runs in both places it MUST stay
 * self-contained: no imports, no closure references, no renderer globals —
 * everything arrives as a parameter. The minimal structural types below erase
 * at compile time, so the stringified function stays plain JS.
 */

import type { TourTarget } from './collect-targets'

/** The slice of a driver.js instance the engine drives. */
export interface TourDriver {
  destroy: () => void
  drive: (stepIndex?: number) => void
  getActiveIndex: () => number | undefined
  /** driver.js's own state bag. Used to read the live, eased stage rect so a
   *  companion layer can travel with the cutout instead of jumping ahead. */
  getState?: (key?: string) => unknown
  highlight: (step: object) => void
  isActive: () => boolean
  isLastStep: () => boolean
  moveNext: () => void
  movePrevious: () => void
}

export type TourDriverFactory = (config?: object) => TourDriver

/** Look-and-feel passed into every driver instance: the scrim, how the cutout
 *  hugs the highlighted element, and the transition speed. Surface-supplied so
 *  the app can match its own overlay tokens while a guest page keeps
 *  driver.js's defaults. */
export interface TourStyle {
  duration?: number
  overlayColor?: string
  overlayOpacity?: number
  popoverOffset?: number
  stagePadding?: number
  stageRadius?: number
}

/** Where a surface keeps its live instance between actions (module state in
 *  the app, a window global in the preview page). */
export interface TourHolder {
  driver?: TourDriver
  /** Tears down the keep-alive observer. Runs with the driver. */
  release?: () => void
}

/** One highlighted moment: an element, what to say about it, and optionally
 *  where the app must be for the element to exist. `navigate` (a route path)
 *  and `pane` (a desktop pane name) run before the step is shown; the host
 *  supplies the actual navigation (see TourHost), so the engine itself stays
 *  self-contained and portable to a guest page. */
export interface TourStep {
  navigate?: string
  pane?: string
  selector?: string
  side?: 'top' | 'right' | 'bottom' | 'left'
  text?: string
  title?: string
}

/** How the engine reaches the app around it. Omitted on the preview surface,
 *  where a guest page has neither routes nor panes. */
export interface TourHost {
  /** Where the app is right now, so the tour can put it back when it ends. */
  currentRoute?: () => string
  /** Go to a route path. */
  navigate?: (to: string) => void
  /** Reveal a desktop pane by name. */
  revealPane?: (pane: string) => void
}

/** A normalized action. `kind` is the verb; the rest is per-verb payload. */
export interface TourAction extends TourStep {
  kind: 'targets' | 'show' | 'start' | 'next' | 'prev' | 'stop'
  startAt?: number
  steps?: TourStep[]
}

export interface TourResult {
  action?: string
  activeStep?: number
  done?: boolean
  error?: string
  hint?: string
  steps?: number
  success: boolean
  targets?: TourTarget[]
  title?: string
  url?: string
}

/** Run one tour action. Self-contained (see module doc). */
export function runTourEngine(
  factory: TourDriverFactory,
  holder: TourHolder,
  action: TourAction,
  collectTargets: (doc: Document, max: number) => TourTarget[],
  doc: Document,
  style?: TourStyle,
  host?: TourHost
): TourResult {
  const kind = action.kind
  const base = { animate: true, smoothScroll: true, ...(style || {}) }

  /** driver.js reads `popoverClass` off the POPOVER object, not off the step
   *  (`r.popoverClass` where r = step.popover), so the entrance animation has
   *  to be declared in here. `first` settles down into place, every step after
   *  rises — styled in app-tour.css. */
  const popoverOf = (step: TourStep, first = false) =>
    step.title || step.text
      ? {
          description: step.text || '',
          popoverClass: first ? 'tour-pop-in' : 'tour-pop-next',
          side: step.side || undefined,
          title: step.title || ''
        }
      : undefined

  /** Move the app where a step needs to be.
   *
   *  Called from `onHighlightStarted`, which driver.js fires for EVERY way a
   *  step can be reached — the API, its own Next/Prev buttons, arrow keys,
   *  `drive(i)`. Hooking the paths individually is what let a user clicking
   *  Next (or pressing →) advance the text while the app never navigated, so
   *  the step sat waiting for an element on a page it never opened. One hook,
   *  no path can skip it. */
  const moveFor = (step: TourStep) => {
    if (step.pane && host?.revealPane) {
      host.revealPane(step.pane)
    }

    if (step.navigate && host?.navigate) {
      host.navigate(step.navigate)
    }

    // No waiting here: a target that mounts after the move is picked up by the
    // keep-alive below, which re-drives the step the moment the element exists.
    // driver.js's own `waitForElement` can't do the job — it returns BEFORE the
    // hooks run, so the step would wait for an element on a page this hook was
    // about to open, which is the deadlock that looked like a freeze.
  }

  /** Keep the current step bound to a LIVE node.
   *
   *  Covers the two ways a step's element goes missing under it:
   *
   *  - It hasn't mounted yet, because the step just navigated somewhere.
   *  - It mounted, then the surface re-rendered — a poll, a refetch, a list
   *    reordering — and React swapped the node. driver.js holds the highlight
   *    as a node REFERENCE and re-measures it with getBoundingClientRect(), so
   *    a detached node measures all zeros: the spotlight collapses and the
   *    tour looks like it closed itself while the user did nothing.
   *
   *  Both are "the element the step wants isn't the element driver is holding",
   *  so both are fixed the same way: re-drive the step to rebind. */
  const bindKeepAlive = (steps: TourStep[]) => {
    let queued = false

    // Coalesce a mutation burst into one check on the next frame. Falls back to
    // a timeout where rAF is absent (jsdom, and any host page without it) —
    // this module is stringified into the preview webview, so it can't assume
    // browser globals beyond the DOM.
    const soon = (fn: () => void) => {
      if (typeof requestAnimationFrame === 'function') {
        requestAnimationFrame(fn)
      } else {
        setTimeout(fn, 16)
      }
    }

    const observer = new MutationObserver(() => {
      if (queued) {
        return
      }

      queued = true
      soon(() => {
        queued = false

        const driver = holder.driver

        if (!driver?.isActive()) {
          return
        }

        const index = driver.getActiveIndex() ?? 0
        const selector = steps[index]?.selector

        if (!selector) {
          return
        }

        const wanted = doc.querySelector(selector)

        // Nothing to bind to yet — a navigating step whose page is still
        // rendering. The next mutation brings us back.
        if (!wanted) {
          return
        }

        // Already on the right node: nothing to do. This is the common case,
        // so it stays a single identity check per mutation batch.
        if (wanted === doc.querySelector('.driver-active-element')) {
          return
        }

        driver.drive(index)
      })
    })

    observer.observe(doc.body, { childList: true, subtree: true })

    return () => observer.disconnect()
  }

  /** Every step carries its move on `onHighlightStarted`, the one hook driver
   *  fires no matter how the step was reached. */
  const stepOf = (step: TourStep, first = false) => ({
    element: step.selector || undefined,
    onHighlightStarted: () => moveFor(step),
    popover: popoverOf(step, first)
  })

  /** Selectors that match nothing right now — the caller's cue to re-scan.
   *
   *  Skipped entirely once any step moves the app: from that point the later
   *  steps' targets legitimately don't exist yet (they mount on the page the
   *  tour is about to open), so pre-validating them would reject a correct
   *  tour. driver.js waits for each one via `waitForElement` instead. */
  const unmatched = (steps: TourStep[]) =>
    steps.some(step => step.navigate || step.pane)
      ? []
      : steps
          .map(step => step.selector)
          .filter((selector): selector is string => {
            try {
              return !!selector && !doc.querySelector(selector)
            } catch {
              return true
            }
          })

  const missing = (selectors: string[]): TourResult => ({
    error: 'No element matches selector(s): ' + selectors.join(', '),
    hint: "The DOM may have re-rendered — re-scan with 'targets' and prefer targets marked stable.",
    success: false
  })

  if (kind === 'targets') {
    return {
      success: true,
      targets: collectTargets(doc, 150),
      title: doc.title,
      url: doc.location ? doc.location.href : ''
    }
  }

  /** Remove any driver DOM this engine no longer owns.
   *
   *  driver.js tears itself down through the instance that built it, so an
   *  instance we lose our handle to — a hot reload, a double install, an
   *  exception between create and store — leaves its overlay and popover on
   *  screen forever. They then sit ON TOP of the next tour: the new step is
   *  running underneath a stale popover, which reads as "the tour froze".
   *  Cheap to check, and the only way to guarantee a clean start. */
  const clearOrphans = () => {
    for (const node of doc.querySelectorAll('.driver-popover, svg.driver-overlay')) {
      node.remove()
    }

    doc.body.classList.remove('driver-active', 'driver-fade', 'driver-simple', 'driver-no-scroll')
  }

  if (kind === 'show') {
    const gone = unmatched([action])

    if (gone.length > 0) {
      return missing(gone)
    }

    if (!holder.driver) {
      clearOrphans()
      holder.driver = factory(base)
    }

    // A one-off highlight is its own arrival, so it uses the settle-down enter.
    holder.driver.highlight(stepOf(action, true))

    return { action: kind, success: true }
  }

  if (kind === 'start') {
    const steps = action.steps || []
    const gone = unmatched(steps)

    if (gone.length > 0) {
      return missing(gone)
    }

    if (holder.driver) {
      holder.driver.destroy()
    }

    // Whatever the previous tour left behind goes now, so this one isn't
    // rendered underneath a stale popover.
    clearOrphans()

    const startAt = action.startAt || 0

    // A tour that navigates must hand the app back. Captured before the first
    // step moves anywhere, and restored on EVERY exit — Esc, the ✕, an overlay
    // click, and the last step all funnel through onDestroyed.
    const origin = steps.some(step => step.navigate) ? host?.currentRoute?.() : undefined

    holder.driver = factory({
      ...base,
      onDestroyed: () => {
        holder.driver = undefined
        holder.release?.()
        holder.release = undefined

        if (origin !== undefined && host?.navigate && host.currentRoute?.() !== origin) {
          host.navigate(origin)
        }
      },
      showProgress: steps.length > 1,
      // Each step moves the app from its own onHighlightStarted, so buttons,
      // arrow keys and the API all take the same path (see moveFor).
      steps: steps.map((step, i) => stepOf(step, i === startAt))
    })

    // Survive the surface re-rendering underneath the tour (see bindKeepAlive).
    holder.release?.()
    holder.release = bindKeepAlive(steps)

    holder.driver.drive(startAt)

    return { action: kind, activeStep: startAt, steps: steps.length, success: true }
  }

  if (kind === 'next' || kind === 'prev') {
    const driver = holder.driver

    if (!driver || !driver.isActive()) {
      return { error: 'No tour is running — start one first.', success: false }
    }

    if (kind === 'next' && driver.isLastStep()) {
      driver.destroy()
      holder.driver = undefined

      return { action: kind, done: true, success: true }
    }

    // No move here: driver fires the step's own onHighlightStarted, which is
    // where every path — this one, the buttons, arrow keys — does the moving.
    if (kind === 'next') {
      driver.moveNext()
    } else {
      driver.movePrevious()
    }

    return { action: kind, activeStep: driver.getActiveIndex(), success: true }
  }

  if (kind === 'stop') {
    if (holder.driver) {
      holder.driver.destroy()
      holder.driver = undefined
    }

    // Belt and braces: destroy() only cleans up what THIS instance built.
    clearOrphans()
    holder.release?.()
    holder.release = undefined

    return { action: kind, success: true }
  }

  return { error: 'Unknown tour action: ' + kind, success: false }
}
