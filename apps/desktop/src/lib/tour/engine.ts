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
  highlight: (step: object) => void
  isActive: () => boolean
  isLastStep: () => boolean
  moveNext: () => void
  movePrevious: () => void
}

export type TourDriverFactory = (config?: object) => TourDriver

/** Where a surface keeps its live instance between actions (module state in
 *  the app, a window global in the preview page). */
export interface TourHolder {
  driver?: TourDriver
}

/** One highlighted moment: an element, and what to say about it. Omit the
 *  selector for a centered narration step. */
export interface TourStep {
  selector?: string
  side?: 'top' | 'right' | 'bottom' | 'left'
  text?: string
  title?: string
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
  doc: Document
): TourResult {
  const kind = action.kind

  const popoverOf = (step: TourStep) =>
    step.title || step.text
      ? { description: step.text || '', side: step.side || undefined, title: step.title || '' }
      : undefined

  const stepOf = (step: TourStep) => ({ element: step.selector || undefined, popover: popoverOf(step) })

  /** Selectors that match nothing right now — the caller's cue to re-scan. */
  const unmatched = (steps: TourStep[]) =>
    steps
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

  if (kind === 'show') {
    const gone = unmatched([action])

    if (gone.length > 0) {
      return missing(gone)
    }

    if (!holder.driver) {
      holder.driver = factory({ animate: true, smoothScroll: true })
    }

    holder.driver.highlight(stepOf(action))

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

    const startAt = action.startAt || 0

    holder.driver = factory({
      animate: true,
      showProgress: steps.length > 1,
      smoothScroll: true,
      steps: steps.map(stepOf)
    })
    holder.driver.drive(startAt)

    return { action: kind, activeStep: startAt, steps: steps.length, success: true }
  }

  if (kind === 'next' || kind === 'prev') {
    const driver = holder.driver

    if (!driver || !driver.isActive()) {
      return { error: "No tour is running — start one first.", success: false }
    }

    if (kind === 'next' && driver.isLastStep()) {
      driver.destroy()
      holder.driver = undefined

      return { action: kind, done: true, success: true }
    }

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

    return { action: kind, success: true }
  }

  return { error: 'Unknown tour action: ' + kind, success: false }
}
