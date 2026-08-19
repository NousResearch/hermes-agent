/**
 * RUN TOUR — the tour verbs, on either surface.
 *
 * `runTour` is the generic entry (one normalized action in, a result out) and
 * the named verbs below are the ergonomic API — used by the agent tool through
 * the gateway, and by any curated in-app tour. surface='app' drives driver.js
 * against the Hermes DOM; surface='preview' runs the same engine source inside
 * the preview pane's guest page.
 *
 * Dynamic-imported by gateway-event.ts, so driver.js (and the preview's raw
 * injection payload) stay off the boot path until a tour actually runs.
 */

import 'driver.js/dist/driver.css'
import './app-tour.css'

import { driver as driverFactory } from 'driver.js'

import { runPreviewTour } from '@/app/chat/right-rail/preview-tour'

import { collectTourTargets } from './collect-targets'
import { runTourEngine, type TourAction, type TourHolder, type TourResult, type TourStep } from './engine'

/** Which document a tour runs against. */
export type TourSurface = 'app' | 'preview'

/** The app surface's live driver instance, reused across actions. */
const appHolder: TourHolder = {}

/** Run one tour action on `surface`. Never throws — failures come back as
 *  `{success: false, error}` so a caller (or the agent) can recover. */
export async function runTour(action: TourAction, surface: TourSurface = 'app'): Promise<TourResult> {
  try {
    return surface === 'preview'
      ? await runPreviewTour(action)
      : runTourEngine(driverFactory, appHolder, action, collectTourTargets, document)
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error), success: false }
  }
}

/** What's on screen right now, durable (`stable`) handles first. */
export const listTourTargets = (surface?: TourSurface) => runTour({ kind: 'targets' }, surface)

/** Spotlight one element with a popover, replacing any current highlight. */
export const showTourStep = (step: TourStep, surface?: TourSurface) => runTour({ ...step, kind: 'show' }, surface)

/** Begin a multi-step tour the user pages through with Next/Prev. */
export const startTour = (steps: TourStep[], surface?: TourSurface, startAt = 0) =>
  runTour({ kind: 'start', startAt, steps }, surface)

/** Advance a running tour; past the last step it ends and cleans up. */
export const nextTourStep = (surface?: TourSurface) => runTour({ kind: 'next' }, surface)

/** Step a running tour backward. */
export const previousTourStep = (surface?: TourSurface) => runTour({ kind: 'prev' }, surface)

/** End the tour and clear the overlay. Safe to call when none is running. */
export const stopTour = (surface?: TourSurface) => runTour({ kind: 'stop' }, surface)
