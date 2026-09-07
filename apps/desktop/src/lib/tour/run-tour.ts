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
import { revealDesktopPane } from '@/store/pane-focus'

import { collectTourTargets } from './collect-targets'
import {
  runTourEngine,
  type TourAction,
  type TourHolder,
  type TourHost,
  type TourResult,
  type TourStep,
  type TourStyle
} from './engine'
import { type Stage, stopSpotlightBlur, syncSpotlightBlur } from './spotlight-blur'

/** Which document a tour runs against. */
export type TourSurface = 'app' | 'preview'

/** The app around the tour, consumed straight from what the app already
 *  exposes: HashRouter means a route IS `location.hash`, and pane reveals are
 *  `revealDesktopPane` (the same path the focus_pane tool drives). Nothing is
 *  registered and nothing persists — a step reads these while it runs. */
const APP_HOST: TourHost = {
  currentRoute: () => window.location.hash.replace(/^#/, '') || '/',
  navigate: to => {
    window.location.hash = to
  },
  revealPane: pane => void revealDesktopPane(pane)
}

/** The app's own overlay treatment: a softly rounded cutout and a transition
 *  slow enough to read as travel between steps. The fade itself — scrim colour,
 *  strength, and the desaturation of everything outside the spotlight — lives
 *  in app-tour.css, so both layers are tuned in one place and follow the theme
 *  (light fades toward white, dark toward near-black). */
const TOUR_STYLE: TourStyle = {
  duration: 520,
  popoverOffset: 8,
  stagePadding: 6,
  stageRadius: 8
}

/** The one live tour, if any. This is the ONLY thing that outlives an action,
 *  and it holds a driver instance rather than a copy of any app state — the
 *  origin route a navigating tour restores lives in that instance's own
 *  closure, so it dies with the tour. `onDestroyed` empties this slot however
 *  the tour ended (Esc, ✕, overlay click, last step), leaving nothing behind. */
const appHolder: TourHolder = {}

/** Run one tour action on `surface`. Never throws — failures come back as
 *  `{success: false, error}` so a caller (or the agent) can recover. */
export async function runTour(action: TourAction, surface: TourSurface = 'app'): Promise<TourResult> {
  try {
    if (surface === 'preview') {
      return await runPreviewTour(action)
    }

    const result = runTourEngine(driverFactory, appHolder, action, collectTourTargets, document, TOUR_STYLE, APP_HOST)

    // Mirror driver.js's cutout onto a blur layer for as long as the tour is
    // up, and drop the whole thing once it isn't. The layer reads driver's own
    // per-frame stage rect, so it eases between steps exactly like the cutout.
    if (isTourActive()) {
      syncSpotlightBlur(() => (appHolder.driver?.getState?.('__activeStagePosition') as null | Stage) ?? null)
    } else {
      stopSpotlightBlur()
    }

    return result
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error), success: false }
  }
}

/** Whether a tour is currently on screen (app surface). The user can dismiss a
 *  tour with Esc, the ✕, or an overlay click, so a caller must never track that
 *  with its own flag — ask here. */
export const isTourActive = (): boolean => appHolder.driver?.isActive() ?? false

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
