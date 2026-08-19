/**
 * PREVIEW TOUR — runs tour actions inside the preview pane's guest page, so a
 * tour can walk through ANY web app open in the in-app browser, not just
 * Hermes itself.
 *
 * The guest page is out-of-process; nothing here can touch its DOM directly.
 * Instead the first action injects a self-contained bundle over
 * `executeJavaScript` — the vendored driver.js IIFE, its stylesheet, and the
 * same engine/collector SOURCE the app surface runs (see lib/tour/engine.ts's
 * self-containment contract) — parked on window globals so subsequent actions
 * reuse the live driver instance. Injection is idempotent and vanishes with
 * the page (a navigation resets the tour, which is the right behavior).
 *
 * Dynamic-imported by run-tour.ts so the raw driver.js payload stays out of
 * the boot path.
 */

import driverCss from 'driver.js/dist/driver.css?raw'
import driverIife from 'driver.js/dist/driver.js.iife.js?raw'

import { collectTourTargets } from '@/lib/tour/collect-targets'
import { runTourEngine, type TourAction, type TourResult } from '@/lib/tour/engine'

import { activeTourRunner } from './preview-tour-runner'

/** Build the idempotent inject-and-run script for one tour action. */
function buildTourScript(action: TourAction): string {
  return `(function () {
  var w = window;
  if (!w.__hermesTourEngine) {
    ${driverIife}
    w.__hermesTourHolder = {};
    w.__hermesTourCollect = (${collectTourTargets.toString()});
    w.__hermesTourEngine = (${runTourEngine.toString()});
  }
  if (!document.getElementById('__hermes-tour-style')) {
    var style = document.createElement('style');
    style.id = '__hermes-tour-style';
    style.textContent = ${JSON.stringify(driverCss)};
    (document.head || document.documentElement).appendChild(style);
  }
  return JSON.stringify(w.__hermesTourEngine(
    w.driver.js.driver,
    w.__hermesTourHolder,
    ${JSON.stringify(action)},
    w.__hermesTourCollect,
    document
  ));
})()`
}

/** Run one tour action in the ACTIVE preview tab's page. */
export async function runPreviewTour(action: TourAction): Promise<TourResult> {
  const run = activeTourRunner()

  if (!run) {
    return { error: 'No live page is open in the preview pane — open one first.', success: false }
  }

  const raw = await run(buildTourScript(action))

  if (typeof raw !== 'string' || !raw) {
    return { error: 'The page did not answer the tour action.', success: false }
  }

  return JSON.parse(raw) as TourResult
}
