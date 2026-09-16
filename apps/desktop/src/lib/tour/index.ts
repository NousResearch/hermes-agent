/**
 * TOURS — highlight-and-narrate walkthroughs for any surface.
 *
 * Two consumers, one engine:
 *
 * - The `tour` agent tool, via the gateway (see run-tour.ts).
 * - Your own curated tours, via this module:
 *
 * ```ts
 * import { startTour, showTourStep, stopTour } from '@/lib/tour'
 *
 * startTour([
 *   { selector: '[data-tour="composer"]', title: 'Composer', text: 'Type here.' },
 *   { selector: '[data-tour="files"]', title: 'Files', text: 'Your project.' }
 * ])
 * ```
 *
 * Tours run against whatever is in the DOM — mark elements with `data-tour="…"`
 * to give them durable handles (see `collectTourTargets`, which reports those
 * first and flags every selector as stable or positional).
 */

export { collectTourTargets, type TourTarget } from './collect-targets'
export type { TourAction, TourHost, TourResult, TourStep } from './engine'
export {
  isTourActive,
  listTourTargets,
  nextTourStep,
  previousTourStep,
  runTour,
  showTourStep,
  startTour,
  stopTour,
  type TourSurface
} from './run-tour'
