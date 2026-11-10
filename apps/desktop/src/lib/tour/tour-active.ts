import { atom } from 'nanostores'

/** Reactive twin of `isTourActive` for UI that must wait out a tour. Lives apart from
 *  run-tour.ts, which is dynamic-imported to keep driver.js off the boot path. */
export const $tourActive = atom(false)
