/**
 * The local-setup call to action — eligibility as data in, data out.
 *
 * The one campaign tip the app currently runs: a machine that could serve a
 * local model, on a backend that has none staged, gets one bubble on the
 * model pill saying so, with the button that starts the already-built
 * one-click setup. Everything here is a pure decision over fetched state so
 * the policy is testable without a DOM or a backend.
 *
 * Why this is not a rotation tip: the rotation teaches the app the user
 * already has, on a weeks-long walk. This tip is conditional (most machines
 * either qualify or don't, permanently), actionable (it carries a button),
 * and perishable (setting up local models — or ✕ — ends it forever). It
 * outranks the walk when live because "your GPU can do this" beats "the
 * model name is a button" every time both are true.
 */

import type { LocalCatalogModel, LocalModelsStatus } from '@/types/hermes'

/** Retirement/shown-at ledger id. Not a `TipId` — the rotation never walks it. */
export const LOCAL_SETUP_TIP_ID = 'local-setup'

/** A CTA ignored (timed out) may return, but on a much longer clock than the
 *  rotation's: it is the same message twice, not a tour moving on. */
export const LOCAL_SETUP_RESHOW_MS = 7 * 24 * 60 * 60_000

/**
 * Machines qualify when the catalog has a model that FITS (the backend's
 * physics check, same answer the pane's hero uses) and nothing is servable
 * yet — no runtime or no staged models. A set-up machine never qualifies, so
 * completing setup retires this tip without any bookkeeping.
 *
 * `connectionMode` must be 'local': on a remote backend (cloud resolves to
 * remote) the models would run on the far machine, and a bubble promising
 * "stays on your computer" would be promising someone else's computer.
 */
export function localSetupEligible(
  connectionMode: null | string,
  status: LocalModelsStatus | null,
  catalog: readonly LocalCatalogModel[] | null
): boolean {
  if (connectionMode !== 'local' || !status || !catalog) {
    return false
  }

  const needsSetup = !status.runtime_installed || status.models.length === 0

  return needsSetup && catalog.some(model => model.fits)
}

/** Due = never shown, or shown long enough ago that repeating it reads as a
 *  reminder rather than a nag. Retirement is the caller's ledger, not ours. */
export function localSetupDue(now: number, shownAt: number | undefined): boolean {
  return shownAt === undefined || now - shownAt >= LOCAL_SETUP_RESHOW_MS
}
