// The Desktop half of the consented "close it and retry" for a Desktop app swap blocked by a
// program outside Hermes (#101789). The consent is bound to the ONE hand-off run the retry starts
// (its run id reaches the swap as HERMES_UPDATE_CORRELATION_ID) and to the exact processes the
// user saw (pid + start time + program). The next promotion swap (hermes_cli/desktop_release_holders.py)
// removes the file whatever its source and acts on it only in that run; a retry that never hands
// off removes it here, and the hand-off script removes one its run never used.

import fs from 'node:fs'
import path from 'node:path'

import type { ReleaseHolder } from './handoff-result'

export const RELEASE_HOLDER_CONSENT_FILE = '.close-holders-consent.json'

export function releaseHolderConsentPath(updateRoot: string): string {
  return path.join(updateRoot, 'apps', 'desktop', 'release', RELEASE_HOLDER_CONSENT_FILE)
}

/** Holders the user can close from here: a protected process (no readable program path) cannot be. */
export function closableReleaseHolders(holders: ReleaseHolder[]): ReleaseHolder[] {
  return holders.filter(holder => holder.exe)
}

// Set by the dialog's click for the apply it starts; taken by that apply's hand-off plan.
let pendingConsent: ReleaseHolder[] | null = null

/**
 * Called by the Windows script hand-off once its run id exists, before the script is spawned:
 * writes the consent given for this apply (if any) bound to that run, at most once. A consent
 * that cannot be recorded is logged; the retry then closes nothing.
 */
export function bindReleaseHolderConsent(
  updateRoot: string,
  runId: string | null,
  log: (line: string) => void,
  now: () => number = Date.now
): void {
  const holders = pendingConsent
  pendingConsent = null

  if (!holders?.length || !runId) {
    return // a legacy (run-id-less) hand-off cannot be told apart from any other update
  }

  try {
    fs.writeFileSync(
      releaseHolderConsentPath(updateRoot),
      JSON.stringify({
        written_at: now() / 1000,
        run_id: runId,
        holders: holders.map(holder => ({ pid: holder.pid, start_time: holder.startTime, exe: holder.exe }))
      }),
      'utf8'
    )
  } catch (error) {
    log(`[updates] could not record the consent to close a Desktop folder holder: ${(error as Error).message}`)
  }
}

export interface ReleaseHolderRetryDeps {
  isWindows: boolean
  resolveUpdateRoot: () => string
  applyUpdates: () => Promise<{ ok: boolean; handedOff?: boolean; error?: string; message?: string }>
  log: (line: string) => void
}

/**
 * The dialog's "Close <program> and retry": arm the consent, then run the update hand-off again,
 * which rebuilds and promotes the owed app (applyBody runs even when the checkout is already
 * current). An apply that does not hand off leaves no consent behind.
 */
export function releaseHolderRetry(deps: ReleaseHolderRetryDeps): ((holders: ReleaseHolder[]) => void) | undefined {
  if (!deps.isWindows) {
    return undefined // the cwd hold is a Windows rename refusal; no other host reports holders
  }

  return holders => {
    const closable = closableReleaseHolders(holders)
    const names = closable.map(holder => holder.name).join(', ')
    const updateRoot = deps.resolveUpdateRoot()
    pendingConsent = closable

    const forget = (reason: string) => {
      pendingConsent = null

      try {
        fs.rmSync(releaseHolderConsentPath(updateRoot), { force: true })
      } catch (error) {
        deps.log(`[updates] could not remove the consent to close ${names}: ${(error as Error).message}`)
      }

      deps.log(`[updates] retry after consenting to close ${names} ${reason}`)
    }

    deps
      .applyUpdates()
      .then(result => {
        if (!result.ok || !result.handedOff) {
          forget(`did not hand off: ${result.error ?? result.message ?? 'no hand-off'}`)
        }
      })
      .catch(error => forget(`failed: ${(error as Error).message}`))
  }
}
