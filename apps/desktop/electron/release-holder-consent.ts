// The Desktop half of the consented "close it and retry" for a Desktop app swap blocked by a
// program outside Hermes (#101789). The next promotion swap (hermes_cli/desktop_release_holders.py)
// reads this file, closes only processes running one of these programs whose working folder is
// still inside release/, and removes the file: one consent, one retry. Without it nothing is closed.

import fs from 'node:fs'
import path from 'node:path'

import type { ReleaseHolder } from './handoff-result'

export const RELEASE_HOLDER_CONSENT_FILE = '.close-holders-consent.json'

export function releaseHolderConsentPath(updateRoot: string): string {
  return path.join(updateRoot, 'apps', 'desktop', 'release', RELEASE_HOLDER_CONSENT_FILE)
}

export function writeReleaseHolderConsent(
  updateRoot: string,
  holders: ReleaseHolder[],
  now: () => number = Date.now
): string {
  const file = releaseHolderConsentPath(updateRoot)
  const exes = [...new Set(holders.map(holder => holder.exe).filter(Boolean))]
  fs.writeFileSync(file, JSON.stringify({ written_at: now() / 1000, exes }), 'utf8')

  return file
}

export interface ReleaseHolderRetryDeps {
  isWindows: boolean
  resolveUpdateRoot: () => string
  applyUpdates: () => Promise<{ ok: boolean; error?: string; message?: string }>
  log: (line: string) => void
}

/**
 * The dialog's "Close <program> and retry": record the consent, then run the update hand-off
 * again, which rebuilds and promotes the owed app (applyBody runs even when the checkout is
 * already current). A consent that cannot be written starts no retry.
 */
export function releaseHolderRetry(deps: ReleaseHolderRetryDeps): ((holders: ReleaseHolder[]) => void) | undefined {
  if (!deps.isWindows) {
    return undefined // the cwd hold is a Windows rename refusal; no other host reports holders
  }

  return holders => {
    const names = holders.map(holder => holder.name).join(', ')

    try {
      writeReleaseHolderConsent(deps.resolveUpdateRoot(), holders)
    } catch (error) {
      deps.log(`[updates] could not record consent to close ${names}: ${(error as Error).message}`)

      return
    }

    deps
      .applyUpdates()
      .then(result => {
        if (!result.ok) {
          deps.log(
            `[updates] retry after consenting to close ${names} did not start: ${result.error ?? result.message}`
          )
        }
      })
      .catch(error =>
        deps.log(`[updates] retry after consenting to close ${names} failed: ${(error as Error).message}`)
      )
  }
}
