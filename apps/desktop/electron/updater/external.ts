// updater/external.ts — the steward-owned strategy.
//
// Store deployments (process.windowsStore) and stamps whose updateMechanism
// is 'external': the steward (Microsoft Store / App Installer on-launch
// re-check) owns the update loop. The app does not check, does not apply —
// it only tells the user updates happen outside the app.

import type { UpdaterApplyResultWire, UpdaterStatusWire } from './index'

export const EXTERNAL_UNSUPPORTED_MESSAGE =
  'bundled install: updates are applied by the installer (Microsoft Store or App Installer).'

export class ExternalStrategy {
  readonly mechanism = 'external' as const
  readonly supported = false

  async check(): Promise<UpdaterStatusWire> {
    return {
      supported: false,
      mechanism: 'external',
      reason: 'bundled-not-appinstaller',
      message: EXTERNAL_UNSUPPORTED_MESSAGE,
      fetchedAt: Date.now()
    }
  }

  async apply(): Promise<UpdaterApplyResultWire> {
    return { ok: true, manual: true, bundled: true, mechanism: 'external' }
  }
}
