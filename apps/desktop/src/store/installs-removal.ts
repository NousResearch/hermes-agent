import { atom } from 'nanostores'

import type { DesktopInstallsRemoveResult } from '@/global'
import { translateNow } from '@/i18n'
import { notify } from '@/store/notifications'

export type InstallsRemovalOutcome = { id: string; ok: true } | { id: string; ok: false; message: string }

export interface InstallsRemovalState {
  /** The install being removed, or null. One removal runs at a time. */
  removingId: null | string
  /** Counts finished removals, so a mounted Installs page knows to reload its list. */
  finished: number
  /** The last removal's result, shown inline on the page. */
  outcome: InstallsRemovalOutcome | null
}

/**
 * A removal can take minutes (a Windows package), so its state lives here and not in
 * the Installs page: leaving the page must not lose the result.
 */
export const $installsRemoval = atom<InstallsRemovalState>({ removingId: null, finished: 0, outcome: null })

/** Runs `remove` for `id` and reports the result with a toast on every completion. */
export async function removeInstall(
  remove: (id: string) => Promise<DesktopInstallsRemoveResult>,
  id: string
): Promise<void> {
  if ($installsRemoval.get().removingId !== null) {
    return
  }

  $installsRemoval.set({ ...$installsRemoval.get(), removingId: id, outcome: null })

  let outcome: InstallsRemovalOutcome

  try {
    const result = await remove(id)

    outcome = result.ok
      ? { id, ok: true }
      : { id, ok: false, message: result.message || result.error || translateNow('settings.installsPage.removeFailed') }
  } catch (error) {
    outcome = { id, ok: false, message: error instanceof Error ? error.message : String(error) }
  }

  const { finished } = $installsRemoval.get()

  $installsRemoval.set({ removingId: null, finished: finished + 1, outcome })

  notify(
    outcome.ok
      ? { id: 'installs.removal', kind: 'success', message: translateNow('settings.installsPage.removed', id) }
      : { id: 'installs.removal', kind: 'error', message: outcome.message }
  )
}
