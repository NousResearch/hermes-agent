import { atom } from 'nanostores'

import type { DesktopProfileRoute } from '@/global'
import { withTimeout } from '@/lib/with-timeout'

// Electron owns persistence and cross-window changes. This atom is only a
// mirror: changing the preference must not re-home an existing conversation.
export const $defaultProfileRoute = atom<DesktopProfileRoute | null>(null)
let revision = 0

function publish(route: DesktopProfileRoute | null): void {
  const current = $defaultProfileRoute.get()

  if (current?.connectionId !== route?.connectionId || current?.profile !== route?.profile) {
    $defaultProfileRoute.set(route)
  }
}

export async function refreshDefaultProfile(): Promise<DesktopProfileRoute | null> {
  const getDefault = window.hermesDesktop?.profile?.getDefault

  if (!getDefault) {
    return $defaultProfileRoute.get()
  }

  const requestRevision = ++revision
  const route = await withTimeout(getDefault(), 5_000, 'Timed out loading the default profile')

  if (requestRevision === revision) {
    publish(route)
  }

  return $defaultProfileRoute.get()
}

export async function setDefaultProfile(route: DesktopProfileRoute): Promise<DesktopProfileRoute> {
  const setDefault = window.hermesDesktop?.profile?.setDefault

  if (!setDefault) {
    throw new Error('This Desktop version cannot save a default profile.')
  }

  const requestRevision = ++revision
  const saved = await withTimeout(setDefault(route), 5_000, 'Timed out saving the default profile')

  if (requestRevision === revision) {
    publish(saved)
  }

  return saved
}

export function subscribeDefaultProfile(): (() => void) | undefined {
  return window.hermesDesktop?.profile?.onDefaultChanged?.(route => {
    revision += 1
    publish(route)
  })
}
