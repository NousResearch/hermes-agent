import { atom } from 'nanostores'

import type { DesktopFirstRunState } from '@/global'

const INITIAL: DesktopFirstRunState = { required: false }

export const $firstRun = atom<DesktopFirstRunState>(INITIAL)

// Seed from main's current state (so a late-mounting renderer doesn't miss the
// waiting event) and subscribe to live changes. No-ops on the web build / older
// Electron without the surface. Returns an unsubscribe; safe to call once from
// the primary window at startup.
export function initFirstRunStore(): () => void {
  const firstRun = window.hermesDesktop?.firstRun

  if (!firstRun) {
    return () => {}
  }

  firstRun
    .get()
    .then(state => $firstRun.set(state))
    .catch(() => {
      // Older Electron without the handler — stay in the default (not required).
    })

  const off = firstRun.onChanged(state => $firstRun.set(state))

  return () => off?.()
}

// Resolve main's wait: proceed with the local install/bootstrap. Broadcasts
// {required:false}, which flips the overlay off; DesktopInstallOverlay then
// takes over via the bootstrap manifest event.
export async function chooseInstall(): Promise<void> {
  const firstRun = window.hermesDesktop?.firstRun

  if (!firstRun) {
    return
  }

  // Await for error handling only; onChanged is the single writer of $firstRun
  // (it already delivered the {required:false} transition this invoke triggers).
  await firstRun.choose('install')
}
