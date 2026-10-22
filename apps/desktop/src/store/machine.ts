/** This device's machine facts, for showing its own hostname next to "This device" in the
 *  connection switcher. Loaded once on demand from the main process. */

import { atom } from 'nanostores'

import type { DesktopMachineProfile } from '@/global'

export const $machine = atom<DesktopMachineProfile | null>(null)

export async function loadMachineProfile(): Promise<void> {
  if ($machine.get()) {
    return
  }

  const profile = await window.hermesDesktop?.getMachineProfile?.().catch(() => null)

  if (profile) {
    $machine.set(profile)
  }
}
