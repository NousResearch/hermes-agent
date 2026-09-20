import { atom } from 'nanostores'

/**
 * Why this launch is compositing on the CPU, or null when the GPU is doing it.
 *
 * Main works it out before `ready` from the display it finds itself on — an SSH
 * session, VNC, RDP — and turns hardware acceleration off there, because a
 * remote display flickers with it on (`electron/main.ts`). It cannot change
 * while the app runs, so it is asked for once and read everywhere after.
 *
 * Null until the answer arrives, which is the right default: the GPU path is
 * the common one, and the effects that consult this are the ones that are free
 * on a GPU and ruinous without one.
 */
export const $remoteDisplayReason = atom<null | string>(null)

/** Ask main once. Resolves with the reason, so the caller can report it too. */
export async function loadRemoteDisplayReason(): Promise<null | string> {
  const reason = (await window.hermesDesktop?.getRemoteDisplayReason?.()) ?? null

  $remoteDisplayReason.set(reason)

  return reason
}
