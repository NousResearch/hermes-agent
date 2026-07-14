/**
 * Chat backdrop image (the faint statue behind the transcript). One boolean,
 * on by default. Purely presentational renderer state — the `Backdrop`
 * component just skips rendering when it's off.
 */

import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.backdrop.v1'

export const $backdrop = atom<boolean>(typeof window === 'undefined' ? true : storedBoolean(KEY, true))

export function setBackdrop(on: boolean): void {
  $backdrop.set(on)
}

if (typeof window !== 'undefined') {
  $backdrop.subscribe(on => persistBoolean(KEY, on))
}
