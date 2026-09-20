/**
 * Device-local switch for the large-paste-to-attachment policy.
 *
 * The conversion itself (large-paste.ts) is the right default, but a long
 * specification or agent-task prompt is the prompt itself — for those users
 * an attachment chip is not an equivalent surface. This preference lives in
 * the renderer (same split as haptics: nothing in the main process needs it,
 * so no IPC round-trip), and the composer paste gate reads it live.
 */

import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const LARGE_PASTE_ATTACH_STORAGE_KEY = 'hermes.desktop.composer.largePasteAttach'

export const $largePasteAttachEnabled = atom(storedBoolean(LARGE_PASTE_ATTACH_STORAGE_KEY, true))

$largePasteAttachEnabled.subscribe(enabled => persistBoolean(LARGE_PASTE_ATTACH_STORAGE_KEY, enabled))

export function setLargePasteAttachEnabled(enabled: boolean): void {
  $largePasteAttachEnabled.set(enabled)
}
