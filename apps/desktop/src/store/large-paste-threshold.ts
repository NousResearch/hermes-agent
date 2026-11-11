import { atom } from 'nanostores'

import {
  LARGE_PASTE_ATTACHMENT_THRESHOLD,
  normalizeLargePasteAttachmentThreshold
} from '@/app/chat/composer/large-paste'
import { persistString, storedString } from '@/lib/storage'

// Desktop-local composer behavior, independent of profiles and connections.
const KEY = 'hermes.desktop.large-paste-attachment-threshold.v1'

export const $largePasteAttachmentThreshold = atom(normalizeLargePasteAttachmentThreshold(storedString(KEY)))

export function setLargePasteAttachmentThreshold(value: number): void {
  const threshold = normalizeLargePasteAttachmentThreshold(value)
  $largePasteAttachmentThreshold.set(threshold)
  persistString(KEY, threshold === LARGE_PASTE_ATTACHMENT_THRESHOLD ? null : String(threshold))
}

if (typeof window !== 'undefined') {
  // Other windows share storage, but each renderer owns its own atom. Received
  // changes update only the atom so they cannot trigger a persistence loop.
  window.addEventListener('storage', event => {
    if (event.key === KEY || event.key === null) {
      $largePasteAttachmentThreshold.set(normalizeLargePasteAttachmentThreshold(storedString(KEY)))
    }
  })
}
