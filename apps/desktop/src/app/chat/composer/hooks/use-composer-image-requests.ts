import { useEffect } from 'react'

import { triggerHaptic } from '@/lib/haptics'

import { type ComposerTarget, onComposerAttachImagesRequest } from '../focus'
import type { ChatBarProps } from '../types'

/**
 * Paste-to-focus: clipboard images from an unfocused ⌘V ride the bus (the window dispatcher has no
 * handle on this composer's attachment scope). Same ingestion as a focused paste's image branch,
 * for the composer at `target` only.
 */
export function useComposerImageRequests(target: ComposerTarget, onAttachImageBlob: ChatBarProps['onAttachImageBlob']) {
  useEffect(() => {
    if (!onAttachImageBlob) {
      return undefined
    }

    return onComposerAttachImagesRequest(({ blobs, target: requested }) => {
      if (requested !== target) {
        return
      }

      triggerHaptic('selection')

      for (const blob of blobs) {
        void onAttachImageBlob(blob)
      }
    })
  }, [onAttachImageBlob, target])
}
