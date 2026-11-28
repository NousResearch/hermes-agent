import { useEffect } from 'react'

import {
  ackComposerAttachPastedText,
  onComposerAttachPastedTextRequest,
  type ComposerTarget
} from '../focus'

interface UsePastedTextAttachInput {
  /** The consumer's own `onAttachPastedText`; absent on surfaces that cannot attach. */
  onAttachPastedText?: (text: string) => boolean | Promise<boolean>
  /** Scope of the composer this surface owns; a request aimed elsewhere is refused fast. */
  target: ComposerTarget
  onAttached?: () => void
}

/**
 * Subscribe a composer surface to unfocused large-paste attach requests.
 *
 * `paste-to-focus` routes a paste too big to inline through
 * `requestComposerAttachPastedText`, which dispatches to whichever surface owns
 * the composer target. The owner runs its own `onAttachPastedText` — the same
 * `large-paste.ts` threshold a focused paste obeys — and acks the result. A
 * request for a different target is acked `false` immediately rather than
 * holding the caller until the reply timeout.
 */
export function usePastedTextAttach({
  onAttachPastedText,
  onAttached,
  target
}: UsePastedTextAttachInput): void {
  useEffect(() => {
    if (!onAttachPastedText) {
      return undefined
    }

    return onComposerAttachPastedTextRequest(request => {
      if (request.target !== target) {
        ackComposerAttachPastedText(request.token, false)

        return
      }

      void Promise.resolve(onAttachPastedText(request.text))
        .then(attached => {
          if (attached) {
            onAttached?.()
          }

          ackComposerAttachPastedText(request.token, attached === true)
        })
        .catch(() => ackComposerAttachPastedText(request.token, false))
    })
  }, [onAttachPastedText, onAttached, target])
}
