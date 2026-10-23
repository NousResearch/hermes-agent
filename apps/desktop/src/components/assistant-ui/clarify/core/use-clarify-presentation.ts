// Tell the gateway once that this live card is mounted and usable.
// A hidden pane waits. Clicks, focus, scroll, and card size do not gate or cancel the receipt.
import { type RefObject, useEffect, useRef } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import type { ClarifyRequest } from '@/store/clarify'
import { clarifyPresentationReporter } from '@/store/server-requests'

export function useClarifyPresentation(
  formRef: RefObject<HTMLFormElement | null>,
  request: ClarifyRequest | null,
  disabled: boolean,
  undelivered: boolean
): void {
  const requestId = request?.requestId
  const report = clarifyPresentationReporter(requestId)
  const visible = usePaneVisible()
  const sentFor = useRef<string | undefined>(undefined)

  useEffect(() => {
    if (!requestId || sentFor.current === requestId) {
      return
    }

    const form = formRef.current

    if (!report || disabled || undelivered || !visible || !form?.isConnected) {
      return
    }

    sentFor.current = requestId
    report('presented')
  }, [disabled, formRef, report, requestId, undelivered, visible])
}
