import { useEffect, useRef } from 'react'

import { initQuickEntryBridge, setQuickEntrySubmitHandler } from '@/store/quick-entry'
import { isSecondaryWindow } from '@/store/windows'

interface QuickEntryBridgeParams {
  submitText: (text: string) => Promise<unknown> | unknown
}

/**
 * Wires the global-hotkey Quick Entry window back into the app: text captured
 * there is submitted through THIS window's normal prompt path (`submitText`), so
 * there is exactly one submit pipeline and no bespoke gateway RPC.
 *
 * The handler registers ONCE through a ref tracking the latest callback —
 * re-registering on identity churn leaves a nulled-handler window that can drop
 * a submit (the same bug shape use-pet-bridge guards). Primary window only: a
 * secondary session window must not also claim the global capture channel, or
 * one keystroke would send N prompts.
 */
export function useQuickEntryBridge({ submitText }: QuickEntryBridgeParams): void {
  const submitTextRef = useRef(submitText)
  submitTextRef.current = submitText

  useEffect(() => {
    if (isSecondaryWindow()) {
      return
    }

    setQuickEntrySubmitHandler(text => void submitTextRef.current(text))
    const dispose = initQuickEntryBridge()

    return () => {
      setQuickEntrySubmitHandler(null)
      dispose()
    }
  }, [])
}
