import { useEffect, useRef } from 'react'

import { isMacPlatform } from '@/lib/platform'

import { createHoldCommandDictation } from './hold-command-dictation'

const HOLD_DELAY_MS = 180

interface UseHoldCommandDictationOptions {
  disabled: boolean
  enabled: boolean
  start: () => void
  stop: () => void
}

/** Starts dictation after holding Command alone, without claiming Command chords. */
export function useHoldCommandDictation({ disabled, enabled, start, stop }: UseHoldCommandDictationOptions) {
  const holdRef = useRef(createHoldCommandDictation())

  useEffect(() => {
    if (!enabled || disabled || !isMacPlatform()) {return}

    let timer: number | undefined

    const clearTimer = () => {
      if (timer !== undefined) {
        window.clearTimeout(timer)
        timer = undefined
      }
    }

    const apply = (action: 'arm' | 'cancel' | 'ignore' | 'start' | 'stop') => {
      if (action === 'start') {start()}
      else if (action === 'stop') {stop()}
    }

    const onKeyDown = (event: KeyboardEvent) => {
      const action = holdRef.current.keyDown(event)

      if (action === 'arm') {
        clearTimer()
        timer = window.setTimeout(() => {
          timer = undefined
          apply(holdRef.current.begin())
        }, HOLD_DELAY_MS)
      } else {
        clearTimer()
        apply(action)
      }
    }

    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key === 'Meta') {clearTimer()}
      apply(holdRef.current.keyUp(event))
    }

    const onBlur = () => {
      clearTimer()
      apply(holdRef.current.cancel())
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })
    window.addEventListener('keyup', onKeyUp, { capture: true })
    window.addEventListener('blur', onBlur)

    return () => {
      window.removeEventListener('keydown', onKeyDown, { capture: true })
      window.removeEventListener('keyup', onKeyUp, { capture: true })
      window.removeEventListener('blur', onBlur)
      onBlur()
    }
  }, [disabled, enabled, start, stop])
}
