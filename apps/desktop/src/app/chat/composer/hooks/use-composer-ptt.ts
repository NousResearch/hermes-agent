import { useEffect, useRef } from 'react'

interface ComposerPttOptions {
  active: () => boolean
  blocked: boolean
  cancel: () => void
  maxRecordingSeconds: number
  start: () => Promise<boolean | void>
  stop: () => Promise<string | null>
  submit: (text: string) => Promise<unknown> | unknown
}

/** Composer-local hold-to-talk. Left Alt is observed but never globally consumed. */
export function useComposerPtt({
  active,
  blocked,
  cancel,
  maxRecordingSeconds,
  start,
  stop,
  submit
}: ComposerPttOptions) {
  const heldRef = useRef(false)
  const recordingRef = useRef(false)
  const finishPendingRef = useRef(false)
  const generationRef = useRef(0)
  const timeoutRef = useRef<number | null>(null)
  const blockedRef = useRef(blocked)
  const abandonRef = useRef<() => void>(() => undefined)
  const callbacksRef = useRef({ active, cancel, start, stop, submit })

  blockedRef.current = blocked
  callbacksRef.current = { active, cancel, start, stop, submit }

  // eslint-disable-next-line no-restricted-syntax -- stable window listeners read the latest composer callbacks from refs
  useEffect(() => {
    const clearTimer = () => {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
    }

    const abandon = () => {
      if (!heldRef.current && !recordingRef.current && !finishPendingRef.current) {
        return
      }

      generationRef.current += 1
      heldRef.current = false
      recordingRef.current = false
      finishPendingRef.current = false
      clearTimer()
      callbacksRef.current.cancel()
    }

    abandonRef.current = abandon

    const finish = () => {
      if (!recordingRef.current) {
        return
      }

      const generation = generationRef.current
      heldRef.current = false
      recordingRef.current = false
      finishPendingRef.current = true
      clearTimer()
      void callbacksRef.current
        .stop()
        .then(text => {
          const transcript = text?.trim()

          if (transcript && generation === generationRef.current && callbacksRef.current.active()) {
            return callbacksRef.current.submit(transcript)
          }

          return undefined
        })
        .catch(() => undefined)
        .finally(() => {
          if (generation === generationRef.current) {
            finishPendingRef.current = false
          }
        })
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.code !== 'AltLeft' ||
        event.repeat ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        blockedRef.current ||
        heldRef.current ||
        finishPendingRef.current ||
        !callbacksRef.current.active()
      ) {
        return
      }

      const generation = generationRef.current + 1
      generationRef.current = generation
      heldRef.current = true
      void callbacksRef.current
        .start()
        .then(started => {
          if (started === false) {
            heldRef.current = false

            return
          }

          if (
            heldRef.current &&
            generation === generationRef.current &&
            !blockedRef.current &&
            callbacksRef.current.active()
          ) {
            recordingRef.current = true
            const cap = Math.max(1, Math.min(Math.trunc(maxRecordingSeconds), 600))
            timeoutRef.current = window.setTimeout(finish, cap * 1000)
          } else if (generation === generationRef.current) {
            heldRef.current = false
            callbacksRef.current.cancel()
          }
        })
        .catch(() => {
          if (generation === generationRef.current) {
            heldRef.current = false
          }
        })
    }

    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code !== 'AltLeft' || !heldRef.current) {
        return
      }

      heldRef.current = false
      finish()
    }

    const onFocusIn = () => {
      if (!callbacksRef.current.active()) {
        abandon()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    window.addEventListener('blur', abandon)
    document.addEventListener('focusin', onFocusIn)

    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('blur', abandon)
      document.removeEventListener('focusin', onFocusIn)
      abandon()

      if (abandonRef.current === abandon) {
        abandonRef.current = () => undefined
      }
    }
  }, [maxRecordingSeconds])

  useEffect(() => {
    if (blocked) {
      abandonRef.current()
    }
  }, [blocked])
}
