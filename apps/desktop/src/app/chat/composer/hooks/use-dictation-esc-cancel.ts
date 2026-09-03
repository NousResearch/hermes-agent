import { useEffect, useRef } from 'react'

import { triggerHaptic } from '@/lib/haptics'

import { type ComposerTarget, getActiveComposer } from '../focus'

interface UseDictationEscCancelOptions {
  /** True only while audio is being captured. Transcription in flight is past
   *  the point of recall — the audio already reached the provider. */
  recording: boolean
  onCancel: () => void
  /** This composer's focus-bus key, so only the active composer reacts. */
  target: ComposerTarget
}

/**
 * Esc discards an in-flight dictation.
 *
 * Precedence matters here: `useComposerEscCancel` already claims Esc to halt a
 * running agent turn, and DESIGN.md requires that one cancel gesture does one
 * thing. This handler is deliberately the narrower claim — it only acts while
 * the microphone is actually capturing, and it marks the event handled
 * (`preventDefault`) so the turn-cancel listener bails on its own
 * `defaultPrevented` check. In practice the two rarely overlap (dictation is a
 * draft-time action, turn-cancel a busy-time one), but when they do, discarding
 * the recording the user is staring at beats stopping a background turn.
 *
 * Unlike the turn-cancel handler this does NOT bail when focus sits in a text
 * field: dictation is normally started from the focused composer input, so
 * requiring focus to leave the input first would make the shortcut unreachable
 * exactly when it is needed.
 */
export function useDictationEscCancel({ onCancel, recording, target }: UseDictationEscCancelOptions) {
  const handlerRef = useRef<(event: globalThis.KeyboardEvent) => void>(() => {})

  handlerRef.current = (event: globalThis.KeyboardEvent) => {
    if (event.key !== 'Escape' || event.defaultPrevented || !recording) {
      return
    }

    if (getActiveComposer() !== target) {
      return
    }

    // An open dialog, popover, or completion listbox owns Esc — it must close
    // that surface, never reach past it to the composer underneath. The trigger
    // popover is a plain `role="listbox"`, not a radix popper, so it needs its
    // own entry here.
    if (
      document.querySelector(
        '[role="dialog"],[role="alertdialog"],[role="listbox"],[data-radix-popper-content-wrapper]'
      )
    ) {
      return
    }

    event.preventDefault()
    triggerHaptic('cancel')
    onCancel()
  }

  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent) => handlerRef.current(event)
    // Capture phase: the turn-cancel listener is registered on `window` too, and
    // bubble-phase order would depend on hook call order. Capturing makes the
    // precedence explicit and stable — we see Esc first, and when we consume it
    // the turn-cancel handler bails on `defaultPrevented`.
    window.addEventListener('keydown', onKeyDown, true)

    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [])
}
