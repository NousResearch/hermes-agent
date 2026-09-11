import { atom } from 'nanostores'
import { useEffect } from 'react'

import { isMacPlatform } from '@/lib/platform'
import { $capture } from '@/store/keybinds'

export const $heldTabModifier = atom(false)

/** One delayed observer; it never consumes or dispatches a key. */
export function useTabKeyHints() {
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined
    let down = false

    const clear = () => {
      clearTimeout(timer)
      down = false
      $heldTabModifier.set(false)
    }

    const sync = (event: KeyboardEvent) => {
      const held = isMacPlatform() ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey

      if (!held || event.altKey || event.shiftKey || event.isComposing || $capture.get()) {
        clear()
      } else if (!down) {
        down = true
        timer = setTimeout(() => $heldTabModifier.set(true), 400)
      }
    }

    window.addEventListener('keydown', sync, true)
    window.addEventListener('keyup', sync, true)
    window.addEventListener('blur', clear)
    const stopCapture = $capture.listen(clear)

    return () => {
      clear()
      stopCapture()
      window.removeEventListener('keydown', sync, true)
      window.removeEventListener('keyup', sync, true)
      window.removeEventListener('blur', clear)
    }
  }, [])
}
