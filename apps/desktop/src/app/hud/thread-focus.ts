import { type RefObject, useEffect } from 'react'

import { focusComposerInput } from '@/app/chat/composer/focus'
import { RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'

const THREAD_INTERACTIVE = 'a[href], button, input, textarea, select, [contenteditable], [role="button"], [role="link"]'

/**
 * In HUD mode the band is for reading over another app — clicking a line is not
 * leaving the composer. Without this, mousedown on the scrollback blurs the
 * input, the band fades, and the focus ring vanishes mid-read.
 */
export function useHudThreadFocus(rootRef: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const root = rootRef.current

    if (!root) {
      return
    }

    const onPointerDown = (event: PointerEvent) => {
      if (event.button !== 0) {
        return
      }

      const target = event.target

      if (!(target instanceof Element)) {
        return
      }

      if (!target.closest('[data-slot="composer-bounds"]')) {
        return
      }

      if (target.closest(THREAD_INTERACTIVE)) {
        return
      }

      event.preventDefault()

      focusComposerInput(root.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`))
    }

    root.addEventListener('pointerdown', onPointerDown, true)

    return () => root.removeEventListener('pointerdown', onPointerDown, true)
  }, [rootRef])
}
