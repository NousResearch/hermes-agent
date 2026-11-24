/**
 * Layout edit mode — the shared toggle for the tree renderer's FancyZones-style
 * rearrangement (see tree/renderer.tsx). The toggle hotkey is a `keybinds`
 * contribution (`layout.editMode`, default ⌘⇧\ — the sibling of ⌘\ = flip
 * panes), so it's rebindable and collision-checked like every other action.
 * This hook only owns Escape-to-exit.
 */

import { atom, computed } from 'nanostores'
import { useEffect } from 'react'

import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from '@/lib/escape-layers'
import { $showsAdvancedChrome } from '@/store/interface-mode'

export const $layoutEditMode = atom(false)

// Edit mode force-shows toggle-hidden panes (terminal off, review closed) so
// they can be rearranged. In Simple those panes are resting by policy, not by
// a toggle, and arranging machinery you cannot see is not the job — edit mode
// there works on what Simple shows. The renderers read this, never
// `$layoutEditMode`, for the hidden-pane question.
export const $layoutEditRevealsHidden = computed(
  [$layoutEditMode, $showsAdvancedChrome],
  (editMode, advanced) => editMode && advanced
)

export function toggleLayoutEditMode() {
  $layoutEditMode.set(!$layoutEditMode.get())
}

/** Escape exits edit mode. Registered once by the layout root. */
export function useLayoutEditHotkey(enabled: boolean) {
  useEffect(() => {
    if (!enabled || typeof window === 'undefined') {
      return
    }

    // Own an Escape layer only WHILE edit mode is on, so it doesn't outrank the
    // narrow-pane reveal the rest of the time.
    let releaseLayer: (() => void) | null = null

    const unsub = $layoutEditMode.subscribe(on => {
      if (on && !releaseLayer) {
        releaseLayer = pushEscapeLayer(ESCAPE_PRIORITY.layoutEdit)
      } else if (!on && releaseLayer) {
        releaseLayer()
        releaseLayer = null
      }
    })

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented || !isTopEscapeLayer(ESCAPE_PRIORITY.layoutEdit)) {
        return
      }

      if ($layoutEditMode.get()) {
        e.preventDefault()
        $layoutEditMode.set(false)
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => {
      window.removeEventListener('keydown', onKeyDown)
      unsub()
      releaseLayer?.()
    }
  }, [enabled])
}
