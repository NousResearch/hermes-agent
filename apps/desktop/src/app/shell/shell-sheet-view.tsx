import { useStore } from '@nanostores/react'
import { useEffect } from 'react'
import { useLocation } from 'react-router'

import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from '@/lib/escape-layers'

import { $openSheet, $shellSheet, closeOpenSheet, type OpenSheet } from './shell-sheet'

/** Where each sheet's content lives, for moving focus into it. */
const SHEET_SURFACE: Record<OpenSheet, string> = {
  nav: '[data-slot="global-nav"]',
  page: '[data-page-surface] :is(.page-sidebar-panel, [data-tour="overlay-nav"])',
  pane: '[data-narrow-overlay]',
  sessions: '[data-narrow-overlay]'
}

/** Where focus lands in a sheet: its selected row, else the sheet itself (not its first button,
 *  whose tooltip would open on focus and take the first Esc). */
function sheetFocusTarget(sheet: OpenSheet): HTMLElement | null {
  const surface = document.querySelector<HTMLElement>(SHEET_SURFACE[sheet])
  const current = surface?.querySelector<HTMLElement>('[aria-current="page"]')

  if (current || !surface) {
    return current ?? null
  }

  surface.tabIndex = -1

  return surface
}

/** Behaviour every redesign sheet shares (DESIGN.md › Responsive), mounted once by the redesigned
 *  shell: choosing a destination (any navigation) closes the sheet; focus moves into a sheet when it
 *  opens and returns to what opened it when it closes; Esc closes the band's own sheets (the pane
 *  tree's overlay owns its Esc). The page never scrolls behind a sheet: the shell is a fixed,
 *  viewport-sized layout, and sheets contain their own overscroll. */
export function ShellSheetController() {
  const { pathname, search } = useLocation()
  const sheet = useStore($openSheet)
  const shellSheet = useStore($shellSheet)
  const route = `${pathname}${search}`

  // Any navigation is a pick (mounting with nothing open is a no-op).
  useEffect(() => closeOpenSheet(), [route])

  useEffect(() => {
    if (!sheet) {
      return
    }

    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null
    // The sheet paints in this commit; focus once it has laid out.
    const frame = requestAnimationFrame(() => sheetFocusTarget(sheet)?.focus({ preventScroll: true }))

    return () => {
      cancelAnimationFrame(frame)

      // Only when focus would otherwise be lost (it was in the sheet, now gone): a pick that
      // navigated has already put focus where the new page wants it.
      if (opener?.isConnected && (document.activeElement === document.body || !document.activeElement)) {
        opener.focus({ preventScroll: true })
      }
    }
  }, [sheet])

  useEffect(() => {
    if (!shellSheet) {
      return
    }

    const release = pushEscapeLayer(ESCAPE_PRIORITY.sheet)

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !event.defaultPrevented && isTopEscapeLayer(ESCAPE_PRIORITY.sheet)) {
        event.preventDefault()
        closeOpenSheet()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => {
      window.removeEventListener('keydown', onKeyDown)
      release()
    }
  }, [shellSheet])

  return null
}

/** The scrim under the band's own sheets (a page panel, the compact rail). It renders where the
 *  sheet's stacking context is — inside a route page, which floats over the shell, or in the shell's
 *  band — and a tap on it closes the sheet. The pane tree's overlay draws its own. */
export function ShellSheetScrim() {
  const sheet = useStore($openSheet)

  if (sheet !== 'nav' && sheet !== 'page') {
    return null
  }

  return (
    <div
      aria-hidden="true"
      className="portal-sheet-scrim"
      data-slot="sheet-scrim"
      onPointerDown={event => {
        // A dismissal must not focus or click what lies under the scrim.
        event.preventDefault()
        closeOpenSheet()
      }}
    />
  )
}
