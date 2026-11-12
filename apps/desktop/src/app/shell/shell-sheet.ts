import { atom, computed } from 'nanostores'

import { $widthClass } from '@/app/redesign/breakpoints'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { $narrowOverlayPaneIds } from '@/components/pane-shell/narrow-overlay-state'
import { CHAT_SIDEBAR_PANE_ID } from '@/store/layout'
import { $portalShell } from '@/store/shell-variant'

/** A sheet the redesign's band opens below wide (DESIGN.md › Responsive): a page's sidebar panel
 *  ('page': Agents, Schedules, Settings, an agent's settings menu) or, on compact, the rail alone
 *  ('nav': a page with no panel still needs its destinations). The chat sidebar and the developer
 *  panes are the pane tree's own narrow overlays; `$openSheet` names those too. */
export type ShellSheet = 'nav' | 'page'

export type OpenSheet = ShellSheet | 'pane' | 'sessions'

export const $shellSheet = atom<ShellSheet | null>(null)

export function openShellSheet(sheet: ShellSheet): void {
  $shellSheet.set(sheet)
}

export function toggleShellSheet(sheet: ShellSheet): void {
  $shellSheet.set($shellSheet.get() === sheet ? null : sheet)
}

/** The sheet on screen, if any. Wide windows dock everything, so they never show one. */
export const $openSheet = computed(
  [$portalShell, $widthClass, $shellSheet, $narrowOverlayPaneIds],
  (portal, width, sheet, overlays): OpenSheet | null => {
    if (!portal || width === 'wide') {
      return null
    }

    if (sheet) {
      return sheet
    }

    if (overlays.has(CHAT_SIDEBAR_PANE_ID)) {
      return 'sessions'
    }

    return overlays.size > 0 ? 'pane' : null
  }
)

/** Close whatever sheet is open: the band's own, or the pane tree's overlay through its reveal event
 *  (the tree owns that state; an explicit close never re-opens it). */
export function closeOpenSheet(): void {
  if ($shellSheet.get()) {
    $shellSheet.set(null)

    return
  }

  const [id] = $narrowOverlayPaneIds.get()

  if (id && typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id, mode: 'close' } }))
  }
}

// Crossing to wide docks the panel again; a shell flip leaves no redesign sheet behind.
const dropWhenDocked = () => {
  if (!$portalShell.get() || $widthClass.get() === 'wide') {
    $shellSheet.set(null)
  }
}

$widthClass.listen(dropWhenDocked)
$portalShell.listen(dropWhenDocked)

/** Mirrors the open sheet onto `<html data-portal-sheet>` so CSS can lift the rail into a compact
 *  sheet and stand the band's clusters down under the scrim. Called once at boot. */
export function installShellSheetAttribute(root: HTMLElement = document.documentElement): () => void {
  return $openSheet.subscribe(sheet => {
    if (sheet) {
      root.dataset.portalSheet = sheet
    } else {
      delete root.dataset.portalSheet
    }
  })
}
