/**
 * Per-preview strip tools — the console toggle and the DevTools toggle.
 *
 * These were titlebar tools (`setTitlebarToolGroup`): one global pair, far from
 * the thing they act on and ambiguous the moment two previews were open. They're
 * strip glyphs now, contributed as `PaneStripTool` DATA — the strip renders them
 * with `PaneStripGlyph`, the same button the "+" is, so there is no preview-owned
 * styling to drift.
 *
 * The state has to outlive any one pane render and be addressable by tab id: the
 * console store is created lazily per tab and cached here, and `PreviewPane`
 * registers its DevTools handle for the tab it renders. Both invalidate the
 * strip so `active` / `disabled` stay truthful.
 */

import { invalidateStripTools } from '@/components/pane-shell/tree/store'
import { Codicon } from '@/components/ui/codicon'
import type { PaneStripTool } from '@/components/ui/pane-tab'
import { translateNow } from '@/i18n'

import { createPreviewConsoleState, type PreviewConsoleState } from './preview-console-state'

interface DevToolsHandle {
  open: boolean
  toggle: () => void
}

const consoleStates = new Map<string, PreviewConsoleState>()

/** The console store for a tab, created on first use. Cached so the strip glyph
 *  and the panel in the pane read the SAME store. */
export function previewConsoleState(tabId: string): PreviewConsoleState {
  const existing = consoleStates.get(tabId)

  if (existing) {
    return existing
  }

  const created = createPreviewConsoleState()

  // Opening/closing the console flips the glyph's `active`, and the strip reads
  // its tools during render — so tell it to re-read.
  created.$open.listen(() => invalidateStripTools())
  consoleStates.set(tabId, created)

  return created
}

/** DevTools handles by tab id. The pane registers on mount (the webview only
 *  exists in there); until it lands the glyph renders disabled. */
const devToolsHandles = new Map<string, DevToolsHandle>()

export function registerPreviewDevTools(tabId: string, handle: DevToolsHandle | null) {
  if (handle) {
    devToolsHandles.set(tabId, handle)
  } else if (!devToolsHandles.delete(tabId)) {
    return
  }

  invalidateStripTools()
}

/** Drop a closed tab's state so a long-lived window doesn't pin every preview it
 *  ever opened. */
export function forgetPreviewStripTools(tabId: string) {
  consoleStates.delete(tabId)
  registerPreviewDevTools(tabId, null)
}

/**
 * The console + DevTools glyphs for a preview, as strip-tool DATA. Only wired for
 * `url` previews — a file/artifact peek has no webview, hence no console and no
 * DevTools.
 */
export function previewStripTools(tabId: string): readonly PaneStripTool[] {
  const consoleState = previewConsoleState(tabId)
  const consoleOpen = consoleState.$open.get()
  const devTools = devToolsHandles.get(tabId)

  return [
    {
      active: consoleOpen,
      icon: <Codicon name="terminal" size="0.8125rem" />,
      id: 'preview-console',
      label: translateNow(consoleOpen ? 'preview.web.hideConsole' : 'preview.web.showConsole'),
      onSelect: () => consoleState.setOpen(open => !open)
    },
    {
      active: devTools?.open,
      disabled: !devTools,
      icon: <Codicon name="bug" size="0.8125rem" />,
      id: 'preview-devtools',
      label: translateNow(devTools?.open ? 'preview.web.hideDevTools' : 'preview.web.openDevTools'),
      onSelect: () => devTools?.toggle()
    }
  ]
}
