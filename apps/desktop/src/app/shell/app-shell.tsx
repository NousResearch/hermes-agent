import { useStore } from '@nanostores/react'
import type { CSSProperties, ReactNode } from 'react'
import { useSyncExternalStore } from 'react'

import { NotificationStack } from '@/components/notifications'
import { PaneShell } from '@/components/pane-shell'
import { SidebarProvider } from '@/components/ui/sidebar'
import {
  $fileBrowserOpen,
  $panesFlipped,
  $sidebarOpen,
  FILE_BROWSER_DEFAULT_WIDTH,
  FILE_BROWSER_PANE_ID,
  setSidebarOpen
} from '@/store/layout'
import { $paneWidthOverride } from '@/store/panes'
import { $connection } from '@/store/session'

import { StatusbarControls, type StatusbarItem } from './statusbar-controls'
import { TITLEBAR_HEIGHT, titlebarControlsPosition } from './titlebar'
import { TitlebarControls, type TitlebarTool } from './titlebar-controls'

interface AppShellProps {
  children: ReactNode
  leftStatusbarItems?: readonly StatusbarItem[]
  leftTitlebarTools?: readonly TitlebarTool[]
  onOpenSettings: () => void
  overlays?: ReactNode
  statusbarItems?: readonly StatusbarItem[]
  titlebarTools?: readonly TitlebarTool[]
}

// Renderer-side fallback so layout snaps even when the main-process fullscreen event
// hasn't landed yet (e.g. dev reloads, before the IPC bridge is wired).
function subscribeWindowSize(cb: () => void) {
  window.addEventListener('resize', cb)
  window.addEventListener('fullscreenchange', cb)

  return () => {
    window.removeEventListener('resize', cb)
    window.removeEventListener('fullscreenchange', cb)
  }
}

const viewportIsFullscreen = () =>
  window.innerWidth >= window.screen.width && window.innerHeight >= window.screen.height

export function AppShell({
  children,
  leftStatusbarItems,
  leftTitlebarTools,
  onOpenSettings,
  overlays,
  statusbarItems,
  titlebarTools
}: AppShellProps) {
  const sidebarOpen = useStore($sidebarOpen)
  const fileBrowserOpen = useStore($fileBrowserOpen)
  const panesFlipped = useStore($panesFlipped)
  const fileBrowserWidthOverride = useStore($paneWidthOverride(FILE_BROWSER_PANE_ID))
  const connection = useStore($connection)
  const viewportFullscreen = useSyncExternalStore(subscribeWindowSize, viewportIsFullscreen, () => false)
  const isFullscreen = Boolean(connection?.isFullscreen) || viewportFullscreen
  const titlebarControls = titlebarControlsPosition(connection?.windowButtonPosition, isFullscreen)
  // Width Windows/Linux reserve for the OS-painted min/max/close overlay (zero
  // on macOS, where window controls sit on the left and are reported via
  // windowButtonPosition instead). The right tool cluster has to clear them.
  const nativeOverlayWidth = connection?.nativeOverlayWidth ?? 0
  const titlebarToolsRight = nativeOverlayWidth > 0 ? `${nativeOverlayWidth}px` : '0.75rem'

  // The inset clears the top-left titlebar buttons when nothing covers the
  // window's left edge. Default layout: the sessions sidebar sits there.
  // Flipped layout: the file browser does instead.
  const leftEdgePaneOpen = panesFlipped ? fileBrowserOpen : sidebarOpen

  const titlebarContentInset = leftEdgePaneOpen
    ? 0
    : titlebarControls.left + TITLEBAR_HEIGHT + Math.round(TITLEBAR_HEIGHT / 2)

  // The static system cluster (haptics, profiles, settings, right-sidebar) is
  // hardcoded in TitlebarControls. Pane-supplied tools (preview's group) render
  // in a separate cluster anchored further left.
  //
  // Width math has to include the `gap-x-1` (0.25rem) between buttons:
  // N buttons + (N - 1) inner gaps, plus one extra 0.25rem of breathing room
  // between the pane-tool cluster and the system cluster so they don't sit
  // flush against each other. Modeled as N gaps (N - 1 inner + 1 trailing)
  // to keep the formula generic for any pane-tool count.
  const SYSTEM_TOOL_COUNT = 4
  const paneToolCount = titlebarTools?.filter(tool => !tool.hidden).length ?? 0
  const systemToolsWidth = `calc(${SYSTEM_TOOL_COUNT} * (var(--titlebar-control-size) + 0.25rem))`

  const fileBrowserWidth =
    fileBrowserWidthOverride !== undefined ? `${fileBrowserWidthOverride}px` : FILE_BROWSER_DEFAULT_WIDTH

  // Where the pane-tool cluster's right edge sits, measured from the inner
  // titlebar padding (--titlebar-tools-right). Two anchors:
  //   - file-browser closed → flush against static cluster's left edge
  //   - file-browser open   → flush against the file-browser pane's left edge
  //                           (= preview pane's right edge)
  const previewToolbarGap = fileBrowserOpen ? fileBrowserWidth : systemToolsWidth

  // Used by the drag region to know where the rightmost interactive element
  // ends. When pane tools are present, that's `gap + paneCount * controlSize
  // + paneCount * 0.25rem` (the leftmost button is at `tools-right + gap +
  // paneCount * (size + gap-x-1)`). Otherwise the static cluster's footprint
  // is enough.
  const titlebarToolsWidth =
    paneToolCount > 0
      ? `calc(${previewToolbarGap} + ${paneToolCount} * (var(--titlebar-control-size) + 0.25rem))`
      : systemToolsWidth

  return (
    <SidebarProvider
      className="h-screen min-h-0 flex-col bg-background"
      onOpenChange={setSidebarOpen}
      open={sidebarOpen}
      style={
        {
          // Alias for shadcn <Sidebar> descendants. Resolves to the chat-sidebar
          // pane track via PaneShell's emitted --pane-chat-sidebar-width.
          '--sidebar-width': 'var(--pane-chat-sidebar-width)',
          '--titlebar-height': `${TITLEBAR_HEIGHT}px`,
          '--titlebar-content-inset': `${titlebarContentInset}px`,
          '--titlebar-controls-left': `${titlebarControls.left}px`,
          '--titlebar-controls-top': `${titlebarControls.top}px`,
          '--titlebar-tools-right': titlebarToolsRight,
          '--titlebar-tools-width': titlebarToolsWidth,
          // Anchor for the pane-tool cluster's right edge in TitlebarControls.
          // Sourced from the layout store rather than the PaneShell-emitted
          // --pane-*-width vars because the titlebar is a sibling of PaneShell
          // and CSS variables resolve at the consumer's scope.
          '--shell-preview-toolbar-gap': previewToolbarGap
        } as CSSProperties
      }
    >
      <TitlebarControls leftTools={leftTitlebarTools} onOpenSettings={onOpenSettings} tools={titlebarTools} />

      <main className="relative z-3 flex min-h-0 w-full flex-1 flex-col overflow-hidden transition-none">
        <PaneShell className="min-h-0 flex-1">
          <div
            aria-hidden="true"
            className="pointer-events-none absolute left-0 top-0 z-1 h-(--titlebar-height) w-(--titlebar-controls-left) [-webkit-app-region:drag]"
          />
          <div
            aria-hidden="true"
            className="pointer-events-none absolute top-0 z-1 h-(--titlebar-height) left-[calc(var(--titlebar-controls-left)+(var(--titlebar-control-size)*2)+0.75rem)] right-[calc(var(--titlebar-tools-right)+var(--titlebar-tools-width)+0.75rem)] [-webkit-app-region:drag]"
          />

          {children}
        </PaneShell>

        <StatusbarControls items={statusbarItems} leftItems={leftStatusbarItems} />
      </main>

      {overlays}

      {/* Mounted at the shell root (after overlays) so success/error toasts
          surface above every route and overlay — not just the chat view. */}
      <NotificationStack />
    </SidebarProvider>
  )
}
