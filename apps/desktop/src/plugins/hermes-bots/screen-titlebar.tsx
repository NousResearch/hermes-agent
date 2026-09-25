/**
 * Screen titlebar entry — a titlebar button that opens the Bot Screen pane of
 * the chat the user is looking at, from any chat. Until now the screen was
 * reachable only from Bot Mode surfaces (the sidebar portal, the routines
 * hero, the row context menu); this makes it a window-level affordance.
 *
 * Everything heavy already exists: BotScreenPane owns the display.* RPCs, the
 * sibling /api/display/ws RFB bridge and noVNC from the app's own dependency.
 * This file adds only the entry point. The current chat's owner
 * ($focusedBotOwner, with the Bot Mode fallback) names the bot; its roster row
 * is used when the roster knows it, else synthesized the way
 * ProfileGroupScreenPortal does; with no focused bot chat at all the pane
 * falls back to this window's active profile. The chat you are in IS the
 * picker: the open pane follows the chat you switch to, because the pane's
 * render closure subscribes to the live owner atom — same behavior the pane
 * picker had following the active profile.
 *
 * The toggle follows the app's workspace pattern (host.openWorkspace disposer
 * + host.paneVisibility, feature-detected for older shells) exactly like
 * screen-open.tsx and group-chat-view.tsx, docked right of the workspace
 * rather than taking over the main area.
 */

import { atom, Button, Codicon, host, Tip, TITLEBAR_AREAS, translateNow, useValue } from '@hermes/plugin-sdk'
import type { PluginContext } from '@hermes/plugin-sdk'
import { useMemo } from 'react'
import type { ReactNode } from 'react'

import { $focusedBotOwner, focusedRosterOwner } from './bot-state'
import { $lastRoster } from './data'
import { useBots } from './i18n'
import { resolveBotConnectionRoute } from './routing'
import { BotScreenPane } from './screen-pane'
import { ID } from './shared'
import type { RosterRow } from './types'

/** Workspace id (`host.openWorkspace` prefixes `plugin-workspace:`). */
const WORKSPACE_KEY = `${ID}:screen-global`
const WORKSPACE_PANE_ID = `plugin-workspace:${WORKSPACE_KEY}`

let paneClose: null | (() => void) = null
let paneRender: null | (() => ReactNode) = null

/** Older shells without host.paneVisibility: the open state lives here. */
const $openFallback = atom(false)

/** The visible-pane atom, feature-detected like group-chat-view.tsx does. */
function screenPanelVisibleAtom() {
  if (typeof host.paneVisibility !== 'function') {
    return $openFallback
  }

  try {
    return host.paneVisibility(WORKSPACE_PANE_ID)
  } catch {
    return $openFallback
  }
}

function openScreenPanel(): void {
  if (paneClose || !paneRender) {
    return
  }

  if (typeof host.openWorkspace !== 'function') {
    // Older shells (same gate screen-open.tsx uses).
    host.notify({ kind: 'info', message: translateNow('screen.openNeedsUpdate') })

    return
  }

  try {
    if (typeof host.undismissPane === 'function') {
      host.undismissPane(WORKSPACE_PANE_ID)
    }

    paneClose = host.openWorkspace(WORKSPACE_KEY, {
      // Docked beside the workspace like the other plugin panes — not a
      // main-area tab that takes over the chat it was opened from.
      dock: { pane: 'workspace', pos: 'right' },
      minWidth: '360px',
      onClose: () => {
        paneClose = null
        $openFallback.set(false)
      },
      render: () => paneRender!(),
      title: translateNow('screen.title')
    })

    $openFallback.set(true)
  } catch {
    paneClose = null
    $openFallback.set(false)
    host.notify({ kind: 'error', message: translateNow('screen.openFailed') })
  }
}

function collapseScreenPanel(): void {
  const close = paneClose

  paneClose = null
  $openFallback.set(false)

  if (close) {
    try {
      close()
    } catch {
      /* the pane is gone either way; the button must not stick open */
    }
  }
}

export function toggleScreenPanel(): void {
  if (Boolean(paneClose) || Boolean(screenPanelVisibleAtom()?.get?.())) {
    collapseScreenPanel()
  } else {
    openScreenPanel()
  }
}

/**
 * The chat the user is LOOKING AT names the bot: its roster row when the
 * roster knows it, else a stable synthesized row (the ProfileGroupScreenPortal
 * shape) so a profile the current roster filter does not list stays routable.
 * Memoized on identity — BotScreenPane's effects key on `bot`, so a rebuilt
 * row would re-subscribe the lease listener on every titlebar paint.
 */
function useScreenBot(): RosterRow | null {
  const owner = focusedRosterOwner(useValue($focusedBotOwner))
  const roster = useValue($lastRoster)
  const name = owner?.name ?? null
  // The fallback owner carries '' when no connection is known — treat it as
  // "unspecified" so a roster row without one still matches by name.
  const connectionId = owner?.connectionId || null

  return useMemo(() => {
    if (!name) {
      return null
    }

    return (
      roster.find(row => {
        const resolved = resolveBotConnectionRoute(row)

        return resolved.route
          ? resolved.route.profile === name && resolved.route.connectionId === (connectionId ?? 'local')
          : row.name === name && connectionId === null
      }) ??
      (connectionId
        ? ({
            name,
            sourceScoped: true,
            connectionId,
            connectionKind: connectionId === 'local' ? 'local' : 'remote'
          } as RosterRow)
        : ({ name } as RosterRow))
    )
  }, [connectionId, name, roster])
}

/** The pane body: the focused chat's bot, else this window's active profile. */
function GlobalScreenPanel() {
  const bot = useScreenBot()
  const activeProfile = useValue(host.state.profile)

  // No focused bot chat (a plain session, a non-Bot-Mode window): the
  // window's own profile is still a screen this pane can show — the same
  // default the pane picker shipped with.
  const owner = bot ?? (activeProfile ? ({ name: activeProfile } as RosterRow) : null)

  if (!owner) {
    return null
  }

  return <BotScreenPane bot={owner} />
}

/** Native titlebar tool shape: 24x24 `icon-titlebar` button, ghost, no label. */
export function ScreenTitlebarButton() {
  const t = useBots()
  const open = useValue(screenPanelVisibleAtom())
  const label = open ? t.screen.collapsePanel : t.screen.openPanel

  return (
    <Tip label={label} placement="toolbar">
      <Button
        aria-label={label}
        className="bg-transparent select-none text-muted-foreground/85 hover:bg-(--ui-control-hover-background) hover:text-foreground"
        onClick={toggleScreenPanel}
        onPointerDown={event => event.stopPropagation()}
        size="icon-titlebar"
        type="button"
        variant="ghost"
      >
        <Codicon className="leading-none" name="device-desktop" size={13.9} />
      </Button>
    </Tip>
  )
}

/** Register the titlebar contribution; the disposer also closes the pane. */
export function registerScreenTitlebar(ctx: PluginContext): () => void {
  paneRender = () => <GlobalScreenPanel />

  const unregister = ctx.register({
    id: 'screen',
    area: TITLEBAR_AREAS.right,
    order: 21,
    render: () => <ScreenTitlebarButton />
  })

  return () => {
    unregister()
    collapseScreenPanel()
  }
}

/** Test seam: the toggle's open state is module-level (the pane outlives the
 *  button's mounts), so tests clear it between cases — the same shape
 *  screen-autoraise's resetScreenAutoRaise serves. */
export function resetScreenTitlebar(): void {
  paneClose = null
  paneRender = null
  $openFallback.set(false)
}
