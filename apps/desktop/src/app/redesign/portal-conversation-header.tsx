import { useStore } from '@nanostores/react'
import { useLocation, useNavigate } from 'react-router'

import { tileOwnerRoute } from '@/app/chat/session-tile-owner'
import { useSimplifiedAgents } from '@/app/chat/sidebar/use-simplified-agents'
import { $workspaceIsPage, appViewForPath, isOverlayView, navigateToWorkspacePage, SETTINGS_ROUTE } from '@/app/routes'
import { tabStripVisibleForZone } from '@/components/pane-shell/tree/renderer/strip-visibility'
import type { ZoneHeaderHook } from '@/components/pane-shell/tree/renderer/zone-header-provider'
import { isSessionStripPane } from '@/components/pane-shell/tree/store'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { $primaryGatewayRoute } from '@/store/gateway-primary-route'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $cronSessions,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  $unlistedSessionOwnerRows,
  knownSessionOwner,
  sessionMatchesStoredId
} from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import { usePortalShell } from '@/store/shell-variant'

export function usePortalConversation(paneId: string) {
  const portal = usePortalShell()
  const page = useStore($workspaceIsPage)

  return portal && isSessionStripPane(paneId) && (paneId !== 'workspace' || !page)
}

/** Replace chat-only strips; a mixed tool/page group still needs its tab picker. Installed into
 *  every zone by the contrib controller (provideZoneHeader). */
export const usePortalZoneHeader: ZoneHeaderHook = (node, options) => {
  const portal = usePortalConversation(options.active)

  return {
    stripVisible:
      tabStripVisibleForZone({ ...options, mode: node.tabStrip }) &&
      !(portal && options.shown.every(isSessionStripPane)),
    conversationHeader: portal && !node.minimized ? <PortalConversationHeader paneId={options.active} /> : null
  }
}

/** Chrome stays outside the chat host, so mode/theme changes cannot discard its draft. */
export function PortalConversationHeader({ paneId }: { paneId: string }) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const selected = useStore($selectedStoredSessionId)
  const tiles = useStore($sessionTiles)
  const profile = useStore($activeGatewayProfile)
  const primary = useStore($primaryGatewayRoute)

  const rows = [
    ...useStore($sessions),
    ...useStore($cronSessions),
    ...useStore($messagingSessions),
    ...useStore($unlistedSessionOwnerRows)
  ]

  const sessionId = paneId === 'workspace' ? selected : paneId.slice('session-tile:'.length)
  const row = sessionId ? rows.find(session => sessionMatchesStoredId(session, sessionId)) : undefined
  const { agents, owner } = useSimplifiedAgents(row ? [row] : [])
  const route = sessionId ? (tileOwnerRoute(tiles, rows, sessionId) ?? knownSessionOwner(rows, sessionId)) : undefined
  // An unresolved existing session must never offer the ambient agent's settings.
  const ownerProfile = typeof route === 'string' ? route : (route?.profile ?? (!sessionId ? profile : undefined))

  const connectionId =
    route && typeof route === 'object' ? route.connectionId : sessionId ? primary.connectionId : owner

  const agent = agents.find(item => item.connectionId === connectionId && item.profile === ownerProfile)

  // Route overlays cover the chat but keep its host alive. Covered chrome must
  // leave the tab order without replacing the draft-owning body underneath.
  if (isOverlayView(appViewForPath(pathname))) {
    return null
  }

  function openSettings() {
    if (!ownerProfile || (sessionId && !connectionId)) {
      return
    }

    const params = new URLSearchParams({ agentProfile: ownerProfile })

    if (connectionId) {
      params.set('connection', connectionId)
    }

    navigateToWorkspacePage(navigate, `${SETTINGS_ROUTE}?${params}`)
  }

  // The session title lives in the titlebar pill. What remains overlays the
  // stage's top-right corner out of flow (stage.css), so the stage photo and the
  // transcript start at the panel's top edge; the row itself ignores the pointer
  // and only its controls take it.
  return (
    <header className="@container/conversation" data-slot="portal-conversation-header">
      <span className="hidden min-w-0 truncate text-xs tracking-[0.01em] text-(--ui-text-tertiary) @min-[30rem]/conversation:block">
        {agent ? `${agent.gateway} // ${agent.label}` : ownerProfile}
      </span>
      <Button
        aria-label={t.simplifiedAgents.settings}
        className="shrink-0"
        disabled={!ownerProfile || Boolean(sessionId && !connectionId)}
        onClick={openSettings}
        size="sm"
        variant="ghost"
      >
        <Codicon name="settings-gear" size="0.875rem" />
        <span className="hidden @min-[22rem]/conversation:inline">{t.simplifiedAgents.settings}</span>
      </Button>
    </header>
  )
}
