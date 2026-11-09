import { useStore } from '@nanostores/react'
import { useNavigate } from 'react-router'

import { tileOwnerRoute } from '@/app/chat/session-tile-owner'
import { useSimplifiedAgents } from '@/app/chat/sidebar/use-simplified-agents'
import { $workspaceIsPage, navigateToWorkspacePage, SETTINGS_ROUTE } from '@/app/routes'
import type { GroupNode } from '@/components/pane-shell/tree/model'
import { tabStripVisibleForZone } from '@/components/pane-shell/tree/renderer/strip-visibility'
import { isSessionStripPane } from '@/components/pane-shell/tree/store'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
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

import { usePortalRedesign } from './use-portal-redesign'

export function usePortalConversation(paneId: string) {
  const portal = usePortalRedesign()
  const page = useStore($workspaceIsPage)

  return portal && isSessionStripPane(paneId) && (paneId !== 'workspace' || !page)
}

/** Replace chat-only strips; a mixed tool/page group still needs its tab picker. */
export function usePortalZoneHeader(node: GroupNode, options: Omit<Parameters<typeof tabStripVisibleForZone>[0], 'mode'>) {
  const portal = usePortalConversation(options.active)

  return {
    stripVisible: tabStripVisibleForZone({ ...options, mode: node.tabStrip }) && !(portal && options.shown.every(isSessionStripPane)),
    conversationHeader: portal && !node.minimized
      ? <PortalConversationHeader paneId={options.active} title={options.paneFor(options.active)?.title} />
      : null
  }
}

/** Chrome stays outside the chat host, so mode/theme changes cannot discard its draft. */
export function PortalConversationHeader({ paneId, title }: { paneId: string; title?: string }) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const selected = useStore($selectedStoredSessionId)
  const tiles = useStore($sessionTiles)
  const profile = useStore($activeGatewayProfile)

  const rows = [
    ...useStore($sessions), ...useStore($cronSessions),
    ...useStore($messagingSessions), ...useStore($unlistedSessionOwnerRows)
  ]

  const sessionId = paneId === 'workspace' ? selected : paneId.slice('session-tile:'.length)
  const row = sessionId ? rows.find(session => sessionMatchesStoredId(session, sessionId)) : undefined
  const { agents, owner } = useSimplifiedAgents(row ? [row] : [])
  const route = sessionId ? tileOwnerRoute(tiles, rows, sessionId) ?? knownSessionOwner(rows, sessionId) : undefined
  // An unresolved existing session must never offer the ambient agent's settings.
  const ownerProfile = typeof route === 'string' ? route : route?.profile ?? (!sessionId ? profile : undefined)
  const connectionId = route && typeof route === 'object' ? route.connectionId : owner
  const agent = agents.find(item => item.connectionId === connectionId && item.profile === ownerProfile)

  function openSettings() {
    if (!ownerProfile) {return}
    const params = new URLSearchParams({ agent: ownerProfile })

    if (connectionId) {params.set('connection', connectionId)}
    navigateToWorkspacePage(navigate, `${SETTINGS_ROUTE}?${params}`)
  }

  return (
    <header
      className="@container/conversation flex min-h-14 shrink-0 items-center gap-4 border-b border-(--sidebar-edge-border) bg-(--ui-editor-surface-background) pl-8 pr-6"
      data-slot="portal-conversation-header"
    >
      <h1 className="min-w-0 flex-1 truncate text-base font-semibold" title={title}>
        {title || t.sidebar.nav['new-session']}
      </h1>
      <span className="hidden max-w-[40%] truncate text-xs text-(--ui-text-tertiary) @min-[36rem]/conversation:block">
        {agent ? `${agent.gateway} // ${agent.label}` : ownerProfile}
      </span>
      <Button aria-label={t.simplifiedAgents.settings} className="h-8 shrink-0 rounded-md border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) px-3 text-xs" disabled={!ownerProfile} onClick={openSettings} size="sm" variant="outline">
        <Codicon name="settings-gear" size={12} />
        <span className="hidden @min-[22rem]/conversation:inline">{t.simplifiedAgents.settings}</span>
      </Button>
    </header>
  )
}
