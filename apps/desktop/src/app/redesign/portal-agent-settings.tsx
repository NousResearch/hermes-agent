import { useStore } from '@nanostores/react'
import { useNavigate } from 'react-router'

import { tileOwnerRoute } from '@/app/chat/session-tile-owner'
import { useSimplifiedAgents } from '@/app/chat/sidebar/use-simplified-agents'
import { $workspaceIsPage, navigateToWorkspacePage, SETTINGS_ROUTE } from '@/app/routes'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { $primaryGatewayRoute } from '@/store/gateway-primary-route'
import { $activeGatewayProfile, $newChatRoute } from '@/store/profile'
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

/** The agent a conversation pane belongs to, and the way to its settings. An unresolved existing
 *  session never offers the ambient agent's settings: `open` is null until its owner is known. */
export function useConversationAgent(paneId: string) {
  const navigate = useNavigate()
  const selected = useStore($selectedStoredSessionId)
  const tiles = useStore($sessionTiles)
  const profile = useStore($activeGatewayProfile)
  const primary = useStore($primaryGatewayRoute)
  // A draft belongs to the agent pinned for it (session.create reads the same route), which the
  // ambient profile only follows once that agent's backend is in front.
  const draft = useStore($newChatRoute)

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

  const ownerProfile =
    typeof route === 'string' ? route : (route?.profile ?? (!sessionId ? (draft?.profile ?? profile) : undefined))

  const connectionId =
    route && typeof route === 'object'
      ? route.connectionId
      : sessionId
        ? primary.connectionId
        : (draft?.connectionId ?? owner)

  const agent = agents.find(item => item.connectionId === connectionId && item.profile === ownerProfile)

  const open =
    !ownerProfile || (sessionId && !connectionId)
      ? null
      : () => {
          const params = new URLSearchParams({ agentProfile: ownerProfile })

          if (connectionId) {
            params.set('connection', connectionId)
          }

          navigateToWorkspacePage(navigate, `${SETTINGS_ROUTE}?${params}`)
        }

  return { label: agent ? `${agent.gateway} // ${agent.label}` : ownerProfile, open }
}

/** A conversation's Agent settings as a 30px chrome button whose tooltip names it: the band's
 *  sliders at its trailing end for the main chat (settings-2 16292:15304), and the same button in an
 *  extra tile's own header. */
export function PortalAgentSettingsButton({ paneId }: { paneId: string }) {
  const { t } = useI18n()
  const { open } = useConversationAgent(paneId)

  return (
    <Tip label={t.simplifiedAgents.settings} placement="toolbar">
      <Button
        aria-label={t.simplifiedAgents.settings}
        className="portal-chrome-button"
        data-slot="portal-agent-settings"
        disabled={!open}
        onClick={() => open?.()}
        onPointerDown={event => event.stopPropagation()}
        size="icon-titlebar"
        type="button"
        variant="ghost"
      >
        <Codicon name="settings" />
      </Button>
    </Tip>
  )
}

/** The band's Agent settings: the main chat's, wherever a conversation holds the stage. */
export function PortalBandAgentSettings() {
  const portal = usePortalShell()
  const page = useStore($workspaceIsPage)

  return portal && !page ? <PortalAgentSettingsButton paneId="workspace" /> : null
}
