import { useStore } from '@nanostores/react'
import { useNavigate } from 'react-router'

import { useSimplifiedAgents } from '@/app/chat/sidebar/use-simplified-agents'
import { useChatOwner } from '@/app/chat/use-chat-owner'
import { $workspaceIsPage, navigateToWorkspacePage, SETTINGS_ROUTE } from '@/app/routes'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { $selectedStoredSessionId } from '@/store/session'
import { usePortalShell } from '@/store/shell-variant'

/** The agent a conversation pane belongs to, and the way to its settings. An unresolved existing
 *  session never offers the ambient agent's settings: `open` is null until its owner is known. */
export function useConversationAgent(paneId: string) {
  const navigate = useNavigate()
  const selected = useStore($selectedStoredSessionId)
  const sessionId = paneId === 'workspace' ? selected : paneId.slice('session-tile:'.length)
  const chat = useChatOwner(sessionId)
  const { agents, owner } = useSimplifiedAgents(chat.row ? [chat.row] : [])
  const ownerProfile = chat.profile
  // A draft with no pinned machine belongs to the one in front.
  const connectionId = chat.connectionId ?? (sessionId ? undefined : owner)

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

  return { chat, label: agent ? `${agent.gateway} // ${agent.label}` : ownerProfile, open, sessionId }
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
