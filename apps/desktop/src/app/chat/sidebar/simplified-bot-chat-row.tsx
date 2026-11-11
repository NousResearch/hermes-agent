import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { FigmaAgentIcon } from '@/components/ui/figma-agent-icon'
import { RowButton } from '@/components/ui/row-button'
import { OverflowTip, Tip } from '@/components/ui/tooltip'
import { resolveProfileColor } from '@/lib/profile-color'
import { $openBotChat, $pendingBotOpen } from '@/plugins/hermes-bots/bot-state'
import { CANONICAL_CHAT_TITLE } from '@/plugins/hermes-bots/canonical-chat'
import { $lastRoster, botRosterKey } from '@/plugins/hermes-bots/data'
import { openRosterBot } from '@/plugins/hermes-bots/roster-actions'
import { botWorkspaceOwnerKey } from '@/plugins/hermes-bots/routing'
import type { RosterRow } from '@/plugins/hermes-bots/types'
import { recordIaAction } from '@/store/ia-metrics'
import { $profileColors } from '@/store/profile'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $sessionTiles } from '@/store/session-states'

import type { SimplifiedAgent } from './use-simplified-agents'

interface SimplifiedBotChatRowProps {
  agent: SimplifiedAgent
  appearance?: 'card' | 'hidden'
  summary?: ReactNode
  /** Overrides the profile colour where the row sits on a coloured selection fill. */
  avatarColor?: string
  onOpen?: () => void
}

export function SimplifiedBotChatRow({
  agent,
  appearance = 'card',
  summary,
  avatarColor,
  onOpen
}: SimplifiedBotChatRowProps) {
  const opened = useStore($openBotChat)
  const pending = useStore($pendingBotOpen)
  const focused = useStore($focusedStoredSessionId)
  const tiles = useStore($sessionTiles)
  const colors = useStore($profileColors)
  const roster = useStore($lastRoster)

  const bot: RosterRow = {
    name: agent.profile,
    connectionId: agent.connectionId ?? undefined,
    connectionKind: agent.connectionKind,
    connectionLabel: agent.gateway,
    targetProfile: agent.targetProfile || agent.profile,
    sourceScoped: Boolean(agent.connectionId)
  }

  const key = botRosterKey(bot)
  const description = roster.find(row => botRosterKey(row) === key)?.description
  const loading = pending?.key === key

  const tile = tiles.find(item => item.storedSessionId === focused)

  const selected =
    Boolean(focused) &&
    ((tile?.workspaceMode === 'bots' &&
      tile.workspaceOwnerKey === botWorkspaceOwnerKey(bot) &&
      tile.workspaceTabTitle === CANONICAL_CHAT_TITLE) ||
      (opened?.key === key && (opened.openedSessionId === focused || opened.openedRegistryId === focused)))

  return (
    <Tip label={appearance === 'hidden' ? `${agent.label} · ${agent.gateway}` : undefined}>
      <RowButton
        aria-busy={loading || undefined}
        aria-current={selected ? 'page' : undefined}
        className="portal-agent-chat"
        data-appearance={appearance}
        disabled={loading}
        onClick={() => {
          recordIaAction('ia.session.opened_agent')

          if (onOpen) {
            onOpen()
          } else {
            void openRosterBot(bot)
          }
        }}
      >
        {appearance !== 'hidden' &&
          (loading ? (
            <span className="portal-agent-avatar">
              <Codicon className="animate-spin" name="loading" />
            </span>
          ) : (
            <FigmaAgentIcon
              className="portal-agent-avatar"
              color={avatarColor ?? resolveProfileColor(agent.profile, colors)}
              variant="subtle"
            />
          ))}
        <span className="portal-agent-text">
          <OverflowTip label={appearance === 'hidden' ? undefined : agent.label}>
            <span className="portal-agent-name">{agent.label}</span>
          </OverflowTip>
          {(summary || (appearance === 'card' && description)) && (
            <span className="portal-agent-subtitle">{summary || description}</span>
          )}
        </span>
      </RowButton>
    </Tip>
  )
}
