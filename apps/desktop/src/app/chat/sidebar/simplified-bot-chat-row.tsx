import { useStore } from '@nanostores/react'

import { Codicon } from '@/components/ui/codicon'
import { RowButton } from '@/components/ui/row-button'
import { cn } from '@/lib/utils'
import { $openBotChat, $pendingBotOpen } from '@/plugins/hermes-bots/bot-state'
import { CANONICAL_CHAT_TITLE } from '@/plugins/hermes-bots/canonical-chat'
import { botRosterKey } from '@/plugins/hermes-bots/data'
import { openRosterBot } from '@/plugins/hermes-bots/roster-actions'
import { botWorkspaceOwnerKey } from '@/plugins/hermes-bots/routing'
import type { RosterRow } from '@/plugins/hermes-bots/types'
import { recordIaAction } from '@/store/ia-metrics'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $sessionTiles } from '@/store/session-states'

import type { SimplifiedAgent } from './use-simplified-agents'

export function SimplifiedBotChatRow({ agent }: { agent: SimplifiedAgent }) {
  const opened = useStore($openBotChat)
  const pending = useStore($pendingBotOpen)
  const focused = useStore($focusedStoredSessionId)
  const tiles = useStore($sessionTiles)

  const bot: RosterRow = {
    name: agent.profile,
    connectionId: agent.connectionId ?? undefined,
    connectionKind: agent.connectionKind,
    connectionLabel: agent.gateway,
    targetProfile: agent.targetProfile || agent.profile,
    sourceScoped: Boolean(agent.connectionId)
  }

  const key = botRosterKey(bot)
  const loading = pending?.key === key

  const tile = tiles.find(item => item.storedSessionId === focused)

  const selected =
    Boolean(focused) &&
    ((tile?.workspaceMode === 'bots' &&
      tile.workspaceOwnerKey === botWorkspaceOwnerKey(bot) &&
      tile.workspaceTabTitle === CANONICAL_CHAT_TITLE) ||
      (opened?.key === key && (opened.openedSessionId === focused || opened.openedRegistryId === focused)))

  return (
    <RowButton
      aria-busy={loading || undefined}
      aria-current={selected ? 'page' : undefined}
      className={cn(
        'flex min-h-8.5 w-full min-w-0 items-center gap-2 rounded-sm px-2 text-left text-xs hover:bg-(--chrome-action-hover) focus-visible:outline-2 focus-visible:outline-ring',
        selected && 'bg-(--chrome-action-hover)'
      )}
      disabled={loading}
      onClick={() => {
        recordIaAction('ia.session.opened_agent')
        void openRosterBot(bot)
      }}
    >
      <span className="grid size-4 shrink-0 place-items-center">
        <Codicon className={loading ? 'animate-spin' : undefined} name={loading ? 'loading' : 'robot'} />
      </span>
      <span className="min-w-0 truncate">{CANONICAL_CHAT_TITLE}</span>
    </RowButton>
  )
}
