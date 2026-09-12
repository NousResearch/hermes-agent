/**
 * A bot row's expandable session list: the breakdown underneath each bot in the
 * roster. Data comes from `session.list` against the bot's own profile. Each
 * session link opens it via `host.openSession` scoped to the bot's workspace,
 * and a small new-chat affordance calls `newBotChat(bot)` to start a fresh thread.
 */

import {
  coarseElapsed,
  Codicon,
  DisclosureCaret,
  host,
  RowButton,
  useI18n,
  useQuery
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import { botRosterKey, newBotChat } from './data'
import { useBots } from './i18n'
import { backendTargetProfile, botWorkspaceOwnerKey, requestForBot } from './routing'
import type { RosterRow, SidebarRowLabels } from './types'

interface SessionRowSummary {
  id: string
  message_count: number
  preview?: string
  started_at?: number
  title?: string
}

/** Compact age for sidebar session row (\"now\", \"52m\", \"3h\", \"18d\") — same form
 *  the bot rows already use. */
function sessionRowAge(ms: number, r: SidebarRowLabels): string {
  const { unit, value } = coarseElapsed(Date.now() - ms)

  return unit === 'second' ? r.ageNow : `${value}${unit === 'day' ? r.ageDay : unit === 'hour' ? r.ageHour : r.ageMin}`
}

export interface BotSessionsListProps {
  bot: RosterRow
}

export function BotSessionsList({ bot }: BotSessionsListProps) {
  const { t } = useI18n()
  const bots = useBots()
  const [expanded, setExpanded] = useState(false)
  const rosterKey = botRosterKey(bot)
  const route = bot.route
  const profile = route?.profile ?? bot.name

  const { data, isLoading } = useQuery({
    enabled: expanded,
    queryKey: ['hermes-bots', 'sessions', rosterKey],
    queryFn: async () => {
      try {
        const targetProfile = backendTargetProfile(route, profile)

        const res = await requestForBot<{ sessions?: SessionRowSummary[] }>(bot, 'session.list', {
          profile: targetProfile,
          limit: 15
        })

        return (res?.sessions ?? []).filter(s => s.id && s.title !== 'Bot Chat')
      } catch {
        return []
      }
    },
    refetchInterval: expanded ? 5000 : false,
    staleTime: 5000
  })

  const sessions = data ?? []
  const ownerKey = botWorkspaceOwnerKey(bot)

  const openSession = (storedId: string) => {
    if (typeof host.openSession !== 'function') {
      return
    }

    void host.openSession(storedId, {
      ...(route ? { route } : {}),
      profile,
      intent: 'in-place',
      workspaceMode: 'bots',
      workspaceOwnerKey: ownerKey
    })
  }

  const handleNewChat = () => {
    newBotChat(bot)
  }

  const sidebarLabels: SidebarRowLabels = {
    ageNow: t.sidebar.row.ageNow,
    ageMin: t.sidebar.row.ageMin,
    ageHour: t.sidebar.row.ageHour,
    ageDay: t.sidebar.row.ageDay
  }

  return (
    <>
      <RowButton
        aria-expanded={expanded}
        className="flex w-full min-w-0 items-center gap-1.5 rounded-md px-2 py-1.5 text-left text-[0.6875rem] font-semibold uppercase tracking-wider text-(--ui-text-quaternary) transition-colors hover:bg-(--chrome-action-hover) hover:text-(--ui-text-secondary)"
        onClick={() => setExpanded(prev => !prev)}
      >
        <DisclosureCaret open={expanded} />
        <span className="min-w-0 flex-1 truncate">{bots.bot.sessionsHeading}</span>
        {sessions.length > 0 ? (
          <span className="shrink-0 font-normal tabular-nums text-(--ui-text-quaternary)">{sessions.length}</span>
        ) : null}
      </RowButton>
      {expanded ? (
        <div className="space-y-0.5">
          {isLoading ? (
            <div className="flex min-w-0 items-center gap-1.5 px-3 py-1.5 text-xs text-(--ui-text-quaternary)">
              <span>{bots.bot.sessionsLoading}</span>
            </div>
          ) : sessions.length === 0 ? (
            <div className="flex min-w-0 items-center gap-1.5 px-3 py-1.5 text-xs text-(--ui-text-quaternary)">
              <span>{bots.bot.sessionsEmpty}</span>
            </div>
          ) : (
            sessions.map(session => {
              const age = session.started_at ? sessionRowAge(session.started_at * 1000, sidebarLabels) : null
              const title = session.title || bots.bot.sessionUntitled

              return (
                <RowButton
                  className="flex w-full min-w-0 items-center gap-2 rounded-md px-3 py-1.5 text-left text-xs transition-colors hover:bg-(--chrome-action-hover)"
                  key={session.id}
                  onClick={() => openSession(session.id)}
                >
                  <div className="min-w-0 flex-1 truncate">{title}</div>
                  {age ? <span className="shrink-0 text-(--ui-text-quaternary)">{age}</span> : null}
                </RowButton>
              )
            })
          )}
          <RowButton
            className="flex w-full min-w-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-left text-xs transition-colors hover:bg-(--chrome-action-hover)"
            onClick={handleNewChat}
          >
            <Codicon className="shrink-0 text-(--ui-text-quaternary)" name="add" />
            <span>{bots.bot.newSession}</span>
          </RowButton>
        </div>
      ) : null}
    </>
  )
}
