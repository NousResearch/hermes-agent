import { host, queryClient } from '@hermes/plugin-sdk'
import { QueryObserver } from '@tanstack/react-query'

import { $botMeta, rosterQueryOptions } from './data'
import { sortRosterBots } from './roster-pane-derivation'
import { publishRosterSnapshot } from './roster-pane-lifecycle'

type RosterSnapshot = Awaited<ReturnType<ReturnType<typeof rosterQueryOptions>['queryFn']>>

/**
 * The roster's publisher where the shell hosts no Bots pane (the redesign's Agents page): the
 * pane is otherwise the only thing that keeps the shared roster state live, and every other
 * reader — the redesign's agent rows and group chats, the chat-empty card, group membership,
 * hidden bots, Screen auto-raise — would see an empty roster. It observes the pane's own query
 * (same cache entry, same 5 s poll), re-keys with the active connection, refetches when the
 * socket opens, and publishes each new `profiles`/`sources` pair exactly as the pane does.
 * Returns the stop function.
 */
export function startRosterPublisher(): () => void {
  const observer = new QueryObserver(queryClient, rosterQueryOptions(host.state.connectionId.get()))
  let published: { profiles?: unknown; sources?: unknown } = {}

  const publish = (data: RosterSnapshot | undefined) => {
    const live = Array.isArray(data?.profiles) ? data.profiles : null

    if (!live || (published.profiles === live && published.sources === data?.sources)) {
      return
    }

    published = { profiles: live, sources: data?.sources }
    const allMeta = $botMeta.get()
    const { roster } = sortRosterBots(live, allMeta)

    publishRosterSnapshot({ data, roster, allMeta, activeSourceRoster: roster.filter(bot => !bot.remoteSource) })
  }

  const stopResults = observer.subscribe(result => publish(result.data))
  publish(observer.getCurrentResult().data)

  const stopConnection = host.state.connectionId.listen(connectionId =>
    observer.setOptions(rosterQueryOptions(connectionId))
  )

  const stopGateway = host.state.gateway.listen(state => {
    if (state === 'open') {
      void observer.refetch()
    }
  })

  return () => {
    stopResults()
    stopConnection()
    stopGateway()
  }
}
