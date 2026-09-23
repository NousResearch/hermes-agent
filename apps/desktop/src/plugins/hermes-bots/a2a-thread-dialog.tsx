/**
 * The bot-to-bot thread dialog: what two bots said to each other, in one
 * ordered view, plus the one line that answers "are they done?" — who still
 * owes a reply.
 *
 * Opened from a roster row whose preview is a DM (the receiver's side). Both
 * sides are read through `host.sessionMessages`, so a hidden Bot Chat on
 * another profile reads exactly like the one on screen — the user watches the
 * exchange instead of inferring it from two unrelated chats.
 */

import {
  Badge,
  Button,
  cn,
  coarseElapsed,
  Codicon,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  host,
  ScrollArea,
  useI18n
} from '@hermes/plugin-sdk'
import { useCallback, useEffect, useState } from 'react'

import { type A2aEvent, a2aEventsForSide, a2aThreadState, mergeA2aThread } from './a2a-thread'
import { botHandle } from './data'
import { useBots } from './i18n'
import { displayName } from './labels'
import { openRosterBot } from './roster-actions'
import { backendTargetProfile, resolveBotConnectionRoute } from './routing'
import { botCanonicalSessionId } from './row-helpers'
import type { RosterRow, SidebarRowLabels } from './types'

/** The sidebar's compact age form ("now", "52m", "3h", "18d") — same
 *  `coarseElapsed` + suffix pair the rows use, so the thread and the rail
 *  above it never disagree about how an age is spelled. */
function threadAge(ms: number, labels: SidebarRowLabels): string {
  const { unit, value } = coarseElapsed(Date.now() - ms)

  return unit === 'second' ? labels.ageNow : `${value}${unit === 'day' ? labels.ageDay : unit === 'hour' ? labels.ageHour : labels.ageMin}`
}

/** The transcript page each side contributes. A bot chat is long; the thread
 *  only needs the recent exchange, and the route caps at 500 anyway. */
const A2A_THREAD_FETCH_LIMIT = 200

type ThreadStatus = 'error' | 'loading' | 'ready' | 'unsupported'

interface A2aThreadDialogProps {
  /** The row the thread was opened from — the bot that RECEIVED the DM. */
  bot: RosterRow
  onClose: () => void
  open: boolean
  /** Handle of the other bot, as the row's preview named it. */
  peerHandle: string
  /** Full roster, so the peer's own row (and canonical chat) resolves. */
  roster: RosterRow[]
}

async function sideMessages(bot: RosterRow | null | undefined, limit: number) {
  if (!bot || typeof host.sessionMessages !== 'function') {
    return null
  }

  const route = resolveBotConnectionRoute(bot).route
  const sessionId = botCanonicalSessionId(bot)

  if (!sessionId) {
    return null
  }

  return host.sessionMessages(route, {
    limit,
    order: 'latest',
    profile: backendTargetProfile(route, String(bot.name || 'default')),
    sessionId
  })
}

export function A2aThreadDialog({ bot, onClose, open, peerHandle, roster }: A2aThreadDialogProps) {
  const { t } = useI18n()
  const b = useBots()
  const [events, setEvents] = useState<A2aEvent[]>([])
  const [status, setStatus] = useState<ThreadStatus>('loading')

  const peer = roster.find(row => botHandle(row.name, row) === peerHandle) || null

  const load = useCallback(async () => {
    if (typeof host.sessionMessages !== 'function') {
      setStatus('unsupported')

      return
    }

    setStatus('loading')

    try {
      const [mine, theirs] = await Promise.all([sideMessages(bot, A2A_THREAD_FETCH_LIMIT), sideMessages(peer, A2A_THREAD_FETCH_LIMIT)])

      setEvents(
        mergeA2aThread(
          a2aEventsForSide(mine?.messages || [], String(bot.name || ''), peerHandle),
          a2aEventsForSide(theirs?.messages || [], String(peer?.name || peerHandle), String(bot.name || ''))
        )
      )
      setStatus('ready')
    } catch {
      setStatus('error')
    }
  }, [bot, peer, peerHandle])

  useEffect(() => {
    if (open) {
      void load()
    }
  }, [open, load])

  const info = a2aThreadState(events)
  const age = info.lastTs ? threadAge(info.lastTs, t.sidebar.row) : ''

  return (
    <Dialog onOpenChange={next => (next ? undefined : onClose())} open={open}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{b.a2a.title}</DialogTitle>
          <DialogDescription>{b.a2a.description(displayName(bot), displayName(peer || { name: peerHandle }))}</DialogDescription>
        </DialogHeader>

        <div className="flex items-center gap-2">
          {status === 'ready' ? (
            <>
              <Badge variant={info.state === 'awaiting-reply' ? 'warn' : 'muted'}>
                {info.state === 'awaiting-reply'
                  ? b.a2a.awaiting(info.waitingOn || peerHandle)
                  : info.state === 'active'
                    ? b.a2a.active
                    : b.a2a.settled}
              </Badge>
              {age ? <span className="text-xs text-(--ui-text-tertiary)">{b.a2a.lastAt(age)}</span> : null}
            </>
          ) : null}
        </div>

        <ScrollArea className="max-h-[26rem] min-h-[8rem] pr-3">
          {status === 'loading' ? (
            <div className="flex items-center gap-2 py-6 text-sm text-(--ui-text-tertiary)">
              <Codicon className="animate-spin" name="loading" />
              {t.common.loading}
            </div>
          ) : status === 'error' ? (
            <p className="py-6 text-sm text-(--ui-text-tertiary)">{b.a2a.loadFailed}</p>
          ) : status === 'unsupported' ? (
            <p className="py-6 text-sm text-(--ui-text-tertiary)">{b.a2a.unsupported}</p>
          ) : events.length === 0 ? (
            <p className="py-6 text-sm text-(--ui-text-tertiary)">{b.a2a.empty}</p>
          ) : (
            <ol className="flex flex-col gap-2 py-1">
              {events.map((event, index) => (
                <li className="flex flex-col gap-0.5 rounded-md border border-(--ui-border) px-2.5 py-1.5" key={`${event.ts}-${index}`}>
                  <div className="flex items-center gap-1.5 text-xs text-(--ui-text-tertiary)">
                    <Codicon
                      className={cn('text-[0.75rem]', event.kind === 'reply' && 'text-(--ui-accent)')}
                      name={event.kind === 'dm' ? 'mail' : event.kind === 'reply' ? 'reply' : 'send'}
                    />
                    <span className="font-mono">{`@${event.from}`}</span>
                    <span>{event.kind === 'dm' ? b.a2a.received : event.kind === 'reply' ? b.a2a.replied : b.a2a.sent}</span>
                    <span className="ml-auto shrink-0">{threadAge(event.ts, t.sidebar.row)}</span>
                  </div>
                  <p className="whitespace-pre-wrap text-[0.8125rem]">{event.text}</p>
                </li>
              ))}
            </ol>
          )}
        </ScrollArea>

        <DialogFooter>
          {peer ? (
            <Button onClick={() => void openRosterBot(peer)} size="sm" variant="outline">
              {b.a2a.openChat(peerHandle)}
            </Button>
          ) : null}
          <Button onClick={() => void load()} size="sm" variant="ghost">
            {b.a2a.refresh}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
