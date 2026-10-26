/**
 * "Waiting on teammates" — a compact card above a bot's Bot Chat composer listing
 * every `message_agent` delivery from THIS bot that has not settled yet: who it
 * went to, where it stands (queued / picked up / working / waiting on you /
 * failed with its typed reason), and for how long. Open fronts the teammate's
 * chat (where its question or approval card is); Dismiss hides a failure.
 *
 * Read-only by construction: the rows come from the gateway's
 * `bot_delivery.list` (the target profiles' delivery receipts), polled on the
 * roster's 5 s cadence. Nothing here can resend — an unknown outcome is shown as
 * unknown, with the rule spelled out, never retried.
 */

import { atom, coarseElapsed, Codicon, host, Tip, useComposerSession, useQuery, useValue } from '@hermes/plugin-sdk'

import { avatarColor, botAppearance, BotFace } from './avatar'
import { $botMeta, botRosterKey, useRoster } from './data'
import { useBots } from './i18n'
import { displayName } from './labels'
import { openRosterBot } from './roster-actions'
import { botRosterMeta, requestForBot } from './routing'
import type { TeammateWaitMessages } from './teammate-wait-locales'
import type { RosterRow } from './types'

export type DeliveryState = 'claimed' | 'failed' | 'queued' | 'waiting_on_you' | 'working'

/** One row of `bot_delivery.list` (contract: `tui_gateway/contracts/groups_bot_relay.py`). */
export interface TeammateDelivery {
  delivery_id: string
  target: string
  target_handle: string
  state: DeliveryState
  since: number
  preview: string
  waiting_on?: null | string
  reason?: null | string
  error?: null | string
}

const DISMISSED_KEY = 'hermes.plugin.hermes-bots.teammate-waits-dismissed'

const readDismissed = (): Record<string, true> => {
  try {
    return JSON.parse(globalThis.localStorage?.getItem(DISMISSED_KEY) || '{}') || {}
  } catch {
    return {}
  }
}

/** Failure ids the user dismissed. Persisted so a restart does not bring them back for the
 *  rest of the gateway's 24 h failure window. */
export const $dismissedDeliveries = atom<Record<string, true>>(readDismissed())

export function dismissDelivery(id: string) {
  const next = { ...$dismissedDeliveries.get(), [id]: true as const }
  $dismissedDeliveries.set(next)
  globalThis.localStorage?.setItem(DISMISSED_KEY, JSON.stringify(next))
}

/** The bot whose canonical Bot Chat this composer is for, or null for any other chat. */
export function senderForSession(roster: RosterRow[], ids: Array<null | string | undefined>): null | RosterRow {
  const wanted = new Set(ids.filter(Boolean))

  return (
    roster.find(bot => {
      const chat = bot.canonical_session

      return !bot.ghost && Boolean(chat) && (wanted.has(chat?.id) || wanted.has(chat?.resolved_id))
    }) ?? null
  )
}

/** Rows the strip shows: dismissed failures drop; only failures can be dismissed. */
export function visibleDeliveries(rows: TeammateDelivery[], dismissed: Record<string, true>): TeammateDelivery[] {
  return rows.filter(row => row.state !== 'failed' || !dismissed[row.delivery_id])
}

/** Plain words for a failed row's typed reason. A code this build has no words for (or the owner's
 *  `unknown`) shows the owner's own error line instead, never the raw code. */
export function failureReasonText(
  reason: null | string | undefined,
  error: null | string | undefined,
  reasons: TeammateWaitMessages['reasons']
) {
  if (reason && reason !== 'unknown' && Object.hasOwn(reasons, reason)) {
    return reasons[reason]
  }

  return error?.trim() || reasons.unknown
}

function teammateFor(roster: RosterRow[], sender: RosterRow, row: TeammateDelivery): RosterRow {
  const sameSource = (bot: RosterRow) => (bot.connectionId || '') === (sender.connectionId || '')

  return (
    roster.find(bot => sameSource(bot) && bot.name === row.target) ??
    ({ ...sender, canonical_session: null, name: row.target, title: null } as RosterRow)
  )
}

const compactAge = (since: number, now: number) => {
  const { unit, value } = coarseElapsed(now - since * 1000)

  return `${value}${String(unit).charAt(0)}`
}

async function listDeliveries(sender: RosterRow): Promise<TeammateDelivery[]> {
  try {
    const res = await requestForBot<{ deliveries?: TeammateDelivery[] }>(sender, 'bot_delivery.list', {})

    return Array.isArray(res?.deliveries) ? res.deliveries : []
  } catch {
    // An older gateway has no `bot_delivery.list`: no strip, never an error toast per poll.
    return []
  }
}

interface DeliveryRowProps {
  /** When the rows were last fetched: an unchanged poll result still re-renders, so ages keep counting. */
  now: number
  roster: RosterRow[]
  row: TeammateDelivery
  sender: RosterRow
}

function DeliveryRow({ now, row, sender, roster }: DeliveryRowProps) {
  const b = useBots()
  const meta = useValue($botMeta)
  const teammate = teammateFor(roster, sender, row)
  const look = botAppearance(teammate.name, botRosterMeta(teammate, meta))
  const unknown = row.state === 'failed' && row.reason === 'unknown_outcome'

  const label =
    row.state === 'waiting_on_you'
      ? row.waiting_on === 'approval'
        ? b.waiting.waitingApproval
        : b.waiting.waitingClarify
      : row.state === 'failed'
        ? unknown
          ? b.waiting.unknownOutcome
          : b.waiting.failed(failureReasonText(row.reason, row.error, b.waiting.reasons))
        : b.waiting[row.state]

  return (
    <li className="flex min-w-0 items-center gap-2 py-0.5" data-delivery-state={row.state}>
      <BotFace
        color={avatarColor(look.color, teammate.name)}
        image={look.image ?? null}
        name={teammate.name}
        shape={look.shape}
        size={18}
      />
      <span className="max-w-[9rem] shrink-0 truncate font-medium text-(--ui-text-secondary)">
        {displayName(teammate, botRosterMeta(teammate, meta))}
      </span>
      <Tip label={row.error || row.preview || label}>
        <span
          className={
            row.state === 'waiting_on_you'
              ? 'min-w-0 truncate font-medium text-(--ui-accent)'
              : row.state === 'failed'
                ? 'min-w-0 truncate text-(--ui-danger)'
                : 'min-w-0 truncate text-(--ui-text-tertiary)'
          }
        >
          {label}
        </span>
      </Tip>
      <span className="ml-auto shrink-0 tabular-nums text-(--ui-text-quaternary)">{compactAge(row.since, now)}</span>
      <button
        className="shrink-0 rounded px-1.5 text-(--ui-text-secondary) hover:bg-(--ui-surface-hover)"
        onClick={() => void openRosterBot(teammate)}
        type="button"
      >
        {b.waiting.open}
      </button>
      {row.state === 'failed' ? (
        <Tip label={b.waiting.dismiss}>
          <button
            aria-label={b.waiting.dismiss}
            className="shrink-0 rounded px-1 text-(--ui-text-quaternary) hover:bg-(--ui-surface-hover)"
            onClick={() => dismissDelivery(row.delivery_id)}
            type="button"
          >
            <Codicon name="close" />
          </button>
        </Tip>
      ) : null}
    </li>
  )
}

export function TeammateWaits() {
  const b = useBots()
  const { sessionId, storedSessionId } = useComposerSession()
  const roster = useRoster().data?.profiles ?? []
  const sender = senderForSession(roster, [storedSessionId, sessionId])
  const dismissed = useValue($dismissedDeliveries)

  const deliveries = useQuery({
    queryKey: ['hermes-bots', 'teammate-waits', sender ? botRosterKey(sender) : ''],
    queryFn: () => (sender ? listDeliveries(sender) : Promise.resolve([])),
    enabled: Boolean(sender) && typeof host.requestProfile === 'function',
    refetchInterval: 5000,
    staleTime: 5000
  })

  const rows = visibleDeliveries(deliveries.data ?? [], dismissed)

  if (!sender || !rows.length) {
    return null
  }

  const anyUnknown = rows.some(row => row.reason === 'unknown_outcome')

  return (
    <section
      aria-label={b.waiting.title}
      className="rounded-md border border-(--ui-border) bg-(--ui-surface-raised) px-2 py-1 text-xs"
      data-slot="teammate-waits"
    >
      <Tip label={b.waiting.neverResent}>
        <div className="flex items-center gap-1 font-medium text-(--ui-text-secondary)">
          <Codicon name="comment-discussion" />
          {b.waiting.title}
        </div>
      </Tip>
      <ul className="m-0 list-none p-0">
        {rows.map(row => (
          <DeliveryRow key={row.delivery_id} now={deliveries.dataUpdatedAt} roster={roster} row={row} sender={sender} />
        ))}
      </ul>
      {anyUnknown ? <p className="m-0 pt-0.5 text-(--ui-text-tertiary)">{b.waiting.neverResent}</p> : null}
    </section>
  )
}
