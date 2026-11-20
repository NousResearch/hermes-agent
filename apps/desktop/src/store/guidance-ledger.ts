/**
 * What the redesign's discovery surfaces have already settled for this person: one mark per topic,
 * and the chats in which a task of theirs finished (the starter ideas leave after two).
 *
 * This module is the ONLY place that knows where those marks live. Today that is one machine-level
 * UI preference (`lib/ui-preferences/keys.ts`), so a splash seen in the desktop app or in one
 * browser is seen in every browser that opens this machine's Webapp; before a visitor has a machine
 * it is this browser, and the first contact with their new machine carries it over. The marks are
 * meant to move to the account; that move replaces this module's storage and nothing that calls it.
 */

import { computed } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'

/** Each topic's marks:
 *  - a splash is `seen` (shown and acknowledged) or `not_applicable` (settled without showing: a new
 *    install never sees the rename splash), so a later visit does not reconsider it;
 *  - the starter ideas are `hidden` (by the person) or `wanted` (brought back with Show ideas); with
 *    no mark they follow the finished-chat count;
 *  - the usage-stats offer's sidebar card is `deferred` (its × moved it to the tray) or `cleared`
 *    (cleared from the tray too); with no mark an unanswered offer shows as the card. Answering
 *    is the backend's, never a mark here;
 *  - the check-in ("How's Hermes going so far?") is `answered` or `dismissed` (then the tray offers
 *    it); with no mark it is due once two chats have had a task finish;
 *  - the developer-view offer is `accepted` or `declined`; with no mark it may be offered. */
const TOPIC_MARKS = {
  renameSplash: ['not_applicable', 'seen'],
  firstTimeSplash: ['not_applicable', 'seen'],
  starterIdeas: ['hidden', 'wanted'],
  usageStatsOffer: ['cleared', 'deferred'],
  checkIn: ['answered', 'dismissed'],
  developerViewOffer: ['accepted', 'declined']
} as const

export type GuidanceTopic = keyof typeof TOPIC_MARKS

export type GuidanceMarkOf<T extends GuidanceTopic> = (typeof TOPIC_MARKS)[T][number]

/** The chats a finished task is counted in. The ideas only need to know about two. */
export const FINISHED_TASK_CHATS_KEPT = 2

type GuidanceLedger = { [T in GuidanceTopic]?: GuidanceMarkOf<T> } & {
  /** Stored chat ids, oldest first, at most {@link FINISHED_TASK_CHATS_KEPT}. */
  finishedTaskChats?: string[]
}

export const GUIDANCE_LEDGER_KEY = 'hermes.desktop.guidance.v1'

const isTopic = (topic: string): topic is GuidanceTopic => Object.hasOwn(TOPIC_MARKS, topic)

function sanitize(value: unknown): GuidanceLedger {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const ledger: Record<string, unknown> = {}

  for (const [key, mark] of Object.entries(value)) {
    if (isTopic(key) && (TOPIC_MARKS[key] as readonly unknown[]).includes(mark)) {
      ledger[key] = mark
    }
  }

  const chats: unknown = (value as Record<string, unknown>).finishedTaskChats

  if (Array.isArray(chats)) {
    const ids = [...new Set(chats.filter((id): id is string => typeof id === 'string' && id.length > 0))]

    if (ids.length) {
      ledger.finishedTaskChats = ids.slice(0, FINISHED_TASK_CHATS_KEPT)
    }
  }

  return ledger as GuidanceLedger
}

export const $guidanceLedger = persistentAtom<GuidanceLedger>(GUIDANCE_LEDGER_KEY, {}, Codecs.json(sanitize))

export function guidanceMark<T extends GuidanceTopic>(topic: T): GuidanceMarkOf<T> | undefined {
  return $guidanceLedger.get()[topic] as GuidanceMarkOf<T> | undefined
}

/** Records a topic's mark. `seen` is final: nothing downgrades it back to `not_applicable`. */
export function markGuidance<T extends GuidanceTopic>(topic: T, mark: GuidanceMarkOf<T>): void {
  const ledger = $guidanceLedger.get()

  if (ledger[topic] === mark || ledger[topic] === 'seen') {
    return
  }

  $guidanceLedger.set({ ...ledger, [topic]: mark })
}

/** How many chats a task finished in, up to {@link FINISHED_TASK_CHATS_KEPT}. */
export const $finishedTaskChats = computed($guidanceLedger, ledger => ledger.finishedTaskChats?.length ?? 0)

/** Counts `chatId` as a chat in which a task finished; the same chat counts once. Returns whether the
 *  count moved. */
export function noteFinishedTaskChat(chatId: string): boolean {
  const ledger = $guidanceLedger.get()
  const chats = ledger.finishedTaskChats ?? []

  if (!chatId || chats.includes(chatId) || chats.length >= FINISHED_TASK_CHATS_KEPT) {
    return false
  }

  $guidanceLedger.set({ ...ledger, finishedTaskChats: [...chats, chatId] })

  return true
}
