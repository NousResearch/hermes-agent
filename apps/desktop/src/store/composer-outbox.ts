import { onListedSubmissions } from './admitted-submissions'
import { draftKey as scopeKey } from './composer'
import { type ComposerDraftRetry, draftRetryTextMatches } from './composer-draft-retry'
import { sendLanded } from './composer-outbox-landed'

// Enter clears the composer and hands the words to an async send chain (session resume or create,
// attachment sync, the prepared journal) before `prompt.submit` is written. Closing or reloading the
// window inside that gap ended the page with the words only in JS memory: nothing reached the
// backend, and the emptied draft is what persisted. Each send is journaled here synchronously at
// Enter and retired once the composer knows its outcome (accepted, or given back by the rejection
// path). An entry no live page holds is a send its page never finished: the next load of its draft
// scope restores it as the draft with its retry identity, so sending it again is the explicit retry
// under the same submission id and the owner never runs it twice. Restored, never re-sent on its
// own: like the prepared journal, this is not an automatic outbox. An entry whose send the owner
// already admitted (the window closed after the frame left but before the ack came back) is in the
// owner's pending snapshot or stored transcript under its submission id: it is dropped instead, so
// a send is never both shown and offered again.
//
// The saved draft can already hold the same words (main's draft persistence wrote them before the
// send cleared the composer). Then the entry hands that draft its identity instead of being offered
// later beside it: two copies of one send, each with its own id, would both run.
const PREFIX = 'hermes:composer-outbox:v1:'
// An entry older than this is no longer a send the user is waiting on: dropped, never offered. A week,
// so a send cut off before a weekend away still comes back.
const ENTRY_TTL_MS = 7 * 24 * 60 * 60_000

interface OutboxEntry {
  scope: string
  retry: ComposerDraftRetry
  /** When Enter journaled it (ms): bounds the transcript search for its stored row and its lifetime. */
  at: number
}

// Journal keys of sends this page has in flight: key -> release of the Web Lock that marks the
// entry as owned by a live page for every other window of the origin.
const ownedHere = new Map<string, () => void>()

const entryKey = (scope: string, id: string) => `${PREFIX}${JSON.stringify([scope, id])}`
const lockName = (key: string) => `${key}:owner`

const locks = () => (typeof navigator === 'undefined' ? undefined : navigator.locks)

function read(key: string): OutboxEntry | undefined {
  try {
    const entry = JSON.parse(window.localStorage.getItem(key) || 'null') as OutboxEntry | null

    return entry &&
      typeof entry.scope === 'string' &&
      typeof entry.retry?.text === 'string' &&
      typeof entry.at === 'number' &&
      Date.now() - entry.at < ENTRY_TTL_MS
      ? entry
      : undefined
  } catch {
    return undefined
  }
}

function write(key: string, entry: OutboxEntry | undefined): void {
  try {
    if (entry) {
      window.localStorage.setItem(key, JSON.stringify(entry))
    } else {
      window.localStorage.removeItem(key)
    }
  } catch {
    // Best-effort like the draft text itself (quota/private mode never blocks a send).
  }
}

function hold(key: string): void {
  ownedHere.set(key, () => undefined)
  void locks()?.request(lockName(key), () =>
    ownedHere.has(key) ? new Promise<void>(release => ownedHere.set(key, release)) : undefined
  )
}

export interface OutgoingSend {
  /** The send's draft scope changed (a fresh chat's create handoff assigned its session). */
  move: (scope: string | null) => void
  /** The composer knows the outcome: drop the entry and release its owner lock. */
  retire: () => void
}

/** Journal one send synchronously, before any await of the send chain can be cut by unload. */
export function journalOutgoingSend(scope: string | null, retry: ComposerDraftRetry): OutgoingSend {
  const at = Date.now()
  let key = entryKey(scopeKey(scope), retry.id)
  write(key, { at, scope: scopeKey(scope), retry })
  hold(key)

  const release = () => {
    write(key, undefined)
    const unlock = ownedHere.get(key)
    ownedHere.delete(key)
    unlock?.()
  }

  return {
    move: next => {
      if (!ownedHere.has(key) || entryKey(scopeKey(next), retry.id) === key) {
        return
      }

      release()
      key = entryKey(scopeKey(next), retry.id)
      write(key, { at, scope: scopeKey(next), retry })
      hold(key)
    },
    retire: release
  }
}

// A snapshot can list the admission only after the draft was offered (the owner committed the dead
// page's frame after this page's resume, or the snapshot came late). The owner's listing is the
// answer: `withdraw` takes the offered draft back if the composer still holds it untouched.
const WITHDRAW_WATCH_MS = 120_000

function withdrawOnceListed(entry: OutboxEntry, withdraw: (retry: ComposerDraftRetry) => void): void {
  let stop = () => {}
  const timer = setTimeout(() => stop(), WITHDRAW_WATCH_MS)

  stop = onListedSubmissions(entry.scope, ids => {
    if (ids.has(entry.retry.id)) {
      clearTimeout(timer)
      stop()
      withdraw(entry.retry)
    }
  })
}

/** Outbox keys no live page of this window holds, oldest first. Expired or unreadable entries are
 *  deleted on the way (`scope` given: only that scope's are returned). */
function orphanKeys(scope?: string): string[] {
  const keys: string[] = []
  const dead: string[] = []

  for (let index = 0; index < window.localStorage.length; index += 1) {
    const key = window.localStorage.key(index)

    if (!key?.startsWith(PREFIX) || ownedHere.has(key)) {
      continue
    }

    const entry = read(key)

    if (!entry) {
      dead.push(key)
    } else if (scope === undefined || entry.scope === scope) {
      keys.push(key)
    }
  }

  dead.forEach(key => write(key, undefined))

  return keys.sort()
}

/** Stored-session scopes with an interrupted send (the dead-session prune probes these). */
export function interruptedSendScopes(): string[] {
  return [...new Set(orphanKeys().flatMap(key => read(key)?.scope ?? []))]
}

/** The session is gone: its interrupted sends can never be offered again. */
export function dropInterruptedSends(scope: string): void {
  orphanKeys(scope).forEach(key => write(key, undefined))
}

export interface InterruptedSendHandlers {
  /** Put the send in the (empty) composer; false leaves the entry for a later load. */
  offer: (retry: ComposerDraftRetry) => boolean
  /** The owner already holds an adopted send: drop a draft that still is exactly those words. */
  landed: (retry: ComposerDraftRetry) => void
  /** The owner listed it after it was offered: take the offered draft back if untouched. */
  withdraw: (retry: ComposerDraftRetry) => void
}

// Offered or adopted: the composer now holds the send under its identity. Its landed check still
// runs, so a copy of a send the owner holds is taken back instead of inviting a second Enter.
async function settleTaken(entry: OutboxEntry, { landed, withdraw }: InterruptedSendHandlers): Promise<void> {
  if (await sendLanded(entry.scope, entry.retry.id, entry.at)) {
    landed(entry.retry)
  } else {
    withdrawOnceListed(entry, withdraw)
  }
}

/** The saved draft `text` (written at `savedAt`) already brought an interrupted send of `scope` back:
 *  it was persisted before the send cleared the composer. `adopt` hands that send's identity to the
 *  draft (false: the composer changed meanwhile, the entry stays for a later load) and the entry is
 *  retired so it is never offered beside it.
 *
 *  Only a draft saved before the send counts: equal words typed again later are a new message. And
 *  only under the entry's Web Lock, like `restoreInterruptedSend`: a held lock is a live window's send
 *  still in flight, never a copy this draft may take over. */
export async function adoptInterruptedSend(
  scope: string | null | undefined,
  text: string,
  savedAt: number | undefined,
  handlers: InterruptedSendHandlers & { adopt: (retry: ComposerDraftRetry) => boolean }
): Promise<void> {
  if (!text.trim() || savedAt === undefined) {
    return
  }

  const savedBefore = (entry: OutboxEntry | undefined): entry is OutboxEntry =>
    Boolean(entry && savedAt <= entry.at && draftRetryTextMatches(entry.retry, text))

  const take = (key: string) => {
    const entry = read(key)

    if (!ownedHere.has(key) && savedBefore(entry) && handlers.adopt(entry.retry)) {
      write(key, undefined)
      void settleTaken(entry, handlers)

      return true
    }

    return false
  }

  const available = locks()

  for (const key of orphanKeys(scopeKey(scope)).filter(key => savedBefore(read(key)))) {
    const taken = available
      ? await available.request(lockName(key), { ifAvailable: true }, lock => Boolean(lock && take(key)))
      : take(key)

    if (taken) {
      return
    }
  }
}

/** Offer `scope`'s interrupted sends (ones no live page owns) to an empty composer: an entry is
 *  removed when its send already landed in the stored transcript or when `offer` took it, so a
 *  composer the user already typed into leaves it for the next load. */
export async function restoreInterruptedSend(
  scope: string | null | undefined,
  handlers: InterruptedSendHandlers
): Promise<void> {
  const take = async (key: string) => {
    const entry = read(key)

    if (!entry || ownedHere.has(key)) {
      return
    }

    if (await sendLanded(entry.scope, entry.retry.id, entry.at)) {
      write(key, undefined)
    } else if (read(key) && handlers.offer(entry.retry)) {
      write(key, undefined)
      withdrawOnceListed(entry, handlers.withdraw)
    }
  }

  const available = locks()

  for (const key of orphanKeys(scopeKey(scope))) {
    if (!available) {
      await take(key)
    } else {
      // A held lock is another live window's send still in flight: never adopted from here. The
      // lock stays held across the transcript read, so two windows never both adopt one entry.
      await available.request(lockName(key), { ifAvailable: true }, lock => (lock ? take(key) : undefined))
    }
  }
}
