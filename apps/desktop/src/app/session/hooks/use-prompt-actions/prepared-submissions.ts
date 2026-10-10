import type { ComposerAttachment } from '@/store/composer'
import type { SessionOwnerScope } from '@/store/session-request-router'

import type { SubmissionDestination } from './submission-destination'
import type { SubmitTextOptions } from './utils'

const STORAGE_KEY = 'hermes.desktop.preparedSubmissions.v1'

export interface PreparedSubmission {
  id: string
  owner: SessionOwnerScope
  attachments: ComposerAttachment[]
  text: string
  displayText?: string
  params: Record<string, unknown>
  legacyAttempted?: boolean
}

// Admitted entries whose durable removal failed (ENOSPC/EIO): journal key -> spent submission id.
// Their identity is spent: no window adopts, lists or slots them again, so a later send is never
// deduplicated into an earlier admission. The tombstone is written BEFORE the journal removal to a
// separate store and read on every journal access, so a reload (fresh module memory) still honors
// it; module memory only covers a tombstone write that itself failed. A tombstone hides only the
// entry carrying that exact id, never a newer send that reused the journal key.
// One localStorage key per tombstone: every window of the origin shares this storage with
// last-writer-wins per KEY, so a single blob let one window's read-modify-write drop the tombstone
// another window had just written for a different entry (and a reload then re-adopted that
// admitted input). Each retirement now writes only the record it owns.
const SPENT_PREFIX = 'hermes.desktop.preparedSubmissions.spent.v2:'
// The pre-v2 single blob: still honored, drained entry by entry as each tombstone clears.
const LEGACY_SPENT_KEY = 'hermes.desktop.preparedSubmissions.spent.v1'
const retired = new Map<string, string>()

function readLegacySpent(): Record<string, string> {
  const raw = window.localStorage.getItem(LEGACY_SPENT_KEY)
  const parsed: unknown = raw ? JSON.parse(raw) : {}

  return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Record<string, string> : {}
}

function readSpent(): Record<string, string> {
  const spent = readLegacySpent()

  for (let index = 0; index < window.localStorage.length; index += 1) {
    const key = window.localStorage.key(index)
    const id = key?.startsWith(SPENT_PREFIX) ? window.localStorage.getItem(key) : null

    if (key && id) { spent[key.slice(SPENT_PREFIX.length)] = id }
  }

  return { ...spent, ...Object.fromEntries(retired) }
}

function writeSpent(key: string, id: string | undefined): void {
  if (id !== undefined) {
    window.localStorage.setItem(SPENT_PREFIX + key, id)

    return
  }

  window.localStorage.removeItem(SPENT_PREFIX + key)
  const legacy = readLegacySpent()

  if (!(key in legacy)) { return }
  delete legacy[key]

  if (Object.keys(legacy).length) { window.localStorage.setItem(LEGACY_SPENT_KEY, JSON.stringify(legacy)) }
  else { window.localStorage.removeItem(LEGACY_SPENT_KEY) }
}

// A journal, not an automatic outbox. Only an explicit retry may reuse an
// uncertain admission. Read storage each time so a remount cannot lose it.
async function readJournal(): Promise<Record<string, PreparedSubmission>> {
  const native = window.hermesDesktop?.preparedSubmissions

  const parsed: unknown = JSON.parse(native
    ? await native.read()
    : window.localStorage.getItem(STORAGE_KEY) || '{}')

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Invalid prepared submission journal')
  }

  const journal = parsed as Record<string, PreparedSubmission>

  for (const [key, id] of Object.entries(readSpent())) {
    if (journal[key]?.id === id) {
      delete journal[key]
      // Storage may have recovered since: finish the retirement so the stale entry stops lingering.
      await retireJournalEntry(key).then(() => clearSpent(key), error => console.warn('[prepared-submission-retire]', error))
    } else {
      // Gone, or replaced by a newer send under the same key: the spent entry no longer exists.
      clearSpent(key)
    }
  }

  return journal
}

function clearSpent(key: string): void {
  retired.delete(key)
  writeSpent(key, undefined)
}

async function retireJournalEntry(key: string): Promise<void> {
  const native = window.hermesDesktop?.preparedSubmissions

  if (native) {
    await native.update(key, null)

    return
  }

  const journal: Record<string, PreparedSubmission> = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}')
  delete journal[key]
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(journal))
}

export function preparedSubmissionKey(
  target: string | null | undefined,
  destination: SubmissionDestination,
  rawText: string,
  attachments: ComposerAttachment[],
  options?: SubmitTextOptions
): string {
  return JSON.stringify([
    destination.scopeKey,
    target,
    options?.retryText ?? rawText,
    attachments.map(a => a.occurrenceId ?? a.id),
    options?.displayKind,
    Boolean(options?.fromQueue),
    // Slash expands once, then retries the prepared wire payload, not a new
    // generated ID/expansion. Explicit queue IDs remain distinct intents.
    options?.retryText && !options.fromQueue ? null : options?.submission_id
  ])
}

export async function listPreparedImageDrafts(target: string, scopeKey: string) {
  return Object.entries(await readJournal()).flatMap(([key, entry]) => {
    const [scope, session, text, , displayKind, fromQueue, submissionId] = JSON.parse(key)

    // An ordinary draft restores with its own identity, so sending it again is the
    // explicit retry of that entry. Queue and slash submissions own their recovery
    // (historical slash keys omit submissionId; the retained invocation is excluded).
    return scope === scopeKey && session === target && !displayKind && !fromQueue &&
      (!submissionId || submissionId === entry.id) && !String(text).trimStart().startsWith('/') &&
      !entry.legacyAttempted && entry.attachments.some(attachment => attachment.kind === 'image')
      ? [{ key, id: entry.id, text: String(text), attachments: entry.attachments }]
      : []
  })
}

export async function readPreparedSubmission(key: string): Promise<PreparedSubmission | undefined> {
  return (await readJournal())[key]
}

// Uncertain sends this window journaled or adopted: journal key -> release of its Web Lock.
// The journal is shared by every window of the origin; a held lock marks an entry whose
// window is alive, and only that window may retry it. A closed window's lock is freed, so its
// entry stays adoptable after a reload. Without Web Locks there is no other window to exclude.
const owned = new Map<string, () => void>()

function holdPreparedSubmission(key: string): Promise<boolean> {
  const locks = typeof navigator === 'undefined' ? undefined : navigator.locks

  if (owned.has(key)) { return Promise.resolve(true) }

  if (!locks) {
    owned.set(key, () => undefined)

    return Promise.resolve(true)
  }

  return new Promise<boolean>(acquired => {
    void locks.request(`${STORAGE_KEY}.${key}`, { ifAvailable: true }, lock => {
      if (!lock) {
        acquired(false)

        return null
      }

      return new Promise<void>(release => {
        owned.set(key, release)
        acquired(true)
      })
    })
  })
}

const sameDestination = (intent: string, key: string) => {
  const [scope, session] = JSON.parse(intent)
  const [savedScope, savedSession] = JSON.parse(key)

  return scope === savedScope && session === savedSession
}

/** The retained entry an explicit retry may reuse: the one carrying exactly `submissionId` on the
 *  same destination (scope + session) as `intent`, held by this window or left by a closed one.
 *  Equal text never selects an entry: a new message is a new intent even when it repeats an
 *  uncertain one, and a live other window's uncertain send is never adopted. */
export async function adoptPreparedSubmission(
  intent: string,
  submissionId: string
): Promise<{ key: string; entry: PreparedSubmission } | undefined> {
  const journal = await readJournal()

  const keys = Object.keys(journal)
    .filter(key => journal[key].id === submissionId && sameDestination(intent, key))
    .sort()

  for (const key of keys) {
    if (await holdPreparedSubmission(key)) { return { key, entry: journal[key] } }
  }

  return undefined
}

/** A journal key for a NEW send of `intent`, held by this window. Never another live window's
 *  entry, so a separate send from another window cannot overwrite or share its identity. */
export async function preparedSubmissionSlot(intent: string): Promise<string> {
  if (!(await readJournal())[intent] && (await holdPreparedSubmission(intent))) { return intent }
  const key = JSON.stringify([...JSON.parse(intent), crypto.randomUUID()])
  await holdPreparedSubmission(key)

  return key
}

export async function writePreparedSubmission(key: string, entry: PreparedSubmission): Promise<void> {
  const native = window.hermesDesktop?.preparedSubmissions

  if (native) {
    await native.update(key, JSON.stringify(entry))
    clearSpent(key)

    return
  }

  const journal: Record<string, PreparedSubmission> = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}')
  journal[key] = entry
  // Browser-only clients retain reload recovery, not a process-crash guarantee.
  // Native write failures never fall back here: sending requires their ACK.
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(journal))
  clearSpent(key)
}

/** Retire an admitted entry. Its identity is spent — durably, in a separate store — before the
 *  journal removal is attempted, so a failed removal can only leave a stale file entry that every
 *  later read (this window, a reload, another window) hides and retries, never a reusable one. */
export async function removePreparedSubmission(key: string, id: string): Promise<void> {
  retired.set(key, id)

  // The tombstone store failing too (quota) leaves the in-memory mark; the removal still runs.
  try { writeSpent(key, id) } catch (error) { console.warn('[prepared-submission-tombstone]', error) }
  await retireJournalEntry(key)
  clearSpent(key)
  owned.get(key)?.()
  owned.delete(key)
}
