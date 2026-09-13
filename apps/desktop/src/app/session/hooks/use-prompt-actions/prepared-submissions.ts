import type { ComposerAttachment } from '@/store/composer'
import type { SessionOwnerScope } from '@/store/session-request-router'

import { compareJournal, journalOwner, readJournal } from './prepared-journal-storage'
import type { SubmissionDestination } from './submission-destination'
import type { SubmitTextOptions } from './utils'

export interface PreparedSubmission {
  id: string
  owner: SessionOwnerScope
  attachments: ComposerAttachment[]
  text: string
  displayText?: string
  params: Record<string, unknown>
  legacyAttempted?: boolean
  acknowledged?: boolean
  journal?: { lookup: string; storageKey: string; owner: string }
}

const snapshots = new WeakMap<PreparedSubmission, string>()
const acknowledged = new Set<string>()
const recoverySelection = new Map<string, string>()

function forgetSelection(entry: PreparedSubmission) {
  if (!entry.journal) {return}
  const key = JSON.stringify([entry.journal.owner, entry.journal.lookup])

  if (recoverySelection.get(key) === entry.journal.storageKey) {recoverySelection.delete(key)}
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

export async function listPreparedDrafts(target: string, scopeKey: string) {
  return Object.entries(await readJournal<PreparedSubmission>()).flatMap(([key, entry]) => {
    // The origin journal also hosts other prepared operations, not just Sends.
    if (!entry || typeof entry.id !== 'string' || typeof entry.text !== 'string' || !Array.isArray(entry.attachments)) {return []}
    const [scope, session, text, , displayKind, fromQueue, submissionId] = JSON.parse(entry.journal?.lookup ?? key)

    // Restoring an ordinary draft must recreate its exact retry key. Queue and
    // slash submissions own their recovery. Historical slash keys intentionally
    // omit submissionId, so the retained invocation must also be excluded.
    return scope === scopeKey && session === target && !displayKind && !fromQueue && !submissionId &&
      !String(text).trimStart().startsWith('/') &&
      !entry.legacyAttempted && !entry.acknowledged && !acknowledged.has(key)
      ? [{ key, text: String(text), attachments: entry.attachments, expected: JSON.stringify(entry) }]
      : []
  })
}

export async function readPreparedSubmission(key: string): Promise<PreparedSubmission | undefined> {
  const owner = await journalOwner()
  const explicitId = JSON.parse(key)[6]

  const matches = Object.entries(await readJournal<PreparedSubmission>()).filter(([storageKey, entry]) =>
    entry.journal?.lookup === key && entry.journal.storageKey === storageKey && (entry.journal.owner === owner || explicitId === entry.id) &&
    (!(entry.acknowledged || acknowledged.has(storageKey)) || explicitId === entry.id))

  // Content is a window-local recovery index, never a shared intent lock.
  const selected = recoverySelection.get(JSON.stringify([owner, key]))

  if (matches.length > 1 && !selected) {throw new Error('Multiple prepared drafts require explicit recovery')}
  let entry = (selected ? matches.find(([storageKey]) => storageKey === selected) : matches[0])?.[1]

  if (entry?.journal && entry.journal.owner !== owner) {
    const next = { ...entry, journal: { ...entry.journal, owner } }

    if (!await compareJournal(entry.journal.storageKey, JSON.stringify(entry), JSON.stringify(next), true)) {throw new Error('Prepared draft changed during recovery')}
    entry = next
  }

  if (entry) {
    snapshots.set(entry, JSON.stringify(entry))

    if (acknowledged.has(entry.journal!.storageKey)) {entry.acknowledged = true}
  }

  return entry
}

export async function writePreparedSubmission(key: string, entry: PreparedSubmission, freshGenerated = false): Promise<void> {
  const owner = await journalOwner()
  const [scope, session] = JSON.parse(key)

  if (entry.journal && (entry.journal.owner !== owner || entry.journal.lookup !== key)) {throw new Error('Prepared draft ownership changed')}
  // Caller IDs belong to a destination/session deduplication domain. Keep
  // existing records at their recorded address; only new records use v2 keys.
  entry.journal ??= { lookup: key, storageKey: JSON.stringify(['intent-v2', scope, session, entry.id]), owner }
  const expected = snapshots.get(entry) ?? null
  const next = JSON.stringify(entry)

  if (!await compareJournal(entry.journal.storageKey, expected, next, false, freshGenerated)) {throw new Error('Prepared draft changed before write')}
  snapshots.set(entry, next)
}

export async function removePreparedSubmission(key: string, entry?: PreparedSubmission): Promise<void> {
  // No proof of the exact prepared entry means no authority to retire a slot.
  if (!entry?.journal || entry.journal.lookup !== key) {return}

  if (entry.journal.owner !== await journalOwner()) {throw new Error('Prepared draft ownership changed')}
  const expected = snapshots.get(entry)

  if (!expected || !await compareJournal(entry.journal.storageKey, expected, null)) {throw new Error('Prepared draft changed before retirement')}
  snapshots.delete(entry)
  forgetSelection(entry)
}

export async function claimPreparedSubmission(storageKey: string, expected?: string): Promise<void> {
  const entry = (await readJournal<PreparedSubmission>())[storageKey]

  if (!entry || entry.legacyAttempted || entry.acknowledged || acknowledged.has(storageKey)) {throw new Error('Prepared draft unavailable')}

  if (expected !== undefined && JSON.stringify(entry) !== expected) {throw new Error('Prepared draft changed during recovery')}
  const owner = await journalOwner()
  const next = { ...entry, journal: { lookup: entry.journal?.lookup ?? storageKey, storageKey, owner } }

  if (!await compareJournal(storageKey, JSON.stringify(entry), JSON.stringify(next), true)) {throw new Error('Prepared draft changed during recovery')}
  recoverySelection.set(JSON.stringify([owner, next.journal.lookup]), storageKey)
}

export async function settlePreparedSubmission(key: string, entry: PreparedSubmission): Promise<void> {
  if (!entry.journal) {return}
  acknowledged.add(entry.journal.storageKey)
  entry.acknowledged = true
  forgetSelection(entry)

  try {
    await writePreparedSubmission(key, entry)
    acknowledged.delete(entry.journal.storageKey)
  } catch (error) {console.warn('Accepted submission journal acknowledgement could not be saved', error)}

  try {
    await removePreparedSubmission(key, entry)
    acknowledged.delete(entry.journal.storageKey)
  } catch (error) {
    // Never retry admission or clear turn state because local housekeeping
    // failed. A surviving pending record still carries the original input ID.
    console.warn('Accepted submission journal cleanup failed', error)
  }
}
