import type { ComposerAttachment } from '@/store/composer'
import type { SessionOwnerScope } from '@/store/session-request-router'

import type { SubmissionDestination } from './submission-destination'
import type { SubmitTextOptions } from './utils'

const STORAGE_KEY = 'hermes.desktop.preparedSubmissions.v1'
// Keep upstream's successful-admission fence if durable cleanup fails.
const retired = new Set<string>()

export interface PreparedSubmission {
  id: string
  owner: SessionOwnerScope
  attachments: ComposerAttachment[]
  text: string
  displayText?: string
  params: Record<string, unknown>
  legacyAttempted?: boolean
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

  for (const key of retired) { delete (parsed as Record<string, PreparedSubmission>)[key] }

  return parsed as Record<string, PreparedSubmission>
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
    // Identity belongs to one send, never to its text or to another window.
    // Only a caller explicitly retrying the retained input supplies this ID.
    options?.submission_id ?? crypto.randomUUID()
  ])
}

export async function listPreparedDrafts(target: string, scopeKey: string) {
  return Object.entries(await readJournal()).flatMap(([key, entry]) => {
    const [scope, session, text, , displayKind, fromQueue] = JSON.parse(key)

    // Recover one explicit operation, including text/slash sends which may no
    // longer fit in the shared draft row after another window writes there.
    return scope === scopeKey && session === target && !displayKind && !entry.legacyAttempted
      ? [{ key, submissionId: entry.id, text: String(text), attachments: entry.attachments, fromQueue: Boolean(fromQueue) }]
      : []
  })
}

export async function readPreparedSubmission(key: string): Promise<PreparedSubmission | undefined> {
  const journal = await readJournal()

  return journal[retainedSubmissionKey(journal, key)]
}

function submissionScope(key: string): unknown[] | undefined {
  try {
    const parts = JSON.parse(key)

    // The preceding release appended an eighth per-window slot to this key.
    return Array.isArray(parts) && (parts.length === 7 || parts.length === 8) ? parts : undefined
  } catch { return undefined }
}

function retainedSubmissionKey(journal: Record<string, PreparedSubmission>, key: string): string {
  if (journal[key]) { return key }
  const requested = submissionScope(key)

  if (typeof requested?.[6] !== 'string') { return key }

  // A renderer rehydration can change whitespace or chip staging metadata.
  // Only the explicitly restored UUID and exact owner/session select its
  // committed payload; equal text in another operation never grants a retry.
  return Object.keys(journal).find(candidate => {
    const saved = submissionScope(candidate)

    return journal[candidate].id === requested[6] && saved?.[0] === requested[0] && saved?.[1] === requested[1]
  }) ?? key
}

export async function readRetryablePreparedSubmission(key: string): Promise<PreparedSubmission | undefined> {
  const entry = await readPreparedSubmission(key)

  if (entry?.legacyAttempted) {
    throw new Error('The previous send may have completed, but its acknowledgement was lost. Check the conversation history before sending a new message; this send cannot be retried safely.')
  }

  return entry
}

export async function retireAcceptedSubmission(key: string, report: (error: unknown) => void): Promise<void> {
  // The admission ACK stays authoritative even when local journal cleanup fails.
  try { await removePreparedSubmission(key) } catch (error) { report(error) }
}

export async function writePreparedSubmission(key: string, entry: PreparedSubmission): Promise<void> {
  const native = window.hermesDesktop?.preparedSubmissions

  if (native) {
    const retainedKey = retainedSubmissionKey(await readJournal(), key)
    await native.update(retainedKey, JSON.stringify(entry))
    retired.delete(retainedKey)

    return
  }

  const journal: Record<string, PreparedSubmission> = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}')
  const retainedKey = retainedSubmissionKey(journal, key)
  journal[retainedKey] = entry
  // Browser-only clients retain reload recovery, not a process-crash guarantee.
  // Native write failures never fall back here: sending requires their ACK.
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(journal))
  retired.delete(retainedKey)
}

export async function removePreparedSubmission(key: string): Promise<void> {
  const native = window.hermesDesktop?.preparedSubmissions

  if (native) {
    const retainedKey = retainedSubmissionKey(await readJournal(), key)
    retired.add(retainedKey)
    await native.update(retainedKey, null)
    retired.delete(retainedKey)

    return
  }

  const journal: Record<string, PreparedSubmission> = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}')
  const retainedKey = retainedSubmissionKey(journal, key)
  retired.add(retainedKey)
  delete journal[retainedKey]
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(journal))
  retired.delete(retainedKey)
}
