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
    // Slash expands once, then retries the prepared wire payload, not a new
    // generated ID/expansion. Explicit queue IDs remain distinct intents.
    options?.retryText && !options.fromQueue ? null : options?.submission_id
  ])
}

export async function listPreparedImageDrafts(target: string, scopeKey: string) {
  return Object.entries(await readJournal()).flatMap(([key, entry]) => {
    const [scope, session, text, , displayKind, fromQueue, submissionId] = JSON.parse(key)

    // Restoring an ordinary draft must recreate its exact retry key. Queue and
    // slash submissions have additional intent fields and own their recovery.
    return scope === scopeKey && session === target && !displayKind && !fromQueue && !submissionId &&
      !entry.legacyAttempted && entry.attachments.some(attachment => attachment.kind === 'image')
      ? [{ key, text: String(text), attachments: entry.attachments }]
      : []
  })
}

export async function readPreparedSubmission(key: string): Promise<PreparedSubmission | undefined> {
  return (await readJournal())[key]
}

export async function writePreparedSubmission(key: string, entry: PreparedSubmission): Promise<void> {
  const native = window.hermesDesktop?.preparedSubmissions

  if (native) {
    await native.update(key, JSON.stringify(entry))

    return
  }

  const journal: Record<string, PreparedSubmission> = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}')
  journal[key] = entry
  // Browser-only clients retain reload recovery, not a process-crash guarantee.
  // Native write failures never fall back here: sending requires their ACK.
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(journal))
}

export async function removePreparedSubmission(key: string): Promise<void> {
  const native = window.hermesDesktop?.preparedSubmissions

  if (native) {
    await native.update(key, null)

    return
  }

  const journal: Record<string, PreparedSubmission> = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}')
  delete journal[key]
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(journal))
}
