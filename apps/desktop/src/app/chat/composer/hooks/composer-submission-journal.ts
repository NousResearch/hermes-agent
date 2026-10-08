import type { SubmitTextOptions } from '@/app/session/hooks/use-prompt-actions/utils'
import { type ComposerAttachment, freshDraftScope } from '@/store/composer'
import { type ComposerDraftRetry, matchingDraftRetry } from '@/store/composer-draft-retry'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { knownOwnerForSession } from '@/store/session-states'

import type { ComposerScope } from '../scope'

const PREFIX = 'hermes.desktop.composerSubmission.v1:'

interface ComposerSubmission {
  id: string
  text: string
  attachmentIds: string[]
  fromQueue?: boolean
}

/** sessionStorage belongs to this window and survives renderer reloads. The
 * layout group/target identifies the composer across React remounts; useId
 * identifies only a mounted surface and must never be a persistence key. */
export function composerSubmissionSlot(session: string | null, scope: ComposerScope, group: string): string {
  const owner = knownOwnerForSession(session)
  const connection = $connection.get()
  const connectionId = scope.connectionId ?? connection?.connectionId ?? connection?.baseUrl ?? 'local'
  const profile = scope.profile ?? $activeGatewayProfile.get()

  const route = owner && typeof owner === 'object'
    ? [owner.connectionId ?? connectionId, owner.profile ?? profile]
    : [connectionId, owner ?? profile]

  return PREFIX + JSON.stringify([...route, group, scope.target, session ?? freshDraftScope()])
}

export function readComposerSubmission(slot: string): ComposerSubmission | undefined {
  const raw = window.sessionStorage.getItem(slot)

  if (!raw) { return undefined }
  const value = JSON.parse(raw) as ComposerSubmission

  if (!value || typeof value.id !== 'string' || typeof value.text !== 'string' ||
      !Array.isArray(value.attachmentIds) || !value.attachmentIds.every(id => typeof id === 'string')) {
    throw new Error('Invalid composer submission journal; reopen the retained input before sending')
  }

  return value
}

export function prepareComposerSubmission(
  slot: string, text: string, attachments: ComposerAttachment[], active: ReadonlySet<string>, explicitId?: string, fromQueue?: boolean
): string {
  const saved = readComposerSubmission(slot)
  const attachmentIds = attachments.map(attachment => attachment.occurrenceId ?? attachment.id)

  // This is an exact-payload guard on one window's retained operation, never a
  // search for a matching message in the shared prepared-submission journal.
  const retry = saved && (!explicitId || explicitId === saved.id) && !active.has(saved.id) && saved.text === text &&
    JSON.stringify(saved.attachmentIds) === JSON.stringify(attachmentIds)

  const id = explicitId ?? (retry ? saved.id : crypto.randomUUID())
  window.sessionStorage.setItem(slot, JSON.stringify({ id, text, attachmentIds, fromQueue: retry ? saved.fromQueue : fromQueue }))

  return id
}

export function retireComposerSubmission(slot: string, id: string): void {
  if (readComposerSubmission(slot)?.id === id) { window.sessionStorage.removeItem(slot) }
}

export function rehomeComposerSubmission(slot: string, session: string, id: string): string {
  const saved = readComposerSubmission(slot)

  if (!saved || saved.id !== id) { return slot }
  const parts = JSON.parse(slot.slice(PREFIX.length)) as string[]
  parts[parts.length - 1] = session
  const next = PREFIX + JSON.stringify(parts)
  window.sessionStorage.setItem(next, JSON.stringify(saved))

  if (next !== slot) { retireComposerSubmission(slot, id) }

  return next
}

interface ComposerOperationInput {
  slot: string
  text: string
  attachments: ComposerAttachment[]
  active: ReadonlySet<string>
  restored?: ComposerDraftRetry
  target?: SubmitTextOptions
  displayKind?: string
  allowWindowRecovery?: boolean
}

export function prepareComposerOperation(input: ComposerOperationInput): { id: string; retry?: ComposerDraftRetry } {
  const { slot, text, attachments, active, target, displayKind } = input

  if (displayKind || (target?.fromQueue && target.submission_id)) {
    return { id: target?.submission_id ?? crypto.randomUUID() }
  }

  const restored = matchingDraftRetry(input.restored, text, attachments)
  const retainedId = restored && !active.has(restored.id) ? restored.id : undefined

  const id = prepareComposerSubmission(slot, text, attachments, active,
    target?.submission_id ?? retainedId ?? (input.allowWindowRecovery ? undefined : crypto.randomUUID()),
    target?.fromQueue ?? restored?.fromQueue)

  return { id, retry: { ...readComposerSubmission(slot)!, pending: true } }
}

export function invokeComposerSubmission(
  submit: (text: string, options?: SubmitTextOptions) => boolean | Promise<boolean>, text: string, options: SubmitTextOptions
): Promise<boolean> {
  // Route capture is synchronous with the gesture. Still settle synchronous
  // callback failures through the same restore/finally path as rejected RPCs.
  try { return Promise.resolve(submit(text, options)) } catch (error) { return Promise.reject(error) }
}
