import { atom } from 'nanostores'

import { AGENT_MESSAGE_RE, messageContentText, PROCESS_NOTIFICATION_RE } from '@/components/assistant-ui/thread/content'
import { connectionScopedAtom } from '@/lib/connection-scoped'
import { Codecs } from '@/lib/persisted'

export interface RecoveryMessage {
  id: string
  role: string
  content: unknown
  status?: { type: string; reason?: string }
  metadata?: { custom?: Record<string, unknown> }
}

export type RecoveryStatus = 'missing' | 'unfinished'
const excludedKinds = new Set(['hidden', 'steer', 'compaction', 'compaction_summary', 'summary'])

/** Presentation only: absence of a visible reply says nothing about execution. */
export function recoveryStatuses(messages: readonly RecoveryMessage[]): ReadonlyMap<string, RecoveryStatus> {
  const result = new Map<string, RecoveryStatus>()
  let replied = false
  let failed = false
  let activity = false

  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]
    const kind = message.metadata?.custom?.displayKind

    if (message.metadata?.custom?.hidden || (typeof kind === 'string' && excludedKinds.has(kind))) {continue}
    const text = messageContentText(message.content)

    if (message.role === 'user' && (PROCESS_NOTIFICATION_RE.test(text) || AGENT_MESSAGE_RE.test(text))) {continue}

    if (message.role === 'assistant') {
      failed ||= message.status?.reason === 'error' || message.status?.type === 'running'

      const visible =
        Boolean(text) ||
        (Array.isArray(message.content) && message.content.some(part => ['image', 'audio', 'file'].includes(part.type)))

      replied ||= !message.metadata?.custom?.interim && visible
      activity ||= Boolean(message.metadata?.custom?.interim) && visible
      activity ||=
        Array.isArray(message.content) &&
        message.content.some(part => part.type === 'reasoning' || part.type === 'tool-call')
    } else if (message.role === 'user' && !message.id.startsWith('user-queued-')) {
      if (!replied && !failed) {result.set(message.id, activity ? 'unfinished' : 'missing')}
      replied = false
      failed = false
      activity = false
    }
  }

  return result
}

export interface RecoveryScope {
  connection: string | null
  profile: string | null
  session: string | null
  durable: boolean
  ready: boolean
  pending: boolean
}

export const $hiddenRecoveryRows = connectionScopedAtom(
  'hermes.desktop.hiddenMessageRows',
  [] as string[],
  Codecs.stringArray
)
export const $hiddenRecoveryEphemeral = atom<string[]>([])

export function recoveryVisibilityKey(
  scope: RecoveryScope,
  message: RecoveryMessage
): { key: string; persistent: boolean } | null {
  if (!scope.session) {return null}
  const rowId = message.metadata?.custom?.rowId

  const persistent =
    scope.durable &&
    Boolean(scope.connection && scope.profile) &&
    typeof rowId === 'number' &&
    Number.isSafeInteger(rowId) &&
    rowId > 0

  return {
    key: JSON.stringify([scope.connection, scope.profile, scope.session, persistent ? rowId : message.id]),
    persistent
  }
}

export function setRecoveryHidden(identity: { key: string; persistent: boolean }, hidden: boolean): void {
  const store = identity.persistent ? $hiddenRecoveryRows : $hiddenRecoveryEphemeral
  const current = store.get()
  store.set(
    hidden
      ? current.includes(identity.key)
        ? current
        : [...current, identity.key]
      : current.filter(key => key !== identity.key)
  )
}
