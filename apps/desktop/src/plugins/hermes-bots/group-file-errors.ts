/** User-facing failure categories and the Files interaction budget. */

export type GroupFileFailure = 'gone' | 'verification' | 'timeout' | 'offline' | 'unavailable' | 'access' | 'error'

export class GroupFileError extends Error {
  constructor(
    readonly kind: GroupFileFailure,
    message: string = kind
  ) {
    super(message)
    this.name = 'GroupFileError'
  }
}

// Exact room-level outcomes from the catalog/service/viewer-read contracts.
// A 4141/4142 alone, or an individual attachment's expiry, is not revocation.
const ROOM_ACCESS_DENIALS = new Set([
  'this group chat is managed by another gateway',
  'hosted room not found',
  'group chat history expired; room_id remains permanently retired',
  'hosted room is being disbanded',
  'room quarantined',
  'room authority is quarantined',
  'stale hosted room authority',
  'hosted room authority changed',
  'attachment catalogue is unavailable for this room authority',
  'group chat is unavailable to viewers',
  'group chat viewer authority changed'
])

export function assertGroupFileIntent(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException('Cancelled', 'AbortError')
  }
}

export function groupFileFailure(error: unknown): GroupFileFailure {
  if (error instanceof GroupFileError) {
    return error.kind
  }

  const outer = error as { code?: unknown; message?: unknown; error?: { code?: unknown; message?: unknown } } | null
  const code = outer?.code ?? outer?.error?.code
  const message = String(outer?.message || outer?.error?.message || '')
  const normalized = message.trim().toLowerCase().replace(/\.$/, '')

  if (
    (code === 4141 || code === 4142) &&
    (ROOM_ACCESS_DENIALS.has(normalized) ||
      normalized.startsWith(
        'this group chat has an unverified authority takeover and is read-only until its history is reconciled ('
      ))
  ) {
    return 'access'
  }

  if (/timed? out|timeout/i.test(message)) {
    return 'timeout'
  }

  if (code === -32601) {
    return 'unavailable'
  }

  if (code === 4141 && /invalid attachment|integrity|SHA-256|blob.*(?:size|validation)/i.test(message)) {
    return 'verification'
  }

  if (code === 4141 && /expired|not committed|not found|unavailable|not owned|disbanded/i.test(message)) {
    return 'gone'
  }

  if (/offline|connection.*(?:closed|lost|unavailable|failed)|socket/i.test(message)) {
    return 'offline'
  }

  return 'error'
}

export async function withGroupFileDeadline<T>(task: Promise<T>, signal?: AbortSignal): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  let abort: (() => void) | undefined

  try {
    return await Promise.race([
      task,
      new Promise<never>((_resolve, reject) => {
        abort = () => reject(new DOMException('Cancelled', 'AbortError'))

        if (signal?.aborted) {
          return abort()
        }

        signal?.addEventListener('abort', abort, { once: true })
        // The real RPC already has a 30s timeout. Files uses a tighter end-to-end UX budget.
        timer = setTimeout(() => reject(new GroupFileError('timeout', 'Files request timed out')), 10_000)
      })
    ])
  } finally {
    clearTimeout(timer)

    if (abort) {
      signal?.removeEventListener('abort', abort)
    }
  }
}
