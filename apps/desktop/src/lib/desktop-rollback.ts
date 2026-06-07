import type { ChatMessage } from '@/lib/chat-messages'

export interface RollbackCheckpoint {
  hash: string
  message?: string
  timestamp?: string
}

export interface RollbackListResponse {
  checkpoints?: RollbackCheckpoint[]
  enabled?: boolean
}

export interface RollbackDiffResponse {
  diff?: string
  rendered?: string
  stat?: string
}

export interface RollbackRestoreResponse {
  error?: string
  history_removed?: number
  message?: string
  reason?: string
  restored_to?: string
  success?: boolean
}

/**
 * Parsed intent of a `/rollback …` invocation. Each variant maps 1:1 to the
 * native gateway RPC the desktop must call — `list` → `rollback.list`,
 * `diff` → `rollback.diff`, `restore` → `rollback.restore`. `usage` short-circuits
 * with a help line and hits no RPC.
 *
 * Routing through these RPCs (instead of `slash.exec`) is the whole point: the
 * slash worker is a throwaway HermesCLI subprocess that restores the file but
 * only undoes its OWN in-memory history copy, leaving the live gateway session
 * (and therefore the agent's next-turn context) out of sync with the restored
 * files. `rollback.restore` mutates the live `session["history"]` and reports
 * how many messages it dropped via `history_removed`.
 */
export type RollbackPlan =
  | { kind: 'diff'; hash: string }
  | { kind: 'list' }
  | { kind: 'restore'; filePath: string; hash: string }
  | { kind: 'usage'; message: string }

// Mirror the CLI's `/rollback diff` cap (cli.py::_handle_rollback_command) so a
// large diff doesn't flood the transcript with one giant system bubble.
export const ROLLBACK_DIFF_LINE_LIMIT = 80

export function parseRollbackCommand(arg: string): RollbackPlan {
  const trimmed = arg.trim()
  const [first = '', ...rest] = trimmed.split(/\s+/).filter(Boolean)
  const lower = first.toLowerCase()

  if (!trimmed || lower === 'list' || lower === 'ls') {
    return { kind: 'list' }
  }

  if (lower === 'diff') {
    const hash = rest[0]

    if (!hash) {
      return { kind: 'usage', message: 'usage: /rollback diff <checkpoint>' }
    }

    return { hash, kind: 'diff' }
  }

  return { filePath: rest.join(' ').trim(), hash: first, kind: 'restore' }
}

export function formatCheckpointList(r: RollbackListResponse): string {
  if (!r.enabled) {
    return 'checkpoints are not enabled'
  }

  const checkpoints = r.checkpoints ?? []

  if (!checkpoints.length) {
    return 'no checkpoints found'
  }

  return [
    'Rollback checkpoints',
    ...checkpoints.map((c, idx) => {
      const meta = [c.timestamp, c.message].filter(Boolean).join(' · ') || '(no metadata)'

      return `${idx + 1}. ${c.hash.slice(0, 10)}  ${meta}`
    })
  ].join('\n')
}

export function formatRollbackDiff(r: RollbackDiffResponse): string {
  const stat = (r.stat || '').trim()
  // Use the plain unified diff, not the ANSI-coloured `rendered` field (that is
  // for the TUI pager and would render as escape-code noise in the transcript).
  const diff = (r.diff || '').trim()

  if (!stat && !diff) {
    return 'no changes since this checkpoint'
  }

  const lines = diff ? diff.split('\n') : []

  const shown =
    lines.length > ROLLBACK_DIFF_LINE_LIMIT
      ? [
          ...lines.slice(0, ROLLBACK_DIFF_LINE_LIMIT),
          `… ${lines.length - ROLLBACK_DIFF_LINE_LIMIT} more lines (showing first ${ROLLBACK_DIFF_LINE_LIMIT})`
        ]
      : lines

  return [stat, shown.join('\n')].filter(Boolean).join('\n\n')
}

export function formatRollbackRestore(r: RollbackRestoreResponse, filePath: string): string {
  if (!r.success) {
    return `rollback failed: ${r.error || r.message || 'unknown error'}`
  }

  const target = filePath || 'workspace'
  const detail = r.reason || r.message || r.restored_to || 'restored'

  return `rollback restored ${target}: ${detail}`
}

/**
 * Drop the last user/assistant exchange from the visible transcript, matching
 * what a full `rollback.restore` removed from the live session history server-side
 * (and the TUI's `trimLastExchange`): pop trailing assistant/tool messages, then
 * the user turn that started the exchange.
 */
export function trimLastExchange(messages: ChatMessage[]): ChatMessage[] {
  const next = [...messages]

  while (next.length && (next.at(-1)?.role === 'assistant' || next.at(-1)?.role === 'tool')) {
    next.pop()
  }

  if (next.at(-1)?.role === 'user') {
    next.pop()
  }

  return next
}
