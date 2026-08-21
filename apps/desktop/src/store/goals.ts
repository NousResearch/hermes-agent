import { atom } from 'nanostores'

import { keyedTimeouts } from '@/lib/keyed-timeouts'

import { $gateway } from './gateway'

export type GoalStatus = 'active' | 'blocked' | 'done' | 'paused' | 'stopped' | 'unachievable' | 'waiting'

export interface SessionGoal {
  detail?: string
  status: GoalStatus
  title: string
  updatedAt: number
}

export const $goalsBySession = atom<Record<string, SessionGoal>>({})

const DONE_LINGER_MS = 8_000
const clearTimers = keyedTimeouts()

export function setSessionGoal(sid: string, goal: SessionGoal) {
  if (!sid) {
    return
  }

  clearTimers.cancel(sid)
  $goalsBySession.set({ ...$goalsBySession.get(), [sid]: goal })

  if (goal.status === 'done') {
    clearTimers.schedule(sid, DONE_LINGER_MS, () => clearSessionGoal(sid))
  }
}

export function clearSessionGoal(sid: string) {
  clearTimers.cancel(sid)

  const map = $goalsBySession.get()

  if (!(sid in map)) {
    return
  }

  const { [sid]: _drop, ...rest } = map
  $goalsBySession.set(rest)
}

const clean = (value: string): string => value.replace(/\r/g, '').trim()

const firstLine = (value: string): string => clean(value).split('\n')[0]?.trim() ?? ''

function goalTitleFromLine(line: string, pattern: RegExp): string {
  return (line.match(pattern)?.[1] ?? '').trim()
}

function goalTitleFromTerminalStatusLine(line: string, prefix: string, status: GoalStatus): string {
  if (!line.startsWith(prefix)) {
    return ''
  }

  const open = line.indexOf('(')

  if (open < 0 || !line.slice(open + 1).startsWith(`${status},`)) {
    return ''
  }

  let depth = 0

  for (let i = open; i < line.length; i += 1) {
    if (line[i] === '(') {
      depth += 1
    } else if (line[i] === ')') {
      depth -= 1

      if (depth === 0 && line[i + 1] === ':') {
        return line.slice(i + 2).trim()
      }
    }
  }

  return ''
}

function nextGoalFromText(text: string, previous?: SessionGoal): SessionGoal | null | undefined {
  const body = clean(text)
  const line = firstLine(body)

  if (!line) {
    return undefined
  }

  if (
    /^No active goal\b/i.test(line) ||
    /^No goal (?:set|to resume)\b/i.test(line) ||
    /^✓ Goal cleared\b/i.test(line)
  ) {
    return null
  }

  const now = Date.now()
  const fromSet = goalTitleFromLine(line, /^⊙ Goal set(?:\s*\([^)]*\))?:\s*(.+)$/)
  const fromActive = goalTitleFromLine(line, /^⊙ Goal\s*\([^)]*active[^)]*\):\s*(.+)$/)
  const fromResume = goalTitleFromLine(line, /^▶ Goal resumed:\s*(.+)$/)

  if (fromSet || fromActive || fromResume) {
    return { status: 'active', title: fromSet || fromActive || fromResume, updatedAt: now }
  }

  const fromWaiting = goalTitleFromLine(line, /^⏳ Goal\s*\([^)]*(?:parked|active)[^)]*\):\s*(.+)$/)

  if (fromWaiting) {
    return { status: 'waiting', title: fromWaiting, updatedAt: now }
  }

  const persistedPaused = goalTitleFromTerminalStatusLine(line, '⏸ Goal (', 'paused')
  const fromPaused = goalTitleFromLine(line, /^⏸ Goal paused:\s*(.+)$/)

  if (persistedPaused || fromPaused) {
    return { status: 'paused', title: persistedPaused || fromPaused, updatedAt: now }
  }

  const fromDone = goalTitleFromLine(line, /^✓ Goal done\s*\([^)]*\):\s*(.+)$/)

  if (fromDone) {
    return { status: 'done', title: fromDone, updatedAt: now }
  }

  if (/^↻ Continuing toward goal\b/i.test(line)) {
    return {
      detail: line.replace(/^↻\s*/, ''),
      status: 'active',
      title: previous?.title || 'Standing goal',
      updatedAt: now
    }
  }

  if (/^⏳ Goal parked\b/i.test(line)) {
    return {
      detail: line.replace(/^⏳\s*/, ''),
      status: 'waiting',
      title: previous?.title || 'Standing goal',
      updatedAt: now
    }
  }

  if (/^⏸ Goal paused\b/i.test(line)) {
    return {
      detail: line.replace(/^⏸\s*/, ''),
      status: 'paused',
      title: previous?.title || 'Standing goal',
      updatedAt: now
    }
  }

  const terminalStatuses = [
    {
      eventPattern: /^⚠ Goal blocked\b/i,
      statusLinePrefix: '⚠ Goal (',
      status: 'blocked' as const
    },
    {
      eventPattern: /^■ Goal stopped\b/i,
      statusLinePrefix: '■ Goal (',
      status: 'stopped' as const
    },
    {
      eventPattern: /^✗ Goal unachievable\b/i,
      statusLinePrefix: '✗ Goal (',
      status: 'unachievable' as const
    }
  ]

  const persistedTerminal = terminalStatuses
    .map(candidate => ({
      ...candidate,
      title: goalTitleFromTerminalStatusLine(line, candidate.statusLinePrefix, candidate.status)
    }))
    .find(candidate => candidate.title)

  if (persistedTerminal) {
    return {
      status: persistedTerminal.status,
      title: persistedTerminal.title,
      updatedAt: now
    }
  }

  const terminal = terminalStatuses.find(candidate => candidate.eventPattern.test(line))

  if (terminal) {
    return {
      detail: line.replace(terminal.eventPattern, '').replace(/^:\s*/, ''),
      status: terminal.status,
      title: previous?.title || 'Standing goal',
      updatedAt: now
    }
  }

  if (/^✓ Goal achieved\b/i.test(line)) {
    return {
      detail: line.replace(/^✓\s*/, ''),
      status: 'done',
      title: previous?.title || 'Standing goal',
      updatedAt: now
    }
  }

  return undefined
}

export function applyGoalStatusText(sid: string, text: string) {
  if (!sid) {
    return
  }

  const next = nextGoalFromText(text, $goalsBySession.get()[sid])

  if (next === null) {
    clearSessionGoal(sid)
  } else if (next) {
    setSessionGoal(sid, next)
  }
}

export async function refreshSessionGoal(sid: string): Promise<void> {
  const gateway = $gateway.get()

  if (!sid || !gateway) {
    return
  }

  try {
    const result = await gateway.request<{ output?: string }>('slash.exec', { command: 'goal status', session_id: sid })
    applyGoalStatusText(sid, result?.output ?? '')
  } catch {
    // Best-effort: older gateways or detached sessions simply won't hydrate it.
  }
}
