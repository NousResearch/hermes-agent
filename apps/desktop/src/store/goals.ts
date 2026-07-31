import { atom } from 'nanostores'

import { $gateway } from './gateway'

export type GoalStatus = 'active' | 'done' | 'paused' | 'waiting'

export interface SessionGoal {
  detail?: string
  status: GoalStatus
  title: string
  updatedAt: number
}

export const $goalsBySession = atom<Record<string, SessionGoal>>({})

const DONE_LINGER_MS = 8_000
const clearTimers = new Map<string, ReturnType<typeof setTimeout>>()

function cancelScheduledClear(sid: string) {
  const timer = clearTimers.get(sid)

  if (timer !== undefined) {
    clearTimeout(timer)
    clearTimers.delete(sid)
  }
}

export function setSessionGoal(sid: string, goal: SessionGoal) {
  if (!sid) {
    return
  }

  cancelScheduledClear(sid)
  $goalsBySession.set({ ...$goalsBySession.get(), [sid]: goal })

  if (goal.status === 'done') {
    clearTimers.set(
      sid,
      setTimeout(() => {
        clearTimers.delete(sid)
        clearSessionGoal(sid)
      }, DONE_LINGER_MS)
    )
  }
}

export function clearSessionGoal(sid: string) {
  cancelScheduledClear(sid)

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

  const fromPaused = goalTitleFromLine(line, /^⏸ Goal(?:\s*\([^)]*\)| paused)?:\s*(.+)$/)

  if (fromPaused) {
    return { status: 'paused', title: fromPaused, updatedAt: now }
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
