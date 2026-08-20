import { atom } from 'nanostores'

import { keyedTimeouts } from '@/lib/keyed-timeouts'

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

function applyGoalText(sid: string, text: string, { allowClear }: { allowClear: boolean }) {
  if (!sid) {
    return
  }

  const next = nextGoalFromText(text, $goalsBySession.get()[sid])

  if (next === null) {
    if (allowClear) {
      clearSessionGoal(sid)
    }

    return
  }

  if (next) {
    setSessionGoal(sid, next)
  }
}

export function applyGoalStatusText(sid: string, text: string) {
  applyGoalText(sid, text, { allowClear: true })
}

const hydrationGeneration = new Map<string, number>()
const HYDRATION_RETRY_MS = [0, 500, 1500, 4000]

export function applyGoalHydrationText(sid: string, text: string) {
  // A best-effort status read can resolve a different live session and reply
  // "No active goal". That must not wipe a row the user still has set.
  applyGoalText(sid, text, { allowClear: false })
}

export async function refreshSessionGoal(sid: string): Promise<void> {
  const gateway = $gateway.get()

  if (!sid || !gateway) {
    return
  }

  const generation = (hydrationGeneration.get(sid) ?? 0) + 1
  hydrationGeneration.set(sid, generation)

  for (let attempt = 0; attempt < HYDRATION_RETRY_MS.length; attempt += 1) {
    const waitMs = HYDRATION_RETRY_MS[attempt] ?? 0

    if (waitMs > 0) {
      await new Promise(resolve => setTimeout(resolve, waitMs))
    }

    if (hydrationGeneration.get(sid) !== generation) {
      return
    }

    try {
      const result = await gateway.request<{ output?: string }>('slash.exec', {
        command: 'goal status',
        session_id: sid
      })

      if (hydrationGeneration.get(sid) !== generation) {
        return
      }

      applyGoalHydrationText(sid, result?.output ?? '')

      return
    } catch {
      // Timeout / older gateway / detached runtime — retry the remaining windows.
    }
  }
}
