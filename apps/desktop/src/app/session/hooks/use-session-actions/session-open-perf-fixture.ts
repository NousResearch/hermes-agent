import type { SessionMessagesResponse } from '@/types/hermes'

/**
 * Internal renderer-perf seam. It is registered by the mounted production
 * session-actions hook only in a perf-probe build; no app/runtime protocol or
 * backend API consumes this shape.
 */
export interface SessionOpenPerfFixture {
  delayRuntimeMs: number
  fetchLatest: (knownDisplayRevision?: number) => Promise<SessionMessagesResponse>
  requestGateway: (method: string, params?: Record<string, unknown>) => Promise<unknown>
  storedSessionId: string
}

export type SessionOpenPerfFixtureCleanup = () => void

type Runner = (fixture: SessionOpenPerfFixture) => Promise<SessionOpenPerfFixtureCleanup | void>

let activeFixture: SessionOpenPerfFixture | null = null
const completedFixtureSessionIds = new Set<string>()
let runner: Runner | null = null

export function getActiveSessionOpenPerfFixture(): SessionOpenPerfFixture | null {
  return activeFixture
}

export function isCompletedSessionOpenPerfFixture(storedSessionId: string): boolean {
  return completedFixtureSessionIds.has(storedSessionId)
}

export function setSessionOpenPerfFixtureRunner(next: Runner | null): void {
  runner = next
}

export async function runSessionOpenPerfFixture(fixture: SessionOpenPerfFixture): Promise<SessionOpenPerfFixtureCleanup> {
  if (!runner || activeFixture) {
    throw new Error('session-open perf fixture runner is unavailable')
  }

  activeFixture = fixture

  try {
    return (await runner(fixture)) ?? (() => undefined)
  } finally {
    activeFixture = null
    completedFixtureSessionIds.add(fixture.storedSessionId)
  }
}
