import type { SessionMessagesResponse } from '@/types/hermes'

/**
 * Shipped-production stand-in for the private renderer perf fixture. Vite
 * aliases the real module to this typed no-op unless the dev server or the
 * isolated VITE_PERF_PROBE production renderer explicitly opts in.
 */
export interface SessionOpenPerfFixture {
  delayRuntimeMs: number
  fetchLatest: () => Promise<SessionMessagesResponse>
  requestGateway: (method: string, params?: Record<string, unknown>) => Promise<unknown>
  storedSessionId: string
}

export type SessionOpenPerfFixtureCleanup = () => void

type Runner = (fixture: SessionOpenPerfFixture) => Promise<SessionOpenPerfFixtureCleanup | void>

export function getActiveSessionOpenPerfFixture(): null {
  return null
}

export function isCompletedSessionOpenPerfFixture(_storedSessionId: string): false {
  return false
}

export function setSessionOpenPerfFixtureRunner(_next: Runner | null): void {}

export async function runSessionOpenPerfFixture(_fixture: SessionOpenPerfFixture): Promise<SessionOpenPerfFixtureCleanup> {
  throw new Error('session-open perf fixture is unavailable in a normal production renderer')
}
