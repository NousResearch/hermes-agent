import { describe, expect, it, vi } from 'vitest'

// The store wires itself to gateway/profile atoms and the REST layer at import
// time paths; mock the seams (same shape as updates.test.ts) so this test only
// exercises the pure transition state machine.
vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(),
  testMcpServer: vi.fn()
}))

vi.mock('@/i18n', () => ({
  translateNow: (key: string) => key
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn()
}))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: { get: () => 'default', listen: () => () => {} },
  normalizeProfileKey: (name: string | null | undefined) => (name ?? '').trim() || 'default'
}))

vi.mock('@/store/session', () => ({
  $gatewayState: { get: () => 'closed', subscribe: () => () => {} }
}))

const { shouldNotifyOnTransition } = await import('./mcp-health')

type Status = 'error' | 'needs-auth' | 'ok'

describe('shouldNotifyOnTransition', () => {
  // The full previous × next decision table: notify only on a TRANSITION into
  // a bad state. Rechecks of an already-bad server stay quiet; ok never nudges.
  it.each<[previous: Status | null, next: Status, notify: boolean]>([
    // First observation of the session (previous unknown).
    [null, 'ok', false],
    [null, 'needs-auth', true],
    [null, 'error', true],
    // Healthy server stays healthy / breaks.
    ['ok', 'ok', false],
    ['ok', 'needs-auth', true],
    ['ok', 'error', true],
    // Already-broken server: rechecks must NOT re-notify…
    ['needs-auth', 'needs-auth', false],
    ['error', 'error', false],
    // …but flipping from one bad state to the other is a new transition.
    ['needs-auth', 'error', true],
    ['error', 'needs-auth', true],
    // Recovery is silent.
    ['needs-auth', 'ok', false],
    ['error', 'ok', false]
  ])('previous=%s next=%s → notify=%s', (previous, next, expected) => {
    expect(shouldNotifyOnTransition(previous, next)).toBe(expected)
  })
})
