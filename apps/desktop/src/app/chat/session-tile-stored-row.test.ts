import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

// Keep the session-tile import focused on ownership resolution. The component's
// normal gateway/project wiring is outside this pure lookup test.
vi.mock('@/store/gateway', () => ({
  $gateway: atom<unknown>(null),
  activeGateway: () => null,
  ensureGatewayForProfile: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined)
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({
  invalidateProfileScopedQueries: vi.fn(),
  queryClient: { invalidateQueries: vi.fn() }
}))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))
vi.mock('@/store/session-unread-remote', async () => {
  const actual = await vi.importActual<Record<string, unknown>>('@/store/session-unread-remote')

  return { ...actual, watchUnreadWriteGuard: vi.fn() }
})

const { tileStoredRow } = await import('./session-tile')
const { $sessions } = await import('@/store/session')
const { $activeGatewayProfile, $gatewaySwapTarget, $profileScope, $showAllProfiles } = await import('@/store/profile')

const row = (id: string, profile: string, title: string): SessionInfo =>
  ({
    archived: false,
    cwd: null,
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    profile,
    source: 'desktop',
    started_at: 0,
    title,
    tool_call_count: 0
  }) as SessionInfo

beforeEach(() => {
  $sessions.set([])
  $showAllProfiles.set(false)
  $activeGatewayProfile.set('default')
  $gatewaySwapTarget.set(null)
})

afterEach(() => {
  $sessions.set([])
  $gatewaySwapTarget.set(null)
})

describe('tileStoredRow ownership', () => {
  it('never titles a tab with a same-id row owned by another profile', () => {
    $activeGatewayProfile.set('work')
    $sessions.set([
      row('same-id', 'work', 'Work chat'),
      row('same-id', 'default', 'Default chat'),
      row('work-only', 'work', 'Work only')
    ])

    expect($profileScope.get()).toBe('work')
    expect(tileStoredRow('same-id')?.title).toBe('Work chat')
    expect(tileStoredRow('work-only')?.title).toBe('Work only')
  })

  it('resolves against the pending swap target, not the settled gateway', () => {
    $activeGatewayProfile.set('default')
    $gatewaySwapTarget.set('work')
    $sessions.set([row('same-id', 'work', 'Work chat'), row('same-id', 'default', 'Default chat')])

    // The scope follows the pending target the moment the rail is clicked,
    // while the gateway is still settling on default.
    expect($profileScope.get()).toBe('work')
    expect(tileStoredRow('same-id')?.title).toBe('Work chat')
  })
})
