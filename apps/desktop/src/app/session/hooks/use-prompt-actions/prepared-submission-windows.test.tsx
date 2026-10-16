import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'

import type { GatewayRequest } from './utils'

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  retainGatewayForSessionTurn: vi.fn(async () => () => undefined)
}))

// Web Locks as Chromium arbitrates them across the windows of one origin: a held
// lock is visible to every window until its holder releases it or goes away.
function sharedLocks() {
  const held = new Set<string>()

  return {
    query: async () => ({ held: [...held].map(name => ({ name })) }),
    request: async (name: string, ...rest: unknown[]) => {
      const callback = rest.at(-1) as (lock: { name: string } | null) => Promise<unknown>
      const options = (rest.length > 1 ? rest[0] : {}) as { ifAvailable?: boolean }

      if (held.has(name) && options.ifAvailable) { return callback(null) }
      held.add(name)

      try { return await callback({ name }) } finally { held.delete(name) }
    }
  }
}

// One Desktop window: its own module graph (renderer), the shared journal and locks.
async function openWindow(submit: (params: Record<string, unknown>) => unknown) {
  vi.resetModules()
  const { useSubmitPrompt } = await import('./submit')

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) =>
    (method === 'prompt.submit' ? submit(params!) : {}) as never)

  const state = createClientSessionState()

  const deps = {
    activeSessionIdRef: { current: 'runtime-a' as string | null },
    selectedStoredSessionIdRef: { current: 'stored-a' as string | null },
    busyRef: { current: false },
    copy: en.desktop,
    createBackendSessionForSend: vi.fn(async () => null),
    getRoutedStoredSessionId: () => null,
    getRuntimeIdForStoredSession: () => 'runtime-a',
    getRouteToken: () => 'same-route',
    requestGateway: requestGateway as GatewayRequest,
    runtimeIdByStoredSessionIdRef: { current: new Map([['stored-a', 'runtime-a']]) },
    resumeStoredSession: vi.fn(),
    syncAttachmentsForSubmit: vi.fn(async (sessionId: string) => ({ sessionId, attachments: [] })),
    updateSessionState: vi.fn((_sid, updater) => updater(state))
  }

  const hook = renderHook(() => useSubmitPrompt(deps))

  return { send: (text: string, options?: { submission_id: string }) => act(async () => hook.result.current(text, options)) }
}

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(navigator, 'locks')
  window.localStorage.clear()
})

it('two windows sending the same text keep separate identities; only the writer retries its own', async () => {
  Object.defineProperty(navigator, 'locks', { configurable: true, value: sharedLocks() })
  window.localStorage.clear()
  const ids: string[] = []
  let loseAck = true

  const admit = (params: Record<string, unknown>) => {
    ids.push(String(params.submission_id))

    if (loseAck) { loseAck = false; throw new Error('connection closed') }

    return { admission_id: params.submission_id, status: 'started' }
  }

  const first = await openWindow(admit)
  expect(await first.send('same text')).toBe(false)

  const second = await openWindow(admit)
  expect(await second.send('same text')).toBe(true)
  // A separate send from a live other window is its own turn, never a retry of the uncertain one,
  // and its acknowledgement must not erase the identity the first window still needs.
  expect(ids[1]).not.toBe(ids[0])

  // The writer's explicit retry reuses its own retained identity.
  expect(await first.send('same text', { submission_id: ids[0] })).toBe(true)
  expect(ids[2]).toBe(ids[0])
})
