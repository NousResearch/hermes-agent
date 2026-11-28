import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'

import type { GatewayRequest } from './utils'

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  retainGatewayForSessionTurn: vi.fn(async () => () => undefined)
}))

// Web Locks as WebKit arbitrates them: names over 1024 UTF-16 code units are refused (#151619).
function webkitLocks(refuse?: DOMException) {
  const held = new Set<string>()

  return {
    request: async (name: string, ...rest: unknown[]) => {
      if (refuse) { throw refuse }

      if (name.length > 1024) { throw new DOMException('Lock name cannot be longer than 1024 characters', 'NotSupportedError') }
      const callback = rest.at(-1) as (lock: { name: string } | null) => Promise<unknown>
      const options = (rest.length > 1 ? rest[0] : {}) as { ifAvailable?: boolean }

      if (held.has(name) && options.ifAvailable) { return callback(null) }
      held.add(name)

      try { return await callback({ name }) } finally { held.delete(name) }
    }
  }
}

async function composer() {
  vi.resetModules()
  const { useSubmitPrompt } = await import('./submit')
  const submitted: Record<string, unknown>[] = []
  let state: ClientSessionState = createClientSessionState()

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'prompt.submit') {
      submitted.push(params!)

      return { admission_id: params!.submission_id, status: 'started' } as never
    }

    return {} as never
  })

  const hook = renderHook(() =>
    useSubmitPrompt({
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
      updateSessionState: vi.fn((_sid: string, updater: (current: ClientSessionState) => ClientSessionState) => {
        state = updater(state)

        return state
      })
    })
  )

  return {
    send: (text: string) => act(async () => hook.result.current(text)),
    submitted,
    state: () => state
  }
}

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(navigator, 'locks')
  window.localStorage.clear()
})

it('a long prompt under WebKit lock-name limits reaches prompt.submit with its full text', async () => {
  Object.defineProperty(navigator, 'locks', { configurable: true, value: webkitLocks() })
  const text = 'a long pasted paragraph '.repeat(400).trim()
  const ui = await composer()

  const sent = await Promise.race([ui.send(text), new Promise(resolve => setTimeout(() => resolve('pending'), 3000))])

  expect(sent).toBe(true)
  expect(ui.submitted.map(params => params.text)).toEqual([text])
})

it('a refused Web Lock ends the send with a visible error instead of a pending composer', async () => {
  Object.defineProperty(navigator, 'locks', {
    configurable: true,
    value: webkitLocks(new DOMException('Lock access refused', 'SecurityError'))
  })
  const ui = await composer()

  const sent = await Promise.race([ui.send('hello'), new Promise(resolve => setTimeout(() => resolve('pending'), 3000))])

  expect(sent).toBe(false)
  expect(ui.submitted).toEqual([])
  expect(ui.state().busy).toBe(false)
  expect(ui.state().messages.some(message => message.role === 'assistant' && message.error)).toBe(true)
})
