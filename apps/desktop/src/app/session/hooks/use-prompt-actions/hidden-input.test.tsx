import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { useSessionTileActions } from '@/app/chat/session-tile-actions'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, $messages } from '@/store/session'
import { $sessionStates, $sessionTiles, setSessionTileDelegate } from '@/store/session-states'

import { clearSingleFlightSessionResumeState } from './single-flight-resume'

import { usePromptActions } from './index'

function mountHiddenInput(tile: boolean, requestGateway: Parameters<typeof usePromptActions>[0]['requestGateway']) {
  const activeSessionIdRef = { current: 'runtime' }
  const updateSessionState = vi.fn((_sid, update) => update(createClientSessionState()))

  setSessionTileDelegate({
    archiveSession: vi.fn(),
    branchSession: vi.fn(),
    deleteSession: vi.fn(),
    executeSlash: vi.fn(),
    interruptSession: vi.fn(),
    resumeTile: vi.fn(),
    submitToSession: vi.fn(),
    updateSession: updateSessionState
  })
  $sessionTiles.set([{ runtimeId: 'runtime', storedSessionId: 'stored' }])

  const result = tile
    ? renderHook(() =>
        useSessionTileActions({
          requestGateway,
          runtimeId: 'runtime',
          storedSessionId: 'stored',
          scope: MAIN_COMPOSER_SCOPE
        })
      ).result
    : renderHook(() =>
        usePromptActions({
          activeSessionId: 'runtime',
          activeSessionIdRef,
          branchCurrentSession: async () => true,
          busyRef: { current: true },
          createBackendSessionForSend: async () => 'runtime',
          getRoutedStoredSessionId: () => null,
          getRuntimeIdForStoredSession: () => null,
          getRouteToken: () => 'stable',
          handleSkinCommand: () => '',
          openMemoryGraph: () => undefined,
          refreshSessions: async () => undefined,
          requestGateway,
          resumeStoredSession: async () => undefined,
          runtimeIdByStoredSessionIdRef: { current: new Map([['stored', 'runtime']]) },
          selectedStoredSessionIdRef: { current: 'stored' },
          startFreshSessionDraft: () => undefined,
          sttEnabled: false,
          updateSessionState
        })
      ).result

  return { result, updateSessionState }
}

afterEach(() => {
  cleanup()
  clearSingleFlightSessionResumeState()
  $activeSessionId.set(null)
  $messages.set([])
  $sessionStates.set({})
  $sessionTiles.set([])
})

describe.each([
  ['primary', false],
  ['tile', true]
] as const)('%s hidden input', (_surface, tile) => {
  it('declares hidden visibility and accepts queued or streaming without creating an optimistic turn', async () => {
    const requestGateway = vi.fn(async () => ({ status: 'queued' }) as never)
    const { result, updateSessionState } = mountHiddenInput(tile, requestGateway)
    const messages = $messages.get()

    for (const [status, accepted] of [
      ['queued', true],
      ['streaming', true],
      ['rejected', false],
      [undefined, false]
    ] as const) {
      requestGateway.mockResolvedValueOnce({ status } as never)
      const handled = await act(async () => result.current.injectHiddenPrompt('  internal tool context  '))

      expect(requestGateway).toHaveBeenLastCalledWith('session.steer', {
        session_id: 'runtime',
        text: 'internal tool context',
        input_visibility: 'hidden'
      })
      expect(handled, String(status)).toBe(accepted)
    }

    expect(updateSessionState).not.toHaveBeenCalled()
    expect($messages.get()).toBe(messages)
  })

  it('retains hidden visibility across runtime recovery and reports failures without a fallback user turn', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.resume') {
        return { session_id: 'recovered' } as never
      }

      if (params?.session_id === 'runtime') {
        throw new Error('session not found')
      }

      return { status: 'streaming' } as never
    })

    const { result, updateSessionState } = mountHiddenInput(tile, requestGateway)

    expect(await act(async () => result.current.injectHiddenPrompt('  '))).toBe(false)
    expect(requestGateway).not.toHaveBeenCalled()
    expect(await act(async () => result.current.injectHiddenPrompt('internal context'))).toBe(true)
    expect(calls.map(call => call.method)).toEqual(['session.steer', 'session.resume', 'session.steer'])
    expect(calls.filter(call => call.method === 'session.steer').map(call => call.params)).toEqual([
      { session_id: 'runtime', text: 'internal context', input_visibility: 'hidden' },
      { session_id: 'recovered', text: 'internal context', input_visibility: 'hidden' }
    ])

    requestGateway.mockRejectedValueOnce(new Error('connection unavailable'))
    expect(await act(async () => result.current.injectHiddenPrompt('more context'))).toBe(false)
    expect(updateSessionState).not.toHaveBeenCalled()
  })
})
