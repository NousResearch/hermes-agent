import { JsonRpcGatewayError } from '@hermes/shared'
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { useSessionTileActions } from '@/app/chat/session-tile-actions'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $notifications } from '@/store/notifications'
import { setSessionTileDelegate } from '@/store/session-states'

import { actRender, Harness } from './index-test-utils'

const replies = [
  { name: 'missing', result: undefined, accepted: undefined },
  { name: 'empty', result: {}, accepted: undefined },
  { name: 'bare rejection', result: { status: 'rejected' }, accepted: undefined },
  { name: 'lost ACK', result: new Error('ack lost'), accepted: undefined },
  { name: 'explicit non-delivery', result: { status: 'rejected', delivered: false }, accepted: false },
  {
    name: 'proven refusal',
    result: new JsonRpcGatewayError('execution_not_ready', { code: 4001, data: { reason: 'execution_not_ready' } }),
    accepted: false
  },
  { name: 'legacy accepted', result: { status: 'queued' }, accepted: true }
]

const cases = (['main', 'tile'] as const).flatMap(owner =>
  [false, true].flatMap(hidden => replies.map(reply => ({ owner, hidden, ...reply })))
)

afterEach(() => {
  cleanup()
  $notifications.set([])
})

it.each(cases)(
  '$owner hidden=$hidden $name cannot authorize uncertain replay',
  async ({ owner, hidden, result, accepted }) => {
    const request = vi.fn(async () => {
      if (result instanceof Error) {
        throw result
      }

      return result as never
    })

    let invoke!: (text: string) => Promise<boolean>
    let state = createClientSessionState()

    if (owner === 'main') {
      await actRender(
        <Harness
          onReady={() => {}}
          onSeedState={next => {
            state = next as unknown as typeof state
          }}
          refreshSessions={async () => {}}
          requestGateway={request}
        >
          {actions => {
            invoke = hidden ? actions.injectHiddenPrompt : actions.redirectPrompt

            return null
          }}
        </Harness>
      )
    } else {
      setSessionTileDelegate({
        updateSession: (_id, updater) => {
          state = updater(state)

          return state
        }
      } as Parameters<typeof setSessionTileDelegate>[0])

      const hook = renderHook(() =>
        useSessionTileActions({
          requestGateway: request,
          runtimeId: 'rt-tile',
          storedSessionId: 'stored-tile',
          scope: MAIN_COMPOSER_SCOPE
        })
      )

      invoke = hidden ? hook.result.current.injectHiddenPrompt : hook.result.current.steerPrompt
    }

    await act(async () => {
      if (accepted === undefined) {
        await expect(invoke('exact correction')).rejects.toThrow(result instanceof Error ? 'ack lost' : /unconfirmed/i)
      } else {
        expect(await invoke('exact correction')).toBe(accepted)
      }
    })
    expect(request).toHaveBeenCalledTimes(1)
    expect(state.messages).toHaveLength(accepted && !hidden ? 1 : 0)
  }
)
