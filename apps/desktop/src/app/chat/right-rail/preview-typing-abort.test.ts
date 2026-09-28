import { afterEach, expect, it, vi } from 'vitest'

import { handleInputRequestEvent } from '@/app/session/hooks/use-message-stream/gateway-event/input-requests'
import {
  handleServerRequest,
  type ServerRequestContext
} from '@/app/session/hooks/use-message-stream/gateway-event/server-requests'
import type { GatewayEventContext } from '@/app/session/hooks/use-message-stream/gateway-event/types'
import { $rightRailActiveTabId } from '@/store/layout'
import { closeRightRail, openPreview } from '@/store/preview'

import { registerPreviewInput } from './preview-input'
import { registerPreviewScriptRunner } from './preview-script-runner'
import { abortPreviewTyping, releasePreviewTyping, trackPreviewTyping } from './preview-typing-abort'

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

const cleanups: Array<() => void> = []

afterEach(() => {
  for (const cleanup of cleanups.splice(0)) {
    cleanup()
  }

  closeRightRail()
})

it.each(['timeout', 'interrupted', 'local-stop'])(
  'keeps replayed typing cancellable after the superseded action settles (%s)',
  async reason => {
    const firstCharacter = deferred<void>()
    const replayCharacter = deferred<void>()
    const originalFinished = deferred<void>()
    const replayFinished = deferred<{ value: string }>()
    const characters: string[] = []
    openPreview({ kind: 'url', label: 'Browser', source: 'https://example.test', url: 'https://example.test' })
    const tabId = $rightRailActiveTabId.get()!
    cleanups.push(
      registerPreviewScriptRunner(tabId, async code => {
        if (code.includes('hermes-focus-probe')) {
          return JSON.stringify({ focused: true, success: true, tag: 'TEXTAREA' })
        }

        return code.includes('"kind":"locate"')
          ? JSON.stringify({
              acted: 'looking at textbox "Comment"',
              point: { x: 40, y: 20 },
              success: true,
              tag: 'TEXTAREA',
              typable: true
            })
          : JSON.stringify({ elements: [], hit: { tag: 'TEXTAREA', trusted: true }, success: true })
      })
    )
    cleanups.push(
      registerPreviewInput(tabId, {
        focus: () => undefined,
        send: event => {
          if (event.type === 'char') {
            characters.push(event.keyCode)

            if (characters.length === 1) {
              firstCharacter.resolve()
            }

            if (characters.length === 2) {
              replayCharacter.resolve()
            }
          }
        }
      })
    )
    let interrupted = false

    const deps = {
      activeSessionIdRef: { current: 'session-a' },
      sessionInterrupted: () => interrupted,
      updateSessionState: vi.fn(),
      upsertToolCall: vi.fn()
    } as ServerRequestContext['deps']

    const request = {
      id: 'srq-replayed-type',
      method: 'preview.act',
      params: { action: 'type', text: 'x'.repeat(100), ref: '@e1', session_id: 'session-a' },
      profile: 'default',
      fail: vi.fn(),
      respond: () => originalFinished.resolve()
    }

    handleServerRequest(request, deps, 'session-a')
    await firstCharacter.promise
    handleServerRequest(
      { ...request, replayed: true, respond: result => replayFinished.resolve(result as { value: string }) },
      deps,
      'session-a'
    )
    await originalFinished.promise
    // The replay goes through locate/focus before its first character, letting
    // the original loop and handler's finally finish while it is still running.
    await replayCharacter.promise
    const payload = { id: request.id, method: 'preview.act', reason }

    if (reason === 'local-stop') {
      interrupted = true
    } else {
      handleInputRequestEvent({
        deps,
        event: { type: 'request.cancel', session_id: 'session-a', payload },
        payload,
        sessionId: 'session-a'
      } as unknown as GatewayEventContext)
    }

    const result = JSON.parse((await replayFinished.promise).value)

    expect(result.success).toBe(false)
    expect(result.error).toMatch(reason === 'timeout' ? /timed out/ : /interrupted/)
    expect(characters.length).toBeLessThan(request.params.text.length)
  }
)

it('releases a normally completed action so late cancellation is inert', () => {
  const signal = trackPreviewTyping('srq-completed')
  releasePreviewTyping('srq-completed', signal)
  abortPreviewTyping('srq-completed', 'timeout')
  expect(signal.aborted).toBe(false)
})
