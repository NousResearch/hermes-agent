import { afterEach, describe, expect, it } from 'vitest'

import type { ScopedServerRequest } from '@/store/gateway'

import { dispatchPluginServerRequest, onPluginServerRequest, pluginWantsServerRequest } from './server-request-tap'

/** A minimal scoped request whose answers are captured, like the tap's callers see one. */
interface CapturedRequest extends ScopedServerRequest {
  answered: { fail?: [code: number, message: string]; result?: Record<string, unknown> }
}

const makeRequest = (method: string): CapturedRequest => {
  const answered: { fail?: [code: number, message: string]; result?: Record<string, unknown> } = {}

  return {
    id: 'r1',
    method,
    params: { session_id: 's1' },
    profile: 'default',
    respond: (result: Record<string, unknown>) => {
      answered.result ??= result
    },
    fail: (code: number, message: string) => {
      answered.fail ??= [code, message]
    },
    answered
  }
}

const disposers: Array<() => void> = []

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
})

describe('the plugin server-request tap', () => {
  it('lets a plugin claim an otherwise-unhandled method and answer it', () => {
    const request = makeRequest('workflow')
    expect(pluginWantsServerRequest('workflow')).toBe(false)

    disposers.push(
      onPluginServerRequest('workflow', req => {
        req.respond({ value: '{"ok":true}' })

        return true
      })
    )

    expect(pluginWantsServerRequest('workflow')).toBe(true)
    expect(dispatchPluginServerRequest(request)).toBe(true)
    expect(request.answered.result).toEqual({ value: '{"ok":true}' })
  })

  it('answers an error when the claiming handler throws, and stops claiming after dispose', () => {
    const request = makeRequest('workflow')

    const dispose = onPluginServerRequest('workflow', () => {
      throw new Error('boom')
    })

    disposers.push(dispose)

    // The throw is answered to the backend (JSON-RPC internal error) instead of
    // surfacing, so the blocked tool returns now rather than at its deadline.
    expect(dispatchPluginServerRequest(request)).toBe(true)
    expect(request.answered.fail).toEqual([expect.any(Number), 'boom'])

    dispose()
    expect(pluginWantsServerRequest('workflow')).toBe(false)
    expect(dispatchPluginServerRequest(makeRequest('workflow'))).toBe(false)
  })
})
