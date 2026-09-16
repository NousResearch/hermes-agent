import { describe, expect, it, vi } from 'vitest'

import {
  routeVoiceSupervisorGatewayEvent,
  routeVoiceSupervisorServerRequest,
  subscribeVoiceSupervisorEvents
} from './voice-supervisor-events'

describe('voice supervisor gateway events', () => {
  it('routes tool progress only to the matching active session', () => {
    const listener = vi.fn()
    const off = subscribeVoiceSupervisorEvents('session-1', listener)

    routeVoiceSupervisorGatewayEvent({
      payload: { name: 'delegate_task' },
      session_id: 'session-2',
      type: 'tool.start'
    })
    routeVoiceSupervisorGatewayEvent({
      payload: { name: 'delegate_task' },
      session_id: 'session-1',
      type: 'tool.start'
    })

    expect(listener).toHaveBeenCalledOnce()
    expect(listener).toHaveBeenCalledWith({ kind: 'narrate-tool', name: 'delegate_task' })
    off()
  })

  it.each([
    ['approval', 'Hermes needs your approval in the app.'],
    ['sudo', 'Hermes needs your administrator password in the app.'],
    ['secret', 'Hermes needs a credential in the app.'],
    ['clarify', 'Hermes has a question for you in the app.']
  ] as const)('routes the %s server request as a spoken notice', (method, text) => {
    const listener = vi.fn()
    const off = subscribeVoiceSupervisorEvents('session-1', listener)

    routeVoiceSupervisorServerRequest({ method, params: { session_id: 'session-1' } })

    expect(listener).toHaveBeenCalledWith({ kind: 'notify', text })
    off()
  })

  it('does not re-announce a request re-delivered by a reconnect replay', () => {
    const listener = vi.fn()
    const off = subscribeVoiceSupervisorEvents('session-1', listener)

    routeVoiceSupervisorServerRequest({ method: 'approval', params: { session_id: 'session-1' }, replayed: true })
    routeVoiceSupervisorServerRequest({ method: 'approval', params: { session_id: 'session-2' } })

    expect(listener).not.toHaveBeenCalled()
    off()
  })
})
