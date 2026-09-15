import { describe, expect, it } from 'vitest'

import {
  VOICE_SUPERVISOR_SERVER_REQUESTS,
  voiceSupervisorSurfaceEvent,
  voiceSupervisorSurfaceRequest
} from './voice-supervisor-events'

describe('voice supervisor gateway event classification', () => {
  it('classifies tool progress and ignores empty names', () => {
    expect(
      voiceSupervisorSurfaceEvent({
        payload: { name: 'delegate_task' },
        type: 'tool.start'
      })
    ).toEqual({ kind: 'narrate-tool', name: 'delegate_task' })
    expect(voiceSupervisorSurfaceEvent({ payload: { name: '' }, type: 'tool.start' })).toBeNull()
  })

  it('keeps every subscribed blocking request classified', () => {
    const notices = VOICE_SUPERVISOR_SERVER_REQUESTS.map(method => voiceSupervisorSurfaceRequest(method))

    expect(notices).toEqual([
      { kind: 'notify', text: 'Hermes needs your approval in the app.' },
      { kind: 'notify', text: 'Hermes needs your administrator password in the app.' },
      { kind: 'notify', text: 'Hermes needs a credential in the app.' },
      { kind: 'notify', text: 'Hermes has a question for you in the app.' }
    ])
    expect(voiceSupervisorSurfaceRequest('terminal.read')).toBeNull()
  })
})
