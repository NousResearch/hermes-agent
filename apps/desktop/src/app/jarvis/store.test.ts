import { describe, expect, it } from 'vitest'

import { $jarvisUi, publishJarvisEvent, resetJarvisSession } from './store'

describe('jarvis ui store', () => {
  it('resets gateway-bound state when the session changes', () => {
    publishJarvisEvent({ type: 'task.started', sessionId: 's1', taskId: 't1', at: 1 })

    resetJarvisSession('s2')

    expect($jarvisUi.get()).toMatchObject({
      sessionId: 's2',
      voice: 'idle',
      task: { id: null, phase: 'idle' },
    })
  })
})
