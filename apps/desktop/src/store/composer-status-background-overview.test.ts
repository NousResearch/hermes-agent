import { describe, expect, it } from 'vitest'

import { allBackgroundProcesses, type ComposerStatusItem } from './composer-status'

describe('allBackgroundProcesses', () => {
  it('flattens background work across runtime sessions and preserves its owning session id', () => {
    const running: ComposerStatusItem = {
      id: 'proc-running',
      state: 'running',
      title: 'npm run dev',
      type: 'background'
    }

    const failed: ComposerStatusItem = {
      exitCode: 1,
      id: 'proc-failed',
      state: 'failed',
      title: 'npm test',
      type: 'background'
    }

    expect(
      allBackgroundProcesses({
        'runtime-a': [running],
        'runtime-b': [failed]
      })
    ).toEqual([
      { ...running, runtimeSessionId: 'runtime-a', sessionId: 'runtime-a' },
      { ...failed, runtimeSessionId: 'runtime-b', sessionId: 'runtime-b' }
    ])
  })
})
