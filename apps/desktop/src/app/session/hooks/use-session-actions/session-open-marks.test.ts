import { afterEach, describe, expect, it, vi } from 'vitest'

import { markSessionOpen, SESSION_OPEN_MARKS } from './session-open-marks'

describe('session open performance marks', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('emits only the fixed mark name without a detail payload', () => {
    const mark = vi.spyOn(performance, 'mark')

    markSessionOpen('hermes.session.rest.commit')

    expect(mark).toHaveBeenCalledExactlyOnceWith('hermes.session.rest.commit')
    expect(SESSION_OPEN_MARKS).toEqual([
      'hermes.session.select',
      'hermes.session.cache.commit',
      'hermes.session.rest.commit',
      'hermes.session.resume.ready',
      'hermes.session.agent.ready',
      'hermes.session.history.ready'
    ])
  })
})
