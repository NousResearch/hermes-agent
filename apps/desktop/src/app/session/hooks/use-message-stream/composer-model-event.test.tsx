import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $currentModel,
  $currentProvider,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider
} from '@/store/session'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

let stream: MessageStreamHarness

function mountStream(activeSessionId: string | null) {
  stream = renderMessageStream(activeSessionId)
}

describe('session.info does not clobber composer model selection', () => {
  beforeEach(() => {
    setCurrentModel('deepseek-v4-flash')
    setCurrentProvider('deepseek')
    setCurrentModelSource('manual')
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    setCurrentModel('')
    setCurrentProvider('')
    setCurrentModelSource('')
  })

  it('keeps a sticky manual pick when a global session.info carries the profile default', () => {
    mountStream(null)

    act(() =>
      stream.handleEvent({
        payload: { model: 'deepseek-chat', provider: 'deepseek' },
        type: 'session.info'
      })
    )

    expect($currentModel.get()).toBe('deepseek-v4-flash')
    expect($currentProvider.get()).toBe('deepseek')
  })

  it('keeps the composer pick when an unscoped session.info arrives with no live session', () => {
    mountStream(null)

    act(() =>
      stream.handleEvent({
        payload: { cwd: '/tmp/project', model: 'deepseek-chat', provider: 'deepseek' },
        type: 'session.info'
      })
    )

    expect($currentModel.get()).toBe('deepseek-v4-flash')
    expect($currentProvider.get()).toBe('deepseek')
  })
})
