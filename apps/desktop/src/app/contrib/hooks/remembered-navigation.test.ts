import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getRememberedRoute, setRememberedRoute, setRememberedSessionId } from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'

import { restoreRememberedNavigation } from './remembered-navigation'

const sessions = [makeSessionInfo({ id: 'saved', profile: 'coder' })]

const restore = (over: Partial<Parameters<typeof restoreRememberedNavigation>[0]> = {}) => {
  const onDestination = vi.fn()

  restoreRememberedNavigation({
    profile: 'coder',
    sessions,
    diskPluginsScanPending: false,
    resumeExhaustedSessionId: null,
    onDestination,
    ...over
  })

  return { onDestination }
}

beforeEach(() => window.localStorage.clear())

describe('shared remembered navigation ladder', () => {
  it('restores a remembered non-session route', () => {
    setRememberedRoute('/capabilities', 'coder')
    expect(restore().onDestination).toHaveBeenCalledWith('/capabilities', null)
  })

  it('ignores an overlay route and falls back to the remembered session', () => {
    setRememberedRoute('/settings', 'coder')
    setRememberedSessionId('saved', 'coder')
    expect(restore().onDestination).toHaveBeenCalledWith('/saved', 'saved')
  })

  it('restores a session belonging to the profile', () => {
    setRememberedRoute('/saved', 'coder')
    expect(restore().onDestination).toHaveBeenCalledWith('/saved', 'saved')
  })

  it('rejects a foreign-profile session route', () => {
    setRememberedRoute('/foreign', 'coder')
    const result = restore({ sessions: [makeSessionInfo({ id: 'foreign', profile: 'other' })] })
    expect(result.onDestination).toHaveBeenCalledWith(null, null)
    expect(getRememberedRoute('coder')).toBe('/foreign')
  })

  it('skips a resume-exhausted session', () => {
    setRememberedRoute('/saved', 'coder')
    setRememberedSessionId('saved', 'coder')
    expect(restore({ resumeExhaustedSessionId: 'saved' }).onDestination).toHaveBeenCalledWith(null, null)
  })

  it('falls back to a fresh draft when nothing is remembered', () => {
    expect(restore({ sessions: [] }).onDestination).toHaveBeenCalledWith(null, null)
  })
})
