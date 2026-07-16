import { afterEach, describe, expect, it } from 'vitest'

import {
  $visibleTranscriptSessionIds,
  registerTranscriptSurface,
  type TranscriptSurfaceRegistration
} from './thread-scroll'

const registrations: TranscriptSurfaceRegistration[] = []

const register = (sessionId: string, atBottom: boolean) => {
  const registration = registerTranscriptSurface(sessionId, atBottom)
  registrations.push(registration)

  return registration
}

afterEach(() => {
  registrations.splice(0).forEach(registration => registration.dispose())
})

describe('transcript surface visibility', () => {
  it('publishes a mounted session only while its transcript is at the bottom', () => {
    const surface = register('stored-session', false)

    expect($visibleTranscriptSessionIds.get()).toEqual([])

    surface.setAtBottom(true)
    expect($visibleTranscriptSessionIds.get()).toEqual(['stored-session'])

    surface.dispose()
    expect($visibleTranscriptSessionIds.get()).toEqual([])
  })

  it('keeps a session visible while any split surface is at the bottom', () => {
    const first = register('stored-session', false)
    const second = register('stored-session', true)

    expect($visibleTranscriptSessionIds.get()).toEqual(['stored-session'])

    second.dispose()
    expect($visibleTranscriptSessionIds.get()).toEqual([])

    first.setAtBottom(true)
    expect($visibleTranscriptSessionIds.get()).toEqual(['stored-session'])
  })
})
