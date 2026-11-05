import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $sessions, _resetSessionOwnerHintsForTests, getSessionOwnerHint, setSessionOwnerHint } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { rememberedOwnerForResume } from './remembered-owner'

const REMOTE = { connectionId: 'ssh-proxmox', profile: 'bot' }

describe('rememberedOwnerForResume', () => {
  beforeEach(() => {
    _resetSessionOwnerHintsForTests({ storage: true })
    $sessions.set([])
  })

  afterEach(() => {
    _resetSessionOwnerHintsForTests({ storage: true })
    $sessions.set([])
  })

  it('keeps the hint of a session with no row (hidden Bot Chat)', () => {
    setSessionOwnerHint('stored-1', REMOTE)

    expect(rememberedOwnerForResume('stored-1')).toMatchObject(REMOTE)
    expect(getSessionOwnerHint('stored-1')).toMatchObject(REMOTE)
  })

  it('drops a hint an untagged row contradicts (#97809)', () => {
    setSessionOwnerHint('stored-1', REMOTE)
    $sessions.set([{ id: 'stored-1' } as SessionInfo])

    expect(rememberedOwnerForResume('stored-1')).toBeUndefined()
    expect(getSessionOwnerHint('stored-1')).toBeUndefined()
  })

  it.each([
    ['default', 'recruiter'],
    ['recruiter', 'default']
  ])('drops a %s hint when the same server row belongs to %s', (hintProfile, rowProfile) => {
    setSessionOwnerHint('copied-session', { connectionId: 'linux-host', profile: hintProfile })
    $sessions.set([
      { connection_id: 'linux-host', id: 'copied-session', profile: rowProfile } as SessionInfo
    ])

    expect(rememberedOwnerForResume('copied-session')).toBeUndefined()
    expect(getSessionOwnerHint('copied-session')).toBeUndefined()
  })

  it('keeps a hint when both the connection and profile match', () => {
    setSessionOwnerHint('stored-1', REMOTE)
    $sessions.set([
      { connection_id: REMOTE.connectionId, id: 'stored-1', profile: REMOTE.profile } as SessionInfo
    ])

    expect(rememberedOwnerForResume('stored-1')).toMatchObject(REMOTE)
    expect(getSessionOwnerHint('stored-1')).toMatchObject(REMOTE)
  })

  it('keeps the backend target of a matching desktop profile alias', () => {
    const alias = { ...REMOTE, targetProfile: 'default' }
    setSessionOwnerHint('stored-1', alias)
    $sessions.set([
      { connection_id: REMOTE.connectionId, id: 'stored-1', profile: REMOTE.profile } as SessionInfo
    ])

    expect(rememberedOwnerForResume('stored-1')).toMatchObject(alias)
  })

  it('drops a hint when the profile matches but the connection differs', () => {
    setSessionOwnerHint('stored-1', REMOTE)
    $sessions.set([
      { connection_id: 'another-host', id: 'stored-1', profile: REMOTE.profile } as SessionInfo
    ])

    expect(rememberedOwnerForResume('stored-1')).toBeUndefined()
    expect(getSessionOwnerHint('stored-1')).toBeUndefined()
  })
})
