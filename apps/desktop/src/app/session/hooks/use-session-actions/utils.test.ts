// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { $approvalModes, approvalModeForProfile } from '@/store/approval-mode'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getSession
}))

import {
  applyRuntimeInfo,
  chatMessageArraysEquivalent,
  isSessionGoneError,
  reconcileResumeMessages,
  resolveStoredSession,
  sessionMatchesStoredId,
  sessionShouldHaveTranscript,
  toBranchMessages
} from './utils'

const msg = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage =>
  ({ id, role, parts: [{ type: 'text', text }], ...extra }) as ChatMessage

const session = (over: Partial<SessionInfo>): SessionInfo => over as SessionInfo

describe('applyRuntimeInfo approval mode', () => {
  beforeEach(() => {
    $approvalModes.set({})
    $activeGatewayProfile.set('work')
  })

  it('reconciles session.info against the gateway profile', () => {
    applyRuntimeInfo({ approval_mode: 'smart', desktop_contract: 3 })

    expect(approvalModeForProfile('work')).toBe('smart')
    expect(approvalModeForProfile('default')).toBe('smart')
  })
})

describe('isSessionGoneError', () => {
  it('is true for 404 / session-not-found, false otherwise', () => {
    expect(isSessionGoneError(new Error('Request failed 404'))).toBe(true)
    expect(isSessionGoneError(new Error('Session not found'))).toBe(true)
    expect(isSessionGoneError(new Error('ECONNREFUSED'))).toBe(false)
    expect(isSessionGoneError(null)).toBe(false)
  })
})

describe('sessionMatchesStoredId', () => {
  it('matches on live id or lineage root', () => {
    expect(sessionMatchesStoredId(session({ id: 'a' }), 'a')).toBe(true)
    expect(sessionMatchesStoredId(session({ id: 'live', _lineage_root_id: 'root' }), 'root')).toBe(true)
    expect(sessionMatchesStoredId(session({ id: 'a' }), 'b')).toBe(false)
  })
})

describe('resolveStoredSession expected profile', () => {
  beforeEach(() => {
    getSession.mockReset()
    $profiles.set([])
    $sessions.set([])
  })

  it('chooses the cached row from the expected profile when stored ids collide', async () => {
    $sessions.set([
      session({ id: 'shared', profile: 'default', title: 'Default' }),
      session({ id: 'shared', profile: 'work', title: 'Work' })
    ])

    await expect(resolveStoredSession('shared', 'work')).resolves.toEqual(
      expect.objectContaining({ profile: 'work', title: 'Work' })
    )
    expect(getSession).not.toHaveBeenCalled()
  })

  it('queries only the expected profile when the row is not cached', async () => {
    $sessions.set([session({ id: 'shared', profile: 'default', title: 'Default' })])
    getSession.mockResolvedValue(session({ id: 'shared', profile: 'work', title: 'Work' }))

    await expect(resolveStoredSession('shared', 'work')).resolves.toEqual(expect.objectContaining({ profile: 'work' }))
    expect(getSession).toHaveBeenCalledOnce()
    expect(getSession).toHaveBeenCalledWith('shared', 'work')
    expect($sessions.get().map(row => row.profile).sort()).toEqual(['default', 'work'])
  })
})

describe('sessionShouldHaveTranscript', () => {
  it('is true only when the session has messages', () => {
    expect(sessionShouldHaveTranscript(session({ message_count: 3 }))).toBe(true)
    expect(sessionShouldHaveTranscript(session({ message_count: 0 }))).toBe(false)
    expect(sessionShouldHaveTranscript(undefined)).toBe(false)
  })
})

describe('toBranchMessages', () => {
  it('keeps only user/assistant turns that carry text', () => {
    const out = toBranchMessages([
      msg('u', 'user', 'hi'),
      msg('blank', 'assistant', '   '),
      msg('sys', 'system', 'ignored'),
      msg('a', 'assistant', 'hello')
    ])

    expect(out.map(b => b.source.id)).toEqual(['u', 'a'])
    expect(out[0]).toMatchObject({ content: 'hi', role: 'user' })
  })
})

describe('chatMessageArraysEquivalent', () => {
  it('compares length and per-message equivalence', () => {
    const a = [msg('1', 'user', 'x'), msg('2', 'assistant', 'y')]
    expect(chatMessageArraysEquivalent(a, [msg('1', 'user', 'x'), msg('2', 'assistant', 'y')])).toBe(true)
    expect(chatMessageArraysEquivalent(a, [msg('1', 'user', 'x')])).toBe(false)
    expect(chatMessageArraysEquivalent(a, [msg('1', 'user', 'x'), msg('2', 'assistant', 'changed')])).toBe(false)
  })
})

describe('reconcileResumeMessages', () => {
  it('returns next untouched when there is no previous transcript', () => {
    const next = [msg('1', 'user', 'hi')]
    expect(reconcileResumeMessages(next, [])).toBe(next)
  })

  it('re-grafts reasoning parts onto a matching assistant turn', () => {
    const next = [msg('a', 'assistant', 'answer')]

    const previous = [
      msg('a', 'assistant', 'answer', {
        parts: [
          { type: 'reasoning', text: 'thinking' },
          { type: 'text', text: 'answer' }
        ]
      } as Partial<ChatMessage>)
    ]

    const [out] = reconcileResumeMessages(next, previous)
    expect(out.parts.some(p => p.type === 'reasoning')).toBe(true)
  })
})
