import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearClarifyRequest, setClarifyRequest } from './clarify'
import {
  $petActionCenter,
  clearPetActionCenterActionStatus,
  selectPetActionCenterItem,
  setPetActionCenterActionStatus
} from './pet-action-center'
import { $profiles } from './profile'
import { clearAllPrompts, setApprovalRequest, setSecretRequest, setSudoRequest } from './prompts'
import { $sessions } from './session'

describe('pet action center projection', () => {
  beforeEach(() => {
    clearAllPrompts()
    clearClarifyRequest()
    $profiles.set([])
    $sessions.set([])
    clearPetActionCenterActionStatus()
    vi.useFakeTimers()
  })

  afterEach(() => {
    clearAllPrompts()
    clearClarifyRequest()
    $profiles.set([])
    $sessions.set([])
    clearPetActionCenterActionStatus()
    vi.useRealTimers()
  })

  it('projects every open request and keeps equal runtime ids distinct by normalized profile', () => {
    $profiles.set([
      {
        has_env: true,
        is_default: true,
        model: null,
        name: 'default',
        path: '/default',
        provider: null,
        skill_count: 0
      },
      { has_env: true, is_default: false, model: null, name: 'work', path: '/work', provider: null, skill_count: 0 }
    ])
    $sessions.set([
      { id: 'stored-default', profile: 'default', title: 'Default thread' },
      { id: 'stored-work', profile: 'work', title: 'Work thread' }
    ] as never)

    vi.setSystemTime(10)
    setApprovalRequest({
      command: 'npm test',
      description: 'Run tests',
      profile: 'default',
      sessionId: 'shared-runtime',
      storedSessionId: 'stored-default'
    })
    vi.setSystemTime(20)
    setClarifyRequest({
      choices: ['Yes', 'No'],
      profile: ' work ',
      question: 'Continue?',
      requestId: 'backend-clarify-id',
      sessionId: 'shared-runtime',
      storedSessionId: 'stored-work'
    })

    expect($petActionCenter.get().items).toEqual([
      expect.objectContaining({
        id: expect.any(String),
        kind: 'approval',
        profile: 'default',
        profileLabel: 'default',
        receivedAt: 10,
        sessionId: 'shared-runtime',
        sessionTitle: 'Default thread',
        storedSessionId: 'stored-default'
      }),
      expect.objectContaining({
        id: expect.any(String),
        kind: 'clarify',
        profile: 'work',
        profileLabel: 'work',
        receivedAt: 20,
        sessionId: 'shared-runtime',
        sessionTitle: 'Work thread',
        storedSessionId: 'stored-work'
      })
    ])
    expect(new Set($petActionCenter.get().items.map(item => item.id)).size).toBe(2)
  })

  it('signals secure-input attention without projecting sudo or secret items', () => {
    vi.setSystemTime(10)
    setSudoRequest({
      profile: 'default',
      requestId: 'sudo-backend-id',
      sessionId: 'sudo-runtime'
    })
    vi.setSystemTime(20)
    setSecretRequest({
      envVar: 'OPENAI_API_KEY',
      profile: 'work',
      prompt: 'Paste the credential value',
      requestId: 'secret-backend-id',
      sessionId: 'secret-runtime'
    })

    expect($petActionCenter.get()).toEqual(
      expect.objectContaining({
        actionableCount: 0,
        attentionCount: 2,
        blockingCount: 2,
        items: [],
        secureInputCount: 2,
        selectedItemId: null
      })
    )

    const serialized = JSON.stringify($petActionCenter.get())

    expect(serialized).not.toContain('OPENAI_API_KEY')
    expect(serialized).not.toContain('Paste the credential value')
    expect(serialized).not.toContain('sudo-backend-id')
    expect(serialized).not.toContain('secret-backend-id')
    expect(serialized).not.toContain('credential')
    expect(serialized).not.toContain('password')
  })

  it('derives safe approval and clarify actions from backend capabilities', () => {
    vi.setSystemTime(10)
    setApprovalRequest({
      allowPermanent: false,
      choices: ['once', 'always', 'deny'],
      command: 'rm build.tmp',
      description: 'Remove a generated file',
      profile: 'default',
      sessionId: 'approval-runtime',
      smartDenied: true
    })
    vi.setSystemTime(20)
    setClarifyRequest({
      choices: ['Keep', 'Replace'],
      profile: 'default',
      question: 'Which version?',
      requestId: 'clarify-backend-id',
      sessionId: 'clarify-runtime'
    })

    expect($petActionCenter.get().items).toEqual([
      expect.objectContaining({
        allowPermanent: false,
        allowedActions: ['approve-once', 'deny'],
        choices: ['once', 'always', 'deny'],
        command: 'rm build.tmp',
        description: 'Remove a generated file',
        kind: 'approval',
        smartDenied: true
      }),
      expect.objectContaining({
        allowedActions: ['clarify-respond', 'clarify-skip'],
        choices: ['Keep', 'Replace'],
        kind: 'clarify',
        question: 'Which version?'
      })
    ])
  })

  it('does not serialize untranslated fallbacks or expose an unverified cross-profile session link', () => {
    $sessions.set([{ id: 'stored-work', profile: 'work', title: 'Work thread' }] as never)
    setApprovalRequest({
      command: '',
      description: '',
      profile: 'default',
      sessionId: 'shared-runtime',
      storedSessionId: 'stored-work'
    })

    expect($petActionCenter.get().items).toEqual([
      expect.objectContaining({
        allowedActions: ['approve-once', 'approve-session', 'approve-always', 'deny'],
        detail: null,
        profile: 'default',
        sessionTitle: null,
        storedSessionId: null,
        summary: null
      })
    ])
  })

  it('sorts overlay-actionable requests first, then received time and stable id', () => {
    vi.setSystemTime(1)
    setSudoRequest({
      profile: 'default',
      requestId: 'sudo-backend-id',
      sessionId: 'secure-runtime'
    })
    vi.setSystemTime(20)
    setApprovalRequest({
      command: 'two',
      description: 'approval',
      profile: 'default',
      sessionId: 'approval-runtime'
    })
    vi.setSystemTime(10)
    setClarifyRequest({
      choices: null,
      profile: 'default',
      question: 'one',
      requestId: 'clarify-backend-id',
      sessionId: 'clarify-runtime'
    })

    expect($petActionCenter.get()).toEqual(
      expect.objectContaining({
        attentionCount: 3,
        blockingCount: 3,
        secureInputCount: 1
      })
    )
    expect($petActionCenter.get().items.map(item => item.kind)).toEqual(['clarify', 'approval'])
  })

  it('preserves item ids across unrelated session-label updates', () => {
    vi.setSystemTime(10)
    setApprovalRequest({
      command: 'npm test',
      description: 'Run tests',
      profile: 'default',
      sessionId: 'runtime'
    })
    const before = $petActionCenter.get().items[0]?.id

    $sessions.set([{ id: 'unrelated', profile: 'default', title: 'Unrelated thread' }] as never)

    expect($petActionCenter.get().items[0]?.id).toBe(before)
  })

  it('validates selection and falls back to the first remaining item', () => {
    setApprovalRequest({
      command: 'one',
      description: 'First',
      profile: 'default',
      sessionId: 'runtime-1'
    })
    setApprovalRequest({
      command: 'two',
      description: 'Second',
      profile: 'work',
      sessionId: 'runtime-2'
    })

    const [first, second] = $petActionCenter.get().items

    expect(first?.id).toBeTruthy()
    expect(second?.id).toBeTruthy()
    expect($petActionCenter.get().selectedItemId).toBe(first?.id)
    expect(selectPetActionCenterItem('missing')).toBe(false)
    expect($petActionCenter.get().selectedItemId).toBe(first?.id)
    expect(selectPetActionCenterItem(second!.id)).toBe(true)
    expect($petActionCenter.get().selectedItemId).toBe(second?.id)

    clearAllPrompts({ profile: 'work', sessionId: 'runtime-2' })

    expect($petActionCenter.get().selectedItemId).toBe(first?.id)
  })

  it('carries serializable i18n-neutral action feedback', () => {
    setPetActionCenterActionStatus({ status: 'error', itemId: 'item-1', errorCode: 'rpc-failed' })

    expect($petActionCenter.get().action).toEqual({
      errorCode: 'rpc-failed',
      itemId: 'item-1',
      status: 'error'
    })
    expect(JSON.parse(JSON.stringify($petActionCenter.get())).action).toEqual($petActionCenter.get().action)
  })
})
