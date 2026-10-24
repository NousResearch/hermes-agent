import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getCurrentReasoningEffortSource,
  markComposerEffortManual,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentReasoningEffortSource
} from '@/store/session'

vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: vi.fn() }))
vi.mock('@/store/profile', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    ensureGatewayAgent: vi.fn(),
    ensureGatewayProfile: vi.fn()
  }
})

import { desktopSessionCreateParams } from './session-create-request'

const route = { connectionId: 'cloud', profile: 'default' }

// session.create must ship the composer's reasoning_effort only when the
// user explicitly PICKED it (the effort's manual/default split, mirroring the
// model picker's COMPOSER_MODEL_SOURCE gate). A seeded level mirrors
// agent.reasoning_effort; shipping it as an override outranked config and
// per-model agent.reasoning_overrides (#87036).
describe('desktopSessionCreateParams reasoning_effort source gate (#87036)', () => {
  beforeEach(() => {
    window.localStorage.clear()
    setCurrentReasoningEffort('')
    setCurrentFastMode(false)
    setCurrentModel('')
    setCurrentProvider('')
    setCurrentModelSource('')
  })

  afterEach(() => {
    window.localStorage.clear()
    setCurrentReasoningEffort('')
    setCurrentReasoningEffortSource('')
    setCurrentFastMode(false)
    setCurrentModel('')
    setCurrentProvider('')
    setCurrentModelSource('')
  })

  it('ships an explicitly picked effort as a per-session override', async () => {
    setCurrentReasoningEffort('max')
    markComposerEffortManual()

    const params = await desktopSessionCreateParams('', route)

    expect(params).toMatchObject({ reasoning_effort: 'max' })
  })

  // The effort pick is independent of the model-override decision: a
  // default-sourced model is omitted while a manually picked effort rides.
  it('ships a manual effort even when the model selection is default-sourced', async () => {
    setCurrentModel('openai/gpt-5.6-sol')
    setCurrentProvider('openai-codex')
    setCurrentModelSource('default')
    setCurrentReasoningEffort('high')
    markComposerEffortManual()

    const params = await desktopSessionCreateParams('', route)

    expect(params).not.toHaveProperty('model')
    expect(params).not.toHaveProperty('provider')
    expect(params).toMatchObject({ reasoning_effort: 'high' })
  })

  it('omits a default-sourced (config-seeded) effort so the backend resolves it', async () => {
    setCurrentReasoningEffort('high')
    setCurrentReasoningEffortSource('default')

    const params = await desktopSessionCreateParams('', route)

    expect(params).not.toHaveProperty('reasoning_effort')
  })

  // Migration: users already hold a value in the sticky key from before the
  // source existed (e.g. one a session.info heartbeat persisted over their
  // pick). With no recorded manual source it must stop shipping immediately.
  it('omits a legacy sticky value that has no recorded pick source', async () => {
    setCurrentReasoningEffort('ultra')
    expect(getCurrentReasoningEffortSource()).toBe('')

    const params = await desktopSessionCreateParams('', route)

    expect(params).not.toHaveProperty('reasoning_effort')
  })

  it('keeps shipping the fast flag alongside the gated effort (unchanged semantics)', async () => {
    setCurrentReasoningEffort('high')
    setCurrentFastMode(true)

    const params = await desktopSessionCreateParams('', route)

    expect(params).not.toHaveProperty('reasoning_effort')
    expect(params).toMatchObject({ fast: true })
  })
})
