import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getGlobalModelInfo } from '@/hermes'
import {
  $activeSessionId,
  $currentModel,
  $currentModelExplicitlySet,
  $currentProvider,
  setCurrentModel,
  setCurrentProvider
} from '@/store/session'

import { syncProfileDefaultTick, useModelProfileSync } from './use-model-profile-sync'

vi.mock('@/hermes', () => ({
  getGlobalModelInfo: vi.fn()
}))

describe('syncProfileDefaultTick', () => {
  beforeEach(() => {
    $activeSessionId.set(null)
    $currentModelExplicitlySet.set(false)
    setCurrentModel('')
    setCurrentProvider('')
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $activeSessionId.set(null)
    $currentModelExplicitlySet.set(false)
    setCurrentModel('')
    setCurrentProvider('')
  })

  it('takes a baseline on the first call and does not write to the composer', async () => {
    setCurrentModel('') // empty — first-run seed path is refreshCurrentModel, not us
    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'openai/gpt-5.5',
      provider: 'openai-codex'
    })

    const next = await syncProfileDefaultTick({ model: '', provider: '' })

    expect(next).toEqual({ model: 'openai/gpt-5.5', provider: 'openai-codex' })
    // First-run seed is NOT this hook's job — empty stays empty.
    expect($currentModel.get()).toBe('')
  })

  it('re-seeds the composer when the server default drifts and the composer was following the previous default', async () => {
    // Simulate post-boot state: composer holding the previously-seen default.
    setCurrentModel('openai/gpt-5.5')
    setCurrentProvider('openai-codex')

    // Server changes (Dashboard Models page, `hermes model`, `hermes config set`).
    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'anthropic/claude-sonnet-4.7',
      provider: 'anthropic'
    })

    const next = await syncProfileDefaultTick({ model: 'openai/gpt-5.5', provider: 'openai-codex' })

    expect(next).toEqual({ model: 'anthropic/claude-sonnet-4.7', provider: 'anthropic' })
    expect($currentModel.get()).toBe('anthropic/claude-sonnet-4.7')
    expect($currentProvider.get()).toBe('anthropic')
  })

  it('never overwrites an explicit user pick, even after the server default drifts', async () => {
    // Composer was auto-seeded, then the user explicitly picked something else.
    setCurrentModel('deepseek/deepseek-v4-pro')
    setCurrentProvider('deepseek')

    // Server's default also changes (independent event).
    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'anthropic/claude-sonnet-4.7',
      provider: 'anthropic'
    })

    const next = await syncProfileDefaultTick({ model: 'openai/gpt-5.5', provider: 'openai-codex' })

    // The pick survives — that's the test contract in use-model-controls.test.tsx.
    expect($currentModel.get()).toBe('deepseek/deepseek-v4-pro')
    expect($currentProvider.get()).toBe('deepseek')
    // Baseline still advances so a future composer-following default would
    // compare against the new server value.
    expect(next).toEqual({ model: 'anthropic/claude-sonnet-4.7', provider: 'anthropic' })
  })

  it('does nothing while a live session is active (footer owns the model label)', async () => {
    setCurrentModel('openai/gpt-5.5')
    setCurrentProvider('openai-codex')

    // A session starts — composer is now driven by the active-session footer.
    $activeSessionId.set('runtime-1')

    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'anthropic/claude-sonnet-4.7',
      provider: 'anthropic'
    })

    const next = await syncProfileDefaultTick({ model: 'openai/gpt-5.5', provider: 'openai-codex' })

    expect($currentModel.get()).toBe('openai/gpt-5.5')
    expect($currentProvider.get()).toBe('openai-codex')
    // Baseline unchanged — we'll re-evaluate when the session ends.
    expect(next).toEqual({ model: 'openai/gpt-5.5', provider: 'openai-codex' })
  })

  it('keeps the previous baseline when the backend call fails', async () => {
    setCurrentModel('openai/gpt-5.5')
    setCurrentProvider('openai-codex')

    vi.mocked(getGlobalModelInfo).mockRejectedValue(new Error('network'))

    const next = await syncProfileDefaultTick({ model: 'openai/gpt-5.5', provider: 'openai-codex' })

    expect(next).toEqual({ model: 'openai/gpt-5.5', provider: 'openai-codex' })
    expect($currentModel.get()).toBe('openai/gpt-5.5')
  })

  it('skips writes when the server value did not change', async () => {
    setCurrentModel('openai/gpt-5.5')
    setCurrentProvider('openai-codex')

    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'openai/gpt-5.5',
      provider: 'openai-codex'
    })

    const next = await syncProfileDefaultTick({ model: 'openai/gpt-5.5', provider: 'openai-codex' })

    expect(next).toEqual({ model: 'openai/gpt-5.5', provider: 'openai-codex' })
    expect($currentModel.get()).toBe('openai/gpt-5.5')
  })

  it('records an empty server value as the new baseline without clobbering the composer', async () => {
    // Composer holding a previous default; server has no default configured.
    setCurrentModel('openai/gpt-5.5')
    setCurrentProvider('openai-codex')

    vi.mocked(getGlobalModelInfo).mockResolvedValue({ model: '', provider: '' })

    const next = await syncProfileDefaultTick({ model: 'openai/gpt-5.5', provider: 'openai-codex' })

    expect(next).toEqual({ model: '', provider: '' })
    // Composer untouched — empty server value is treated as "nothing to seed".
    expect($currentModel.get()).toBe('openai/gpt-5.5')
    expect($currentProvider.get()).toBe('openai-codex')
  })

  it('does not overwrite a picker selection equal to the previous default', async () => {
    // Simulate the edge case from the maintainer review: user explicitly picked
    // the same model as the current default (e.g. opened the picker, selected
    // the default entry). selectModel calls setCurrentModel + sets the flag.
    setCurrentModel('openai/gpt-5.5')
    setCurrentProvider('openai-codex')
    $currentModelExplicitlySet.set(true)

    // External change (Dashboard, `hermes model`, another client on the same profile).
    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'anthropic/claude-sonnet-4.7',
      provider: 'anthropic'
    })

    const next = await syncProfileDefaultTick({ model: 'openai/gpt-5.5', provider: 'openai-codex' })

    // The picker selection survives — provenance overrides the value comparison.
    expect($currentModel.get()).toBe('openai/gpt-5.5')
    expect($currentProvider.get()).toBe('openai-codex')
    // Baseline advances so a future tick (after the user clears the pick, or the
    // profile is swapped) compares against the current server value.
    expect(next).toEqual({ model: 'anthropic/claude-sonnet-4.7', provider: 'anthropic' })
  })

  it('writes an empty provider from the server when the model is non-empty', async () => {
    // Composer following the previous baseline (provider populated).
    setCurrentModel('openai/gpt-5.5')
    setCurrentProvider('openai-codex')

    // Server drifts — new model with an empty provider (e.g. a model that
    // doesn't need an explicit provider override).
    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'anthropic/claude-sonnet-4.7',
      provider: ''
    })

    const next = await syncProfileDefaultTick({ model: 'openai/gpt-5.5', provider: 'openai-codex' })

    // Both model AND provider are written — matches refreshCurrentModel's
    // unconditional write (use-model-controls.ts:69). The empty provider is
    // the server's answer; leaving the stale provider produces a mismatch.
    expect($currentModel.get()).toBe('anthropic/claude-sonnet-4.7')
    expect($currentProvider.get()).toBe('')
    expect(next).toEqual({ model: 'anthropic/claude-sonnet-4.7', provider: '' })
  })
})

describe('useModelProfileSync', () => {
  beforeEach(() => {
    $activeSessionId.set(null)
    setCurrentModel('')
    setCurrentProvider('')
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $activeSessionId.set(null)
    setCurrentModel('')
    setCurrentProvider('')
  })

  it('mounts without crashing in a real component (smoke)', () => {
    function Harness() {
      useModelProfileSync({ gatewayOpen: true })

      return null
    }

    const { unmount } = render(<Harness />)
    expect(() => unmount()).not.toThrow()
  })
})
