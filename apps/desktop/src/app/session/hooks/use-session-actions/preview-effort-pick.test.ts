import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $currentServiceTier,
  markComposerEffortManual,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier
} from '@/store/session'

import { applyStoredSessionPreviewRuntimeInfo } from './utils'

// The cold-resume preview paints while `session.resume` is still in flight. The
// model/provider half already paints transiently (#125336); the effort/tier/
// fast half used the PERSISTING setters, so glancing at an old session wiped
// the composer's reasoning pick from localStorage until the next explicit
// pick or config reseed (#87036).
describe('applyStoredSessionPreviewRuntimeInfo keeps the effort/tier/fast picks', () => {
  beforeEach(() => {
    window.localStorage.clear()
    setCurrentModel('user-pick')
    setCurrentProvider('anthropic')
    setCurrentReasoningEffort('user-effort')
    markComposerEffortManual()
    setCurrentServiceTier('priority')
    setCurrentFastMode(true)
  })

  afterEach(() => {
    window.localStorage.clear()
    setCurrentReasoningEffort('')
    setCurrentServiceTier('')
    setCurrentFastMode(false)
  })

  it('clears the visible effort/tier/fast paint without persisting the clear', () => {
    applyStoredSessionPreviewRuntimeInfo({ cwd: '', model: 'claude-opus-5-5' }, 'session-next')

    // Visible paint happened (the model preview, plus the cleared dimensions)…
    expect($currentModel.get()).toBe('claude-opus-5-5')
    expect($currentProvider.get()).toBe('')
    expect($currentReasoningEffort.get()).toBe('')
    expect($currentServiceTier.get()).toBe('')
    expect($currentFastMode.get()).toBe(false)

    // …but nothing was persisted: the composer's sticky picks survive.
    expect(window.localStorage.getItem('hermes.desktop.composer.model')).toBe('user-pick')
    expect(window.localStorage.getItem('hermes.desktop.composer.provider')).toBe('anthropic')
    expect(window.localStorage.getItem('hermes.desktop.composer.reasoning-effort')).toBe('user-effort')
    expect(window.localStorage.getItem('hermes.desktop.composer.service-tier')).toBe('priority')
    expect(window.localStorage.getItem('hermes.desktop.composer.fast')).toBe('true')
  })
})
