import { beforeEach, describe, expect, it } from 'vitest'

import {
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $currentReasoningEffortWire,
  setCurrentModel,
  setCurrentModelTransient,
  setCurrentProvider,
  setCurrentProviderTransient,
  setCurrentReasoningEffort,
  setCurrentReasoningEffortWire
} from './session'

/** The gateway's clamp for one (provider, model, effort) triple. */
const stampWire = (wire: string) => setCurrentReasoningEffortWire(wire)

describe('reasoning-effort wire scope', () => {
  beforeEach(() => {
    setCurrentModel('')
    setCurrentProvider('')
    setCurrentReasoningEffort('')
    stampWire('')
  })

  it('drops the wire stamp when the model changes, so a stale arrow is not presented as confirmed', () => {
    // The old route clamped `xhigh` up for its own vocabulary; the pill read
    // "XHigh→Max" even though no route can send that pair.
    setCurrentModel('gpt-6.1-sol')
    setCurrentProvider('openai-codex')
    setCurrentReasoningEffort('xhigh')
    stampWire('max')

    expect($currentReasoningEffortWire.get()).toBe('max')

    // A different model clamps a different set, so the stamp no longer describes
    // this session's route.
    setCurrentModel('gpt-6.1-luna')

    expect($currentReasoningEffortWire.get()).toBe('')
    // The pick itself is untouched — only the route claim is withdrawn.
    expect($currentReasoningEffort.get()).toBe('xhigh')
  })

  it('drops the wire stamp when the provider changes', () => {
    setCurrentModel('gpt-6.1-sol')
    setCurrentProvider('openai-codex')
    setCurrentReasoningEffort('xhigh')
    stampWire('max')

    setCurrentProvider('openrouter')

    expect($currentReasoningEffortWire.get()).toBe('')
  })

  it('keeps a confirmed stamp when the same model or provider is re-set', () => {
    setCurrentModel('gpt-6.1-sol')
    setCurrentProvider('openai-codex')
    setCurrentReasoningEffort('ultra')
    stampWire('max')

    // Re-stamping the identical selection describes the same route.
    setCurrentModel('gpt-6.1-sol')
    setCurrentProvider('openai-codex')

    expect($currentReasoningEffortWire.get()).toBe('max')
  })

  it('accepts a functional updater without clearing the stamp on a no-op', () => {
    setCurrentModel('gpt-6.1-sol')
    setCurrentReasoningEffort('ultra')
    stampWire('max')

    setCurrentModel(current => current)

    expect($currentModel.get()).toBe('gpt-6.1-sol')
    expect($currentReasoningEffortWire.get()).toBe('max')

    setCurrentModel(() => 'gpt-6.1-luna')

    expect($currentModel.get()).toBe('gpt-6.1-luna')
    expect($currentReasoningEffortWire.get()).toBe('')
  })

  it('keeps the stamp across a heartbeat that mirrors the runtime model', () => {
    setCurrentModel('gpt-6.1-sol')
    setCurrentProvider('openai-codex')
    setCurrentReasoningEffort('ultra')
    stampWire('max')

    // The periodic `session.info` heartbeat pushes the resolved route without
    // claiming the composer selection. It does not change the route, so it must
    // not withdraw a confirmed clamp — otherwise the pill flickers on every beat.
    setCurrentModelTransient($currentModel.get())
    setCurrentProviderTransient($currentProvider.get())

    expect($currentReasoningEffortWire.get()).toBe('max')
  })
})
