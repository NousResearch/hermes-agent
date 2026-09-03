import { beforeEach, describe, expect, it } from 'vitest'

import type { PromptCoachAnalysis } from '@/lib/prompt-coach'

import {
  $promptCoachPreviewBySession,
  allowPromptCoachOriginal,
  closePromptCoachPreview,
  consumePromptCoachOriginal,
  openPromptCoachPreview,
  reconcilePromptCoachPreview,
  recordPromptCoachAction
} from './prompt-coach'

const analysis: PromptCoachAnalysis = {
  generatedBy: 'local',
  hasPotentialSecret: false,
  missing: ['target', 'success'],
  reason: 'Missing target and success criteria',
  score: 25,
  suggestedPrompt: 'Goal:\nfix it'
}

describe('Prompt Coach preview state', () => {
  beforeEach(() => {
    $promptCoachPreviewBySession.set({})
    window.localStorage.clear()
  })

  it('isolates previews by session and closes only stale drafts', () => {
    openPromptCoachPreview('a', 'fix it', analysis)
    openPromptCoachPreview('b', 'build it', analysis)

    reconcilePromptCoachPreview('a', 'fix it')
    expect($promptCoachPreviewBySession.get().a?.original).toBe('fix it')

    reconcilePromptCoachPreview('a', 'fix something else')
    expect($promptCoachPreviewBySession.get().a).toBeUndefined()
    expect($promptCoachPreviewBySession.get().b?.original).toBe('build it')

    closePromptCoachPreview('b')
    expect($promptCoachPreviewBySession.get()).toEqual({})
  })

  it('persists action counts without persisting prompt contents', () => {
    recordPromptCoachAction('dismissed')
    recordPromptCoachAction('dismissed')
    recordPromptCoachAction('replaced')

    const stored = window.localStorage.getItem('hermes.promptCoach.history.v1')

    expect(JSON.parse(stored ?? '{}')).toEqual({ dismissed: 2, replaced: 1 })
    expect(stored).not.toContain('fix it')
  })

  it('allows an unchanged original through exactly once', () => {
    allowPromptCoachOriginal('a', 'givme that')

    expect(consumePromptCoachOriginal('a', 'changed text')).toBe(false)
    expect(consumePromptCoachOriginal('a', 'givme that')).toBe(true)
    expect(consumePromptCoachOriginal('a', 'givme that')).toBe(false)
  })
})
