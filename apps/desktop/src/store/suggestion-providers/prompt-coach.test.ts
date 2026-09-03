import './prompt-coach'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $composerSuggestionsBySession, sampleComposerDraft } from '@/store/composer-suggestions'
import { $promptCoachPreviewBySession } from '@/store/prompt-coach'

describe('Prompt Coach draft provider', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $composerSuggestionsBySession.set({})
    $promptCoachPreviewBySession.set({})
  })

  afterEach(() => vi.useRealTimers())

  it('offers a session-scoped pill after the shared 600ms debounce and opens the preview on demand', async () => {
    sampleComposerDraft('prompt-coach-session', 'fix it and make it better')

    expect($composerSuggestionsBySession.get()['prompt-coach-session']).toBeUndefined()

    await vi.advanceTimersByTimeAsync(600)

    const suggestions = $composerSuggestionsBySession.get()['prompt-coach-session'] ?? []

    expect(suggestions.map(suggestion => suggestion.label)).toEqual(['Improve prompt'])
    expect($composerSuggestionsBySession.get().other).toBeUndefined()

    await suggestions[0]!.invoke({ cancelled: () => false, sessionId: 'prompt-coach-session' })

    expect($promptCoachPreviewBySession.get()['prompt-coach-session']?.original).toBe('fix it and make it better')
  })

  it('offers the pill on the landing composer before a session id exists', async () => {
    sampleComposerDraft(null, 'fix it and make it better')
    await vi.advanceTimersByTimeAsync(600)

    const landing = $composerSuggestionsBySession.get()['__new-session-draft__'] ?? []

    expect(landing.map(suggestion => suggestion.label)).toEqual(['Improve prompt'])
  })

  it('withdraws when the next sampled draft is already concrete', async () => {
    sampleComposerDraft('prompt-coach-withdraw', 'build this and make it better')
    await vi.advanceTimersByTimeAsync(600)
    expect($composerSuggestionsBySession.get()['prompt-coach-withdraw']).toHaveLength(1)

    sampleComposerDraft('prompt-coach-withdraw', 'fix typo in README.md')
    await vi.advanceTimersByTimeAsync(600)

    expect($composerSuggestionsBySession.get()['prompt-coach-withdraw']).toBeUndefined()
  })

  it('replaces an otherwise-identical pill with the latest draft closure', async () => {
    sampleComposerDraft('prompt-coach-latest', 'build this for mrs')
    await vi.advanceTimersByTimeAsync(600)

    sampleComposerDraft('prompt-coach-latest', 'build this for mars')
    await vi.advanceTimersByTimeAsync(600)

    const suggestion = $composerSuggestionsBySession.get()['prompt-coach-latest']?.[0]
    await suggestion?.invoke({ cancelled: () => false, sessionId: 'prompt-coach-latest' })

    expect($promptCoachPreviewBySession.get()['prompt-coach-latest']?.original).toBe('build this for mars')
  })
})
