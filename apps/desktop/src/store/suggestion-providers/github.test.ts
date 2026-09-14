import { describe, expect, it, vi } from 'vitest'

import type { DraftProvider } from '@/store/composer-suggestions'

const mocks = vi.hoisted(() => ({
  getGhAuthStatus: vi.fn(),
  registerDraftProvider: vi.fn(),
  requestComposerFocus: vi.fn(),
  requestComposerInsert: vi.fn()
}))

vi.mock('@/hermes', () => ({ getGhAuthStatus: mocks.getGhAuthStatus }))
vi.mock('@/store/composer-suggestions', () => ({ registerDraftProvider: mocks.registerDraftProvider }))
vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerFocus: mocks.requestComposerFocus,
  requestComposerInsert: mocks.requestComposerInsert
}))

import { githubHit, invalidateGithubSuggestionIndex } from './github'

it('offers the consolidated GitHub skill for sign-in and inserts it only on invocation', async () => {
  invalidateGithubSuggestionIndex()
  mocks.getGhAuthStatus.mockResolvedValue({ authenticated: false })
  const provider = mocks.registerDraftProvider.mock.calls.find(([name]) => name === 'github')?.[1] as DraftProvider

  const suggestions = await provider({ sessionId: null, text: 'connect github please' })

  expect(suggestions).toHaveLength(1)
  expect(mocks.requestComposerInsert).not.toHaveBeenCalled()
  await suggestions[0].invoke({ cancelled: () => false, sessionId: null })
  expect(mocks.requestComposerInsert).toHaveBeenCalledExactlyOnceWith('/github ', { mode: 'prefix' })
  expect(mocks.requestComposerFocus).toHaveBeenCalledOnce()
})

describe('githubHit', () => {
  it('matches a completed whole-word github mention', () => {
    expect(githubHit('open a github issue for this')).toBe(true)
    expect(githubHit('check GitHub please')).toBe(true)
  })

  it('does not fire while the word is still under the caret', () => {
    expect(githubHit('let me check github')).toBe(false)
  })

  it('does not match inside other words', () => {
    expect(githubHit('mygithubby thing here')).toBe(false)
  })

  it('matches a pasted github.com link immediately', () => {
    expect(githubHit('review https://github.com/NousResearch/hermes-agent/pull/1')).toBe(true)
    expect(githubHit('see https://gist.github.com/foo/abc')).toBe(true)
  })

  it('does not match lookalike domains', () => {
    expect(githubHit('see https://notgithub.com/x more')).toBe(false)
    expect(githubHit('see https://github.com.evil.example/x more')).toBe(false)
  })
})
