import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReviewCommitStack } from '@/global'

import { $reviewOpen } from './review'
import {
  $reviewCommitStack,
  $reviewRestackPending,
  $reviewStackSelected,
  refreshCommitStack,
  requestRestack,
  restackPrompt
} from './review-stack'
import { $currentCwd } from './session'

const submit = vi.fn((_text: string, _opts: unknown) => true)
vi.mock('@/app/chat/composer/focus', () => ({ requestComposerSubmit: (t: string, o: unknown) => submit(t, o) }))
vi.mock('./coding-status', () => ({ refreshRepoStatus: vi.fn(), repoStatusForCwd: () => ({ get: () => null }) }))

const copy = {
  intro: 'Restack since {base}:',
  rules: 'RULES',
  commitLine: (short: string, subject: string) => `- ${short} ${subject}`,
  dirtyNote: 'DIRTY'
}

function stack(over: Partial<HermesReviewCommitStack> = {}): HermesReviewCommitStack {
  return {
    base: 'abcdef0123456789',
    commits: [
      { sha: 'a'.repeat(40), short: 'aaaaaaa', subject: 'first', added: 1, removed: 0, files: [] },
      { sha: 'b'.repeat(40), short: 'bbbbbbb', subject: 'second', added: 2, removed: 1, files: [] }
    ],
    ...over
  }
}

beforeEach(() => {
  submit.mockClear()
  submit.mockReturnValue(true)
  $reviewOpen.set(true)
  $currentCwd.set('/repo')
  $reviewCommitStack.set({ base: null, commits: [] })
  $reviewStackSelected.set(null)
  $reviewRestackPending.set(false)
})

describe('restackPrompt', () => {
  it('spells out the base and every commit, and only mentions the dirty tree when it is dirty', () => {
    const clean = restackPrompt(stack(), false, copy)!

    expect(clean).toContain('Restack since abcdef012345:')
    expect(clean).toContain('- aaaaaaa first')
    expect(clean).toContain('- bbbbbbb second')
    expect(clean).not.toContain('DIRTY')
    expect(clean.endsWith('RULES')).toBe(true)

    expect(restackPrompt(stack(), true, copy)).toContain('DIRTY')
  })

  it('is null with no trunk to measure from, or with no commits and a clean tree', () => {
    expect(restackPrompt(stack({ base: null }), true, copy)).toBeNull()
    expect(restackPrompt(stack({ commits: [] }), false, copy)).toBeNull()
    // A branch with no commits but a dirty tree still has something to commit.
    expect(restackPrompt(stack({ commits: [] }), true, copy)).not.toBeNull()
  })
})

describe('requestRestack', () => {
  it('sends the prompt to the scoped composer, marks pending, and drops the commit selection', () => {
    $reviewCommitStack.set(stack())
    $reviewStackSelected.set('b'.repeat(40))

    expect(requestRestack(copy, false)).toBe('sent')
    expect(submit).toHaveBeenCalledTimes(1)
    expect(submit.mock.calls[0][1]).toEqual({ target: 'main' })
    expect($reviewRestackPending.get()).toBe(true)
    expect($reviewStackSelected.get()).toBeNull()
  })

  it('reports nothing / unavailable without submitting or flipping pending', () => {
    expect(requestRestack(copy, false)).toBe('nothing')
    expect(submit).not.toHaveBeenCalled()

    $reviewCommitStack.set(stack())
    submit.mockReturnValue(false)
    expect(requestRestack(copy, false)).toBe('unavailable')
    expect($reviewRestackPending.get()).toBe(false)
  })
})

describe('refreshCommitStack', () => {
  it('reads the bridge and drops a selection whose sha was rewritten away', async () => {
    const commitStack = vi.fn(async () => stack({ commits: [stack().commits[0]] }))

    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = { git: { review: { commitStack } } }
    $reviewStackSelected.set('b'.repeat(40))

    await refreshCommitStack()

    expect(commitStack).toHaveBeenCalledWith('/repo')
    expect($reviewCommitStack.get().commits.map(c => c.subject)).toEqual(['first'])
    expect($reviewStackSelected.get()).toBeNull()

    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })
})
