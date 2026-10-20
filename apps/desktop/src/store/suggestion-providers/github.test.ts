import { describe, expect, it } from 'vitest'

import { githubHit } from './github'

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
