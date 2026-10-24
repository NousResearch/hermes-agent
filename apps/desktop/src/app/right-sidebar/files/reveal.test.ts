import { describe, expect, it } from 'vitest'

import { ancestorPaths, openStateForReveal } from './reveal'

const ROOT = '/Users/me/project'

describe('ancestorPaths', () => {
  it('lists the folders between the root and the file, outermost first', () => {
    // Order is load-bearing: a lazily-loaded tree can only resolve `src/app`
    // once `src` has been expanded.
    expect(ancestorPaths(`${ROOT}/src/app/main.ts`, ROOT)).toEqual([`${ROOT}/src`, `${ROOT}/src/app`])
  })

  it('is empty for a file sitting directly in the root', () => {
    expect(ancestorPaths(`${ROOT}/README.md`, ROOT)).toEqual([])
  })

  it('is empty for a path outside the workspace', () => {
    // A file from another workspace has no row here; guessing one would scroll
    // the user to something unrelated.
    expect(ancestorPaths('/etc/hosts', ROOT)).toEqual([])
  })

  it('does not treat a sibling with the same prefix as inside the root', () => {
    expect(ancestorPaths('/Users/me/project-other/src/a.ts', ROOT)).toEqual([])
  })

  it('tolerates a trailing separator on the root', () => {
    expect(ancestorPaths(`${ROOT}/src/a.ts`, `${ROOT}/`)).toEqual([`${ROOT}/src`])
  })

  it('handles windows-style paths', () => {
    expect(ancestorPaths('C:\\work\\repo\\src\\a.ts', 'C:\\work\\repo')).toEqual([
      'C:\\work\\repo\\src'
    ])
  })
})

describe('openStateForReveal', () => {
  it('opens every folder on the way to the file', () => {
    const next = openStateForReveal(`${ROOT}/src/app/main.ts`, ROOT, {})
    expect(next[`${ROOT}/src`]).toBe(true)
    expect(next[`${ROOT}/src/app`]).toBe(true)
  })

  it('never closes a folder the user opened elsewhere', () => {
    // Revealing is an offer, not a reset: collapsing unrelated branches would
    // destroy the context the user built.
    const current = { [`${ROOT}/docs`]: true }
    const next = openStateForReveal(`${ROOT}/src/main.ts`, ROOT, current)
    expect(next[`${ROOT}/docs`]).toBe(true)
  })

  it('returns the same object when everything is already open', () => {
    // Reference identity keeps the tree from re-rendering for a file that is
    // already visible — the common case while clicking around.
    const current = { [`${ROOT}/src`]: true }
    expect(openStateForReveal(`${ROOT}/src/main.ts`, ROOT, current)).toBe(current)
  })

  it('returns the same object for a path outside the workspace', () => {
    const current = { [`${ROOT}/src`]: true }
    expect(openStateForReveal('/etc/hosts', ROOT, current)).toBe(current)
  })
})
