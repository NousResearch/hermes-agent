import { beforeEach, describe, expect, it } from 'vitest'

import { $showIgnoredRoots, setShowIgnoredFiles, showsIgnoredFiles } from './prefs'

describe('files show-ignored preference', () => {
  beforeEach(() => {
    $showIgnoredRoots.set([])
  })

  it('defaults to filtering and only opts in the root it was told about', () => {
    expect(showsIgnoredFiles('/repo')).toBe(false)

    setShowIgnoredFiles('/repo', true)

    expect(showsIgnoredFiles('/repo')).toBe(true)
    expect(showsIgnoredFiles('/other')).toBe(false)
  })

  it('round-trips off again', () => {
    setShowIgnoredFiles('/repo', true)
    setShowIgnoredFiles('/repo', false)

    expect(showsIgnoredFiles('/repo')).toBe(false)
    expect($showIgnoredRoots.get()).toEqual([])
  })

  it('treats host spellings of one root as the same project', () => {
    setShowIgnoredFiles('C:\\Repo\\', true)

    expect(showsIgnoredFiles('c:/repo')).toBe(true)
  })

  it('does not distinguish a trailing separator on POSIX roots', () => {
    setShowIgnoredFiles('/repo/', true)

    expect(showsIgnoredFiles('/repo')).toBe(true)
  })

  it('keeps POSIX roots case-sensitive', () => {
    setShowIgnoredFiles('/Repo', true)

    expect(showsIgnoredFiles('/repo')).toBe(false)
  })

  it('ignores an empty root instead of storing a blank entry', () => {
    setShowIgnoredFiles('', true)

    expect(showsIgnoredFiles('')).toBe(false)
    expect($showIgnoredRoots.get()).toEqual([])
  })

  it('never records a root twice', () => {
    setShowIgnoredFiles('/repo', true)
    setShowIgnoredFiles('/repo/', true)

    expect($showIgnoredRoots.get()).toHaveLength(1)
  })
})
