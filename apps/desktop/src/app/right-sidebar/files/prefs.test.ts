import { beforeEach, describe, expect, it } from 'vitest'

import {
  $folderSortPrefs,
  $showIgnoredRoots,
  folderSortPreference,
  setFolderSortPreference,
  setShowIgnoredFiles,
  showsIgnoredFiles
} from './prefs'
import { DEFAULT_FILE_SORT, MEDIA_FOLDER_FILE_SORT } from './sort'

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

describe('files folder-sort preference', () => {
  beforeEach(() => {
    $folderSortPrefs.set({})
  })

  it('defaults ordinary folders to name ascending and media folders to newest-first', () => {
    expect(folderSortPreference('/ws/src')).toEqual(DEFAULT_FILE_SORT)
    expect(folderSortPreference('/ws/renders')).toEqual(MEDIA_FOLDER_FILE_SORT)
  })

  it('remembers an explicit choice per folder across path spellings', () => {
    expect(setFolderSortPreference('C:\\Work\\renders\\', { direction: 'asc', field: 'name' })).toBe(true)
    expect(folderSortPreference('c:/work/renders')).toEqual(DEFAULT_FILE_SORT)
    expect(setFolderSortPreference('c:/work/renders', { direction: 'asc', field: 'name' })).toBe(false)
  })

  it('ignores an empty folder path', () => {
    expect(setFolderSortPreference('', MEDIA_FOLDER_FILE_SORT)).toBe(false)
    expect($folderSortPrefs.get()).toEqual({})
  })
})
