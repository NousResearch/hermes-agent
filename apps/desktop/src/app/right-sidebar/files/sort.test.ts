import { describe, expect, it } from 'vitest'

import {
  compareFileEntries,
  DEFAULT_FILE_SORT,
  fileSortKey,
  isMediaFolderPath,
  MEDIA_FOLDER_FILE_SORT,
  parseFileSortKey,
  resolveFolderSort,
  sortFileEntries
} from './sort'

describe('parseFileSortKey / fileSortKey', () => {
  it('round-trips valid keys', () => {
    expect(parseFileSortKey('mtime:desc')).toEqual(MEDIA_FOLDER_FILE_SORT)
    expect(fileSortKey(DEFAULT_FILE_SORT)).toBe('name:asc')
    expect(parseFileSortKey('size:asc')).toEqual({ direction: 'asc', field: 'size' })
  })

  it('rejects garbage', () => {
    expect(parseFileSortKey(null)).toBeNull()
    expect(parseFileSortKey('')).toBeNull()
    expect(parseFileSortKey('date:desc')).toBeNull()
    expect(parseFileSortKey('name:up')).toBeNull()
  })
})

describe('isMediaFolderPath', () => {
  it('matches known generated-media basenames', () => {
    expect(isMediaFolderPath('/home/u/.hermes/renders')).toBe(true)
    expect(isMediaFolderPath('C:\\Users\\x\\Pictures\\screenshots')).toBe(true)
    expect(isMediaFolderPath('/tmp/cache/audio')).toBe(true)
    expect(isMediaFolderPath('/work/voice-memos')).toBe(true)
    expect(isMediaFolderPath('/proj/hf-renders')).toBe(true)
  })

  it('leaves ordinary source folders alone', () => {
    expect(isMediaFolderPath('/home/u/src')).toBe(false)
    expect(isMediaFolderPath('/home/u/project')).toBe(false)
    expect(isMediaFolderPath('')).toBe(false)
  })
})

describe('resolveFolderSort', () => {
  it('uses explicit per-folder pref over the media heuristic', () => {
    const path = '/home/u/renders'
    expect(resolveFolderSort(path, { [path]: 'name:asc' })).toEqual(DEFAULT_FILE_SORT)
  })

  it('defaults media folders to mtime desc and others to name asc', () => {
    expect(resolveFolderSort('/ws/renders', {})).toEqual(MEDIA_FOLDER_FILE_SORT)
    expect(resolveFolderSort('/ws/src', {})).toEqual(DEFAULT_FILE_SORT)
  })
})

describe('sortFileEntries', () => {
  const mixed = [
    { isDirectory: false, mtimeMs: 100, name: 'old.png', size: 10 },
    { isDirectory: true, mtimeMs: 50, name: 'z-dir', size: 0 },
    { isDirectory: false, mtimeMs: 300, name: 'new.png', size: 30 },
    { isDirectory: true, mtimeMs: 200, name: 'a-dir', size: 0 },
    { isDirectory: false, mtimeMs: 200, name: 'mid.png', size: 20 }
  ]

  it('keeps directories first, then name ascending (today’s order)', () => {
    const names = sortFileEntries(mixed, DEFAULT_FILE_SORT).map(e => e.name)
    expect(names).toEqual(['a-dir', 'z-dir', 'mid.png', 'new.png', 'old.png'])
  })

  it('sorts files newest-first while directories stay on top', () => {
    const names = sortFileEntries(mixed, MEDIA_FOLDER_FILE_SORT).map(e => e.name)
    // dirs: a-dir mtime 200, z-dir mtime 50 → newest dir first within dirs
    expect(names).toEqual(['a-dir', 'z-dir', 'new.png', 'mid.png', 'old.png'])
  })

  it('sorts by size descending', () => {
    const names = sortFileEntries(mixed, { direction: 'desc', field: 'size' }).map(e => e.name)
    expect(names).toEqual(['a-dir', 'z-dir', 'new.png', 'mid.png', 'old.png'])
  })

  it('treats missing mtime as oldest so dated entries still lead', () => {
    const entries = [
      { isDirectory: false, name: 'undated.bin' },
      { isDirectory: false, mtimeMs: 5, name: 'dated.bin' }
    ]

    expect(sortFileEntries(entries, MEDIA_FOLDER_FILE_SORT).map(e => e.name)).toEqual([
      'dated.bin',
      'undated.bin'
    ])
  })

  it('compareFileEntries is antisymmetric on equal names', () => {
    const a = { isDirectory: false, mtimeMs: 1, name: 'same', size: 1 }
    const b = { isDirectory: false, mtimeMs: 1, name: 'same', size: 1 }
    expect(compareFileEntries(a, b, DEFAULT_FILE_SORT)).toBe(0)
  })
})
