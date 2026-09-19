import path from 'node:path'

import { describe, expect, it } from 'vitest'

import {
  previewHomeFallbackCandidates,
  resolvePreviewHomeFallback
} from './preview-home-fallback'

const HOME = path.sep === '\\' ? 'C:\\Users\\suite' : '/home/suite'

describe('previewHomeFallbackCandidates', () => {
  it('retries a home-relative ref against home and the attachments dir', () => {
    expect(
      previewHomeFallbackCandidates('AppData/Local/hermes/attachments/foo.xlsx', HOME)
    ).toEqual([
      path.join(HOME, 'AppData/Local/hermes/attachments/foo.xlsx'),
      path.join(HOME, 'AppData', 'Local', 'hermes', 'attachments', 'foo.xlsx')
    ])
  })

  it('normalizes windows backslashes before joining', () => {
    expect(
      previewHomeFallbackCandidates('AppData\\Local\\hermes\\attachments\\foo.xlsx', HOME)
    ).toEqual([
      path.join(HOME, 'AppData/Local/hermes/attachments/foo.xlsx'),
      path.join(HOME, 'AppData', 'Local', 'hermes', 'attachments', 'foo.xlsx')
    ])
  })

  it('covers a bare basename that lost its directory prefix', () => {
    expect(previewHomeFallbackCandidates('foo.xlsx', HOME)).toEqual([
      path.join(HOME, 'foo.xlsx'),
      path.join(HOME, 'AppData', 'Local', 'hermes', 'attachments', 'foo.xlsx')
    ])
  })

  it('ignores absolute paths, file: urls and blanks', () => {
    expect(previewHomeFallbackCandidates(path.join(HOME, 'foo.xlsx'), HOME)).toEqual([])
    expect(previewHomeFallbackCandidates('file:///home/suite/foo.xlsx', HOME)).toEqual([])
    expect(previewHomeFallbackCandidates('', HOME)).toEqual([])
    expect(previewHomeFallbackCandidates('   ', HOME)).toEqual([])
    expect(previewHomeFallbackCandidates('foo.xlsx', '')).toEqual([])
  })
})

describe('resolvePreviewHomeFallback', () => {
  it('picks the home-joined candidate when it exists', () => {
    const homeHit = path.join(HOME, 'docs/report.xlsx')

    expect(
      resolvePreviewHomeFallback('docs/report.xlsx', {
        exists: candidate => candidate === homeHit,
        homeDir: HOME
      })
    ).toBe(homeHit)
  })

  it('falls through to the attachments basename candidate', () => {
    const attachmentHit = path.join(HOME, 'AppData', 'Local', 'hermes', 'attachments', 'foo.xlsx')

    expect(
      resolvePreviewHomeFallback('stale/prefix/foo.xlsx', {
        exists: candidate => candidate === attachmentHit,
        homeDir: HOME
      })
    ).toBe(attachmentHit)
  })

  it('returns null when neither candidate exists', () => {
    expect(
      resolvePreviewHomeFallback('docs/missing.xlsx', { exists: () => false, homeDir: HOME })
    ).toBeNull()
  })

  it('returns null for absolute, file: and blank targets', () => {
    const exists = () => true

    expect(
      resolvePreviewHomeFallback(path.join(HOME, 'foo.xlsx'), { exists, homeDir: HOME })
    ).toBeNull()
    expect(resolvePreviewHomeFallback('file:///home/suite/foo.xlsx', { exists, homeDir: HOME })).toBeNull()
    expect(resolvePreviewHomeFallback('', { exists, homeDir: HOME })).toBeNull()
  })
})
