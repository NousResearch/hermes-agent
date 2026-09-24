import { describe, expect, it } from 'vitest'

import type { ClipboardFilePathsResult } from '../../../../electron/clipboard-files'

import { resolvePastedFileCandidates } from './clipboard-files'

const fakeFile = (name: string) => new File(['x'], name, { type: 'application/octet-stream' })

const fileEntry = (path: string) => ({ path, isDirectory: false })

describe('resolvePastedFileCandidates (#118181)', () => {
  it('returns the snapshot untouched when every entry already has a path', async () => {
    const snapshot = [{ path: 'C:/already/a.pdf', isDirectory: false, file: fakeFile('a.pdf') }]
    const readNative = () => Promise.reject(new Error('should not be called'))

    await expect(resolvePastedFileCandidates(snapshot, readNative)).resolves.toEqual(snapshot)
  })

  it('zips native paths with snapshot File handles when the count matches', async () => {
    const snapshot = [{ path: '', isDirectory: false, file: fakeFile('a.pdf') }, { path: '', isDirectory: false, file: fakeFile('b.txt') }]

    const readNative = (): Promise<ClipboardFilePathsResult> => Promise.resolve({
      status: 'files',
      files: [fileEntry('C:/native/a.pdf'), fileEntry('C:/native/b.txt')]
    })

    const result = await resolvePastedFileCandidates(snapshot, readNative)

    expect(result).toEqual([
      { path: 'C:/native/a.pdf', isDirectory: false, file: snapshot[0].file },
      { path: 'C:/native/b.txt', isDirectory: false, file: snapshot[1].file }
    ])
  })

  it('replaces the snapshot with the native list when counts disagree', async () => {
    const snapshot = [{ path: '', isDirectory: false, file: fakeFile('a.pdf') }]

    const readNative = (): Promise<ClipboardFilePathsResult> => Promise.resolve({
      status: 'files',
      files: [fileEntry('C:/native/a.pdf'), fileEntry('C:/native/b.txt')]
    })

    const result = await resolvePastedFileCandidates(snapshot, readNative)

    expect(result).toEqual([
      { path: 'C:/native/a.pdf', isDirectory: false },
      { path: 'C:/native/b.txt', isDirectory: false }
    ])
  })

  it('falls back to the snapshot when the native read is empty', async () => {
    const snapshot = [{ path: '', isDirectory: false, file: fakeFile('a.pdf') }]
    const readNative = (): Promise<ClipboardFilePathsResult> => Promise.resolve({ status: 'empty', files: [] })

    await expect(resolvePastedFileCandidates(snapshot, readNative)).resolves.toEqual(snapshot)
  })

  it('falls back to the snapshot when the native read reports unsupported or failed', async () => {
    const snapshot = [{ path: '', isDirectory: false, file: fakeFile('a.pdf') }]
    const unsupported: ClipboardFilePathsResult = { status: 'unsupported', files: [] }
    const failed: ClipboardFilePathsResult = { status: 'failed', files: [] }

    await expect(resolvePastedFileCandidates(snapshot, () => Promise.resolve(unsupported))).resolves.toEqual(snapshot)
    await expect(resolvePastedFileCandidates(snapshot, () => Promise.resolve(failed))).resolves.toEqual(snapshot)
  })

  it('falls back to the snapshot when the IPC bridge itself throws', async () => {
    const snapshot = [{ path: '', isDirectory: false, file: fakeFile('a.pdf') }]
    const readNative = () => Promise.reject(new Error('IPC unavailable'))

    await expect(resolvePastedFileCandidates(snapshot, readNative)).resolves.toEqual(snapshot)
  })
})