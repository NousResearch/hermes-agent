import { describe, expect, it } from 'vitest'

import { readFilePreviewDataUrl } from './file-preview'

function fsError(code: string): Error & { code: string } {
  return Object.assign(new Error(code), { code })
}

describe('readFilePreviewDataUrl', () => {
  it('returns the preview data URL on success', async () => {
    await expect(
      readFilePreviewDataUrl(async () => 'data:image/png;base64,VEVTVA==')
    ).resolves.toBe('data:image/png;base64,VEVTVA==')
  })

  it('returns an empty result when the preview file no longer exists', async () => {
    await expect(
      readFilePreviewDataUrl(async () => {
        throw fsError('ENOENT')
      })
    ).resolves.toBe('')
  })

  it('returns an empty result when a preview path component no longer exists', async () => {
    await expect(
      readFilePreviewDataUrl(async () => {
        throw fsError('ENOTDIR')
      })
    ).resolves.toBe('')
  })

  it('keeps permission errors strict', async () => {
    const error = fsError('EACCES')

    await expect(
      readFilePreviewDataUrl(async () => {
        throw error
      })
    ).rejects.toBe(error)
  })

  it('keeps non-filesystem errors strict', async () => {
    const error = new Error('Sensitive paths are not allowed')

    await expect(
      readFilePreviewDataUrl(async () => {
        throw error
      })
    ).rejects.toBe(error)
  })
})