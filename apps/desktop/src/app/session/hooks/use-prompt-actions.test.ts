import { describe, expect, it } from 'vitest'

import { buildImageAttachParams } from './use-prompt-actions'
import type { ComposerAttachment } from '@/store/composer'

function imageAttachment(overrides: Partial<ComposerAttachment> = {}): ComposerAttachment {
  return {
    id: 'img-1',
    kind: 'image',
    label: 'clip.png',
    path: 'C:\\Users\\eddy\\AppData\\Roaming\\Hermes\\composer-images\\clip.png',
    ...overrides
  }
}

describe('buildImageAttachParams', () => {
  it('includes image data_url so remote backends do not need the client filesystem path', () => {
    const params = buildImageAttachParams('session-1', imageAttachment({ dataUrl: 'data:image/png;base64,AAAA' }))

    expect(params).toEqual({
      session_id: 'session-1',
      path: 'C:\\Users\\eddy\\AppData\\Roaming\\Hermes\\composer-images\\clip.png',
      name: 'clip.png',
      data_url: 'data:image/png;base64,AAAA'
    })
  })

  it('falls back to path-only payloads when no data URL is available', () => {
    const params = buildImageAttachParams('session-2', imageAttachment({ dataUrl: undefined }))

    expect(params).toEqual({
      session_id: 'session-2',
      path: 'C:\\Users\\eddy\\AppData\\Roaming\\Hermes\\composer-images\\clip.png',
      name: 'clip.png'
    })
  })
})
