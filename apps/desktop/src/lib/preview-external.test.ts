import { describe, expect, it } from 'vitest'

import { admitPreviewExternalUrl, PREVIEW_EXTERNAL_CHANNEL } from './preview-external'

describe('admitPreviewExternalUrl', () => {
  it('admits the web and mail schemes the guest handoff exists for', () => {
    expect(admitPreviewExternalUrl('https://www.google.com/search?q=traceback')).toBe(true)
    expect(admitPreviewExternalUrl('http://localhost:8501/')).toBe(true)
    expect(admitPreviewExternalUrl('mailto:support@example.com')).toBe(true)
  })

  it('rejects file: — a guest page must not reach the local-file opener', () => {
    expect(admitPreviewExternalUrl('file:///etc/passwd')).toBe(false)
    expect(admitPreviewExternalUrl('FILE:///Users/me/secret.txt')).toBe(false)
  })

  it('rejects script-capable and opaque schemes', () => {
    expect(admitPreviewExternalUrl('javascript:alert(1)')).toBe(false)
    expect(admitPreviewExternalUrl('data:text/html,hi')).toBe(false)
    expect(admitPreviewExternalUrl('blob:https://example.com/uuid')).toBe(false)
    expect(admitPreviewExternalUrl('chrome://settings')).toBe(false)
  })

  it('rejects unparseable and empty input', () => {
    expect(admitPreviewExternalUrl('')).toBe(false)
    expect(admitPreviewExternalUrl('   ')).toBe(false)
    expect(admitPreviewExternalUrl('not a url')).toBe(false)
    expect(admitPreviewExternalUrl('localhost:8501')).toBe(false)
  })

  it('uses the channel the guest preload sends on', () => {
    expect(PREVIEW_EXTERNAL_CHANNEL).toBe('preview-open-external')
  })
})
