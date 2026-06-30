import { describe, expect, it } from 'vitest'

import { localPreviewTarget } from './local-preview'

describe('localPreviewTarget file:// path resolution', () => {
  it('strips the leading slash from a Windows drive-letter file URL', () => {
    expect(localPreviewTarget('file:///C:/Users/x/index.html')?.path)
      .toBe('C:/Users/x/index.html')
  })

  it('preserves a POSIX absolute file URL path', () => {
    expect(localPreviewTarget('file:///home/u/a.html')?.path)
      .toBe('/home/u/a.html')
  })
})
