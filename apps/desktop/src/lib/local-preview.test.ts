import { describe, expect, it } from 'vitest'

import { localPreviewTarget } from './local-preview'

describe('localPreviewTarget', () => {
  it('resolves relative artifact paths against cwd', () => {
    expect(localPreviewTarget('./dist/index.html', '/work/app')).toMatchObject({
      path: '/work/app/dist/index.html',
      previewKind: 'html',
      url: 'file:///work/app/dist/index.html'
    })
  })

  it('classifies PDFs as binary file previews', () => {
    expect(localPreviewTarget('/tmp/report.pdf')).toMatchObject({
      binary: true,
      path: '/tmp/report.pdf',
      previewKind: 'binary'
    })
  })

  it('keeps Windows absolute paths absolute', () => {
    expect(localPreviewTarget('C:\\Users\\me\\Documents\\report.pdf', '/work')).toMatchObject({
      path: 'C:\\Users\\me\\Documents\\report.pdf',
      url: 'file:///C:/Users/me/Documents/report.pdf'
    })
  })
})
