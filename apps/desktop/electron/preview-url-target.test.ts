import { describe, expect, it } from 'vitest'

import { previewHttpUrlTarget } from './preview-url-target'

describe('previewHttpUrlTarget', () => {
  it('navigates a public https page', () => {
    expect(previewHttpUrlTarget('https://www.uol.com.br')).toEqual({
      kind: 'url',
      label: 'www.uol.com.br',
      source: 'https://www.uol.com.br',
      url: 'https://www.uol.com.br/'
    })
  })

  it('keeps a loopback dev server and rewrites the 0.0.0.0 bind address', () => {
    expect(previewHttpUrlTarget('http://127.0.0.1:3000/app')?.url).toBe('http://127.0.0.1:3000/app')
    expect(previewHttpUrlTarget('http://0.0.0.0:8080/x')?.url).toBe('http://127.0.0.1:8080/x')
  })

  it('rejects a non-http target', () => {
    expect(previewHttpUrlTarget('file:///tmp/a.html')).toBeNull()
    expect(previewHttpUrlTarget('')).toBeNull()
  })
})
