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

  it('navigates a plain-http public page and a LAN host', () => {
    expect(previewHttpUrlTarget('http://example.com/docs')?.url).toBe('http://example.com/docs')
    expect(previewHttpUrlTarget('http://homeassistant.local:8123')?.url).toBe('http://homeassistant.local:8123/')
  })

  it('rejects every non-http scheme the pane must never navigate to', () => {
    expect(previewHttpUrlTarget('file:///tmp/a.html')).toBeNull()
    expect(previewHttpUrlTarget('data:text/html,<script>alert(1)</script>')).toBeNull()
    expect(previewHttpUrlTarget('javascript:alert(1)')).toBeNull()
    expect(previewHttpUrlTarget('not a url')).toBeNull()
    expect(previewHttpUrlTarget('')).toBeNull()
  })
})
