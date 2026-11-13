import { describe, expect, it } from 'vitest'

import { htmlFromDataUrl } from './browser-guest-document'

describe('htmlFromDataUrl', () => {
  it('decodes percent-encoded, base64 and charset-labelled HTML like the browser would', () => {
    expect(htmlFromDataUrl('data:text/html,<p>caf%C3%A9 &amp; %3Cb%3E</p>')).toBe('<p>café &amp; <b></p>')
    expect(htmlFromDataUrl(`data:text/html;base64,${btoa('<h1>hi</h1>')}`)).toBe('<h1>hi</h1>')
    expect(htmlFromDataUrl('data:TEXT/HTML ; charset=ISO-8859-1,caf%E9')).toBe('café')
    // The fragment is not part of the document.
    expect(htmlFromDataUrl('data:text/html,<a>x</a>#section')).toBe('<a>x</a>')
  })

  it('returns null for anything that is not a data: HTML document', () => {
    for (const src of [
      'https://example.com/a.html',
      'data:,plain',
      'data:text/plain,<script>1</script>',
      'data:image/svg+xml,%3Csvg%2F%3E',
      'data:text/html',
      'data:text/html;base64,***',
      'not a url'
    ]) {
      expect(htmlFromDataUrl(src), src).toBeNull()
    }
  })
})
