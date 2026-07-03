import { describe, expect, it } from 'vitest'

import { hardenHtmlForPreview, HTML_PREVIEW_CSP } from './html-preview'

describe('HTML preview hardening', () => {
  it('injects a restrictive CSP into an existing head', () => {
    const hardened = hardenHtmlForPreview('<!doctype html><html><head><title>x</title></head><body>ok</body></html>')

    expect(hardened).toContain(`http-equiv="Content-Security-Policy" content="${HTML_PREVIEW_CSP}"`)
    expect(hardened).toMatch(/<head><meta http-equiv="Content-Security-Policy"/)
    expect(hardened).toContain("script-src 'none'")
    expect(hardened).toContain("connect-src 'none'")
  })

  it('prepends a restrictive CSP when the document has no head', () => {
    const hardened = hardenHtmlForPreview('<body>ok</body>')

    expect(hardened).toMatch(/^<meta http-equiv="Content-Security-Policy"/)
    expect(hardened).toContain('<body>ok</body>')
  })
})
