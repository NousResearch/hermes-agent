import { describe, expect, it } from 'vitest'

import { admitPreviewExternalUrl, routePreviewGuestLink } from './preview-external'

describe('admitPreviewExternalUrl', () => {
  it('admits only http and https', () => {
    expect(admitPreviewExternalUrl('https://example.com/a')).toBe(true)
    expect(admitPreviewExternalUrl('http://example.com')).toBe(true)

    for (const url of ['file:///etc/passwd', 'javascript:alert(1)', 'mailto:a@b.c', 'not a url', '']) {
      expect(admitPreviewExternalUrl(url)).toBe(false)
    }
  })
})

describe('routePreviewGuestLink', () => {
  const link = 'https://www.dianping.com/shop/123'

  it('keeps a remote page browsing inside the in-app Browser', () => {
    expect(routePreviewGuestLink('https://www.dianping.com/beijing', link)).toBe('in-app')
    expect(routePreviewGuestLink('http://example.com', link)).toBe('in-app')
  })

  it("sends a local preview's outbound link to the OS browser (#112941)", () => {
    for (const guest of ['http://localhost:8501', 'http://127.0.0.1:3000/app', 'http://[::1]:4173', 'file:///tmp/a.html']) {
      expect(routePreviewGuestLink(guest, 'https://www.google.com/search?q=traceback')).toBe('external')
    }
  })

  it('does not treat a lookalike host as local', () => {
    expect(routePreviewGuestLink('https://localhost.example.com', link)).toBe('in-app')
  })

  it('falls back to the OS browser when the guest URL is unreadable', () => {
    expect(routePreviewGuestLink('', link)).toBe('external')
    expect(routePreviewGuestLink('about:blank', link)).toBe('external')
  })

  it('denies any link the admission gate rejects, whatever the guest is', () => {
    for (const guest of ['https://example.com', 'http://localhost:8501']) {
      for (const url of ['file:///etc/passwd', 'javascript:alert(1)', 'mailto:a@b.c', '']) {
        expect(routePreviewGuestLink(guest, url)).toBe('deny')
      }
    }
  })
})
