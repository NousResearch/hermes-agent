import assert from 'node:assert/strict'

import { describe, expect, it, vi } from 'vitest'

import {
  computeSite,
  createLinkTitleRequestTracker,
  isSameSite,
  LinkTitleRequestTracker,
  shouldBlockThirdPartyTitleRequest
} from './link-title-third-party'

describe('computeSite and isSameSite', () => {
  it('treats multi-label public suffixes (*.com.cn, *.co.uk) on the same domain as same-site', () => {
    assert.equal(computeSite('https://a.sina.com.cn/news'), 'sina.com.cn')
    assert.equal(computeSite('https://b.sina.com.cn/sports'), 'sina.com.cn')
    assert.equal(isSameSite('https://a.sina.com.cn/news', 'https://b.sina.com.cn/sports'), true)

    assert.equal(computeSite('https://www.example.co.uk/index.html'), 'example.co.uk')
    assert.equal(computeSite('https://static.example.co.uk/bundle.js'), 'example.co.uk')
    assert.equal(isSameSite('https://www.example.co.uk/index.html', 'https://static.example.co.uk/bundle.js'), true)
  })

  it('treats different registrable domains as different sites', () => {
    assert.equal(computeSite('https://k.sina.cn/article_123.html'), 'sina.cn')
    assert.equal(computeSite('https://p.inping.com/ad.js'), 'inping.com')
    assert.equal(isSameSite('https://k.sina.cn/article_123.html', 'https://p.inping.com/ad.js'), false)
  })

  it('compares IP literals and single-label hosts by exact host', () => {
    assert.equal(computeSite('http://127.0.0.1:8080/foo'), '127.0.0.1')
    assert.equal(computeSite('http://127.0.0.1:9090/bar'), '127.0.0.1')
    assert.equal(isSameSite('http://127.0.0.1:8080/foo', 'http://127.0.0.1:9090/bar'), true)

    assert.equal(computeSite('http://192.168.1.1:8080/bar'), '192.168.1.1')
    assert.equal(isSameSite('http://127.0.0.1:8080/foo', 'http://192.168.1.1:8080/bar'), false)

    assert.equal(computeSite('http://localhost:3000/app'), 'localhost')
    assert.equal(computeSite('http://localhost:4000/api'), 'localhost')
    assert.equal(isSameSite('http://localhost:3000/app', 'http://localhost:4000/api'), true)
    assert.equal(isSameSite('http://localhost:3000/app', 'http://127.0.0.1:3000/app'), false)

    assert.equal(computeSite('http://[::1]:8080/test'), '::1')
    assert.equal(computeSite('http://[::1]:9090/test'), '::1')
    assert.equal(isSameSite('http://[::1]:8080/test', 'http://[::1]:9090/test'), true)
    assert.equal(isSameSite('http://[::1]:8080/test', 'http://[2001:db8::1]:8080/test'), false)
  })

  it('handles blob URLs, invalid URLs, and non-URL schemes gracefully', () => {
    assert.equal(isSameSite('blob:https://www.example.co.uk/uuid-1234', 'https://example.co.uk/page'), true)
    assert.equal(isSameSite('javascript:void(0)', 'https://example.com'), false)
    assert.equal(isSameSite('about:blank', 'https://example.com'), false)
    assert.equal(isSameSite('data:text/html,test', 'https://example.com'), false)
    assert.equal(isSameSite('', 'https://example.com'), false)
    assert.equal(isSameSite('not a url', 'https://example.com'), false)
  })
})

describe('shouldBlockThirdPartyTitleRequest', () => {
  const topUrl = 'https://k.sina.cn/article_001.html'

  it('allows same-site subresource requests', () => {
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'script', url: 'https://k.sina.cn/runtime.js' }, topUrl),
      false
    )
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'xhr', url: 'https://api.sina.cn/data.json' }, topUrl),
      false
    )
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'image', url: 'https://img.sina.cn/banner.jpg' }, topUrl),
      false
    )
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'subFrame', url: 'https://k.sina.cn/embed.html' }, topUrl),
      false
    )
  })

  it('cancels third-party script, xhr, subFrame, and image requests', () => {
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'script', url: 'https://p.inping.com/tracker.js' }, topUrl),
      true
    )
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'xhr', url: 'https://adservice.google.com/beacon' }, topUrl),
      true
    )
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'subFrame', url: 'https://doubleclick.net/ad.html' }, topUrl),
      true
    )
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'image', url: 'https://tracking.pixel.com/1x1.gif' }, topUrl),
      true
    )
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'webSocket', url: 'wss://realtime.tracker.io/socket' }, topUrl),
      true
    )
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'other', url: 'https://beacon.other.com/collect' }, topUrl),
      true
    )
  })

  it('always allows mainFrame requests (initial navigations and redirect hops)', () => {
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'mainFrame', url: 'https://k.sina.cn/article_001.html' }, topUrl),
      false
    )
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'mainFrame', url: 'https://external-redirect.com/login' }, topUrl),
      false
    )
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'mainFrame', url: 'https://target.test/' }, undefined),
      false
    )
  })

  it('fails closed when top-level URL is unknown or unresolvable', () => {
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'script', url: 'https://k.sina.cn/runtime.js' }, undefined),
      true
    )
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'xhr', url: 'https://k.sina.cn/data.json' }, ''),
      true
    )
    assert.equal(
      shouldBlockThirdPartyTitleRequest({ resourceType: 'image', url: 'https://k.sina.cn/img.png' }, 'invalid-url'),
      true
    )
  })
})

describe('LinkTitleRequestTracker', () => {
  it('tracks top-level navigations and updates site on mainFrame redirect', () => {
    const tracker = createLinkTitleRequestTracker()

    // Hop 1: initial mainFrame request to shortener
    assert.equal(
      tracker.shouldBlock({ resourceType: 'mainFrame', url: 'https://short.url/xyz', webContentsId: 42 }),
      false
    )

    // Subresource on shortener host is allowed
    assert.equal(
      tracker.shouldBlock({ resourceType: 'script', url: 'https://short.url/bundle.js', webContentsId: 42 }),
      false
    )
    // Subresource on final target host is third-party at this stage
    assert.equal(
      tracker.shouldBlock({ resourceType: 'script', url: 'https://k.sina.cn/news.js', webContentsId: 42 }),
      true
    )

    // Hop 2: 302 redirect of mainFrame to k.sina.cn
    assert.equal(
      tracker.shouldBlock({ resourceType: 'mainFrame', url: 'https://k.sina.cn/article_456.html', webContentsId: 42 }),
      false
    )

    // Now k.sina.cn is the top-level site: allowed
    assert.equal(
      tracker.shouldBlock({ resourceType: 'script', url: 'https://k.sina.cn/news.js', webContentsId: 42 }),
      false
    )
    // Shortener host is now cross-site: blocked
    assert.equal(
      tracker.shouldBlock({ resourceType: 'script', url: 'https://short.url/bundle.js', webContentsId: 42 }),
      true
    )
    // Third-party tracker on destination page is blocked
    assert.equal(
      tracker.shouldBlock({ resourceType: 'script', url: 'https://p.inping.com/tracker.js', webContentsId: 42 }),
      true
    )
  })

  it('isolates concurrent windows by webContentsId', () => {
    const tracker = new LinkTitleRequestTracker()

    tracker.shouldBlock({ resourceType: 'mainFrame', url: 'https://a.com/page', webContentsId: 1 })
    tracker.shouldBlock({ resourceType: 'mainFrame', url: 'https://b.com/page', webContentsId: 2 })

    // Window 1 requests
    assert.equal(tracker.shouldBlock({ resourceType: 'script', url: 'https://a.com/app.js', webContentsId: 1 }), false)
    assert.equal(tracker.shouldBlock({ resourceType: 'script', url: 'https://b.com/app.js', webContentsId: 1 }), true)

    // Window 2 requests
    assert.equal(tracker.shouldBlock({ resourceType: 'script', url: 'https://b.com/app.js', webContentsId: 2 }), false)
    assert.equal(tracker.shouldBlock({ resourceType: 'script', url: 'https://a.com/app.js', webContentsId: 2 }), true)
  })

  it('falls back to details.frame.top.url when webContentsId is untracked', () => {
    const tracker = new LinkTitleRequestTracker()

    const blocked = tracker.shouldBlock({
      frame: { top: { url: 'https://example.com/top' } },
      resourceType: 'script',
      url: 'https://evil-tracker.com/beacon.js'
    })

    const allowed = tracker.shouldBlock({
      frame: { top: { url: 'https://example.com/top' } },
      resourceType: 'script',
      url: 'https://example.com/script.js'
    })

    assert.equal(blocked, true)
    assert.equal(allowed, false)
  })

  it('cleans up webContents tracking on window destruction', () => {
    const tracker = new LinkTitleRequestTracker()
    tracker.recordMainFrame(100, 'https://example.com/article')

    assert.equal(tracker.getTopLevelUrl(100), 'https://example.com/article')
    tracker.deleteWebContents(100)
    assert.equal(tracker.getTopLevelUrl(100), undefined)

    // Now fail-closed because webContents is gone
    assert.equal(tracker.shouldBlock({ resourceType: 'script', url: 'https://example.com/script.js', webContentsId: 100 }), true)
  })
})

describe('hermes:link-titles webRequest wiring', () => {
  it('cancels third-party requests and blocked resource types, but allows mainFrame and first-party scripts', () => {
    const tracker = createLinkTitleRequestTracker()
    let onBeforeRequestListener: ((details: any, callback: (res: { cancel?: boolean }) => void) => void) | null = null

    const mockSession = {
      webRequest: {
        onBeforeRequest: vi.fn((listener: any) => {
          onBeforeRequestListener = listener
        })
      }
    }

    const blockedResources = new Set(['stylesheet', 'font', 'ping'])

    // Attach listener following the link-metadata pattern
    mockSession.webRequest.onBeforeRequest((details: any, callback: any) => {
      if (blockedResources.has(details.resourceType)) {
        return callback({ cancel: true })
      }

      callback({ cancel: tracker.shouldBlock(details) })
    })

    expect(onBeforeRequestListener).toBeDefined()

    // 1. Initial mainFrame request: allowed and records top-level URL
    const cb1 = vi.fn()
    onBeforeRequestListener!({ resourceType: 'mainFrame', url: 'https://k.sina.cn/article.html', webContentsId: 1 }, cb1)
    expect(cb1).toHaveBeenCalledWith({ cancel: false })

    // 2. Blocklist type (stylesheet) from same site: cancelled by resource-type rule
    const cb2 = vi.fn()
    onBeforeRequestListener!({ resourceType: 'stylesheet', url: 'https://k.sina.cn/style.css', webContentsId: 1 }, cb2)
    expect(cb2).toHaveBeenCalledWith({ cancel: true })

    // 3. First-party script: allowed
    const cb3 = vi.fn()
    onBeforeRequestListener!({ resourceType: 'script', url: 'https://k.sina.cn/render.js', webContentsId: 1 }, cb3)
    expect(cb3).toHaveBeenCalledWith({ cancel: false })

    // 4. Third-party tracker script (incident repro): cancelled
    const cb4 = vi.fn()
    onBeforeRequestListener!({ resourceType: 'script', url: 'https://p.inping.com/beacon.js', webContentsId: 1 }, cb4)
    expect(cb4).toHaveBeenCalledWith({ cancel: true })

    // 5. Third-party ad iframe: cancelled
    const cb5 = vi.fn()
    onBeforeRequestListener!({ resourceType: 'subFrame', url: 'https://ad.doubleclick.net/slot', webContentsId: 1 }, cb5)
    expect(cb5).toHaveBeenCalledWith({ cancel: true })
  })
})
