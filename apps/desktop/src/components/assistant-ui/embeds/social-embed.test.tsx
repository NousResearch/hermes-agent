import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { detectEmbed } from './providers'
import SocialEmbedRenderer, { reportedHeight } from './social-embed'

afterEach(() => {
  cleanup()
  globalThis.document.documentElement.classList.remove('dark')
})

function renderEmbed(url: string) {
  const descriptor = detectEmbed(url)

  if (!descriptor) {
    throw new Error(`expected an embed for ${url}`)
  }

  const { container } = render(<SocialEmbedRenderer descriptor={descriptor} />)
  // Checked before the frame lookup so a script injection fails loudly.
  expect(globalThis.document.querySelectorAll('script')).toHaveLength(0)
  const frame = container.querySelector('iframe')

  if (!frame) {
    throw new Error('expected an iframe')
  }

  return frame
}

function post(frame: HTMLIFrameElement, origin: string, data: unknown, source = frame.contentWindow) {
  act(() => {
    window.dispatchEvent(new MessageEvent('message', { data, origin, source }))
  })
}

describe('SocialEmbedRenderer', () => {
  // The app document holds the preload bridge (window.hermesDesktop), so no
  // vendor script may ever be loaded into it.
  it.each([
    ['https://x.com/jack/status/20', 'https://platform.twitter.com/embed/Tweet.html?id=20&theme=light&dnt=true'],
    ['https://www.instagram.com/p/CabcDEF123/', 'https://www.instagram.com/p/CabcDEF123/embed'],
    ['https://www.instagram.com/reel/CabcDEF123/', 'https://www.instagram.com/reel/CabcDEF123/embed']
  ])('renders %s in a sandboxed cross-origin iframe without injecting a script', (url, src) => {
    const frame = renderEmbed(url)

    expect(frame.getAttribute('src')).toBe(src)
    expect(frame.getAttribute('sandbox')).toBe('allow-scripts allow-same-origin')
    expect(frame.getAttribute('referrerpolicy')).toBe('strict-origin-when-cross-origin')
  })

  it('passes the app theme to the tweet frame', () => {
    globalThis.document.documentElement.classList.add('dark')

    expect(new URL(renderEmbed('https://twitter.com/jack/status/20').src).searchParams.get('theme')).toBe('dark')
  })

  it("sizes the tweet frame from X's resize message", () => {
    const frame = renderEmbed('https://x.com/jack/status/20')

    post(frame, 'https://platform.twitter.com', {
      'twttr.embed': {
        id: 'embed-0',
        jsonrpc: '2.0',
        method: 'twttr.private.resize',
        params: [{ height: 225, width: 480 }]
      }
    })

    expect(frame.style.height).toBe('225px')
  })

  it('sizes the Instagram frame from its MEASURE message', () => {
    const frame = renderEmbed('https://www.instagram.com/p/CabcDEF123/')

    expect(frame.style.height).toBe('450px')
    post(frame, 'https://www.instagram.com', JSON.stringify({ details: { height: 608 }, type: 'MEASURE' }))

    expect(frame.style.height).toBe('608px')
  })

  it('ignores height messages from another origin or another window', () => {
    const frame = renderEmbed('https://www.instagram.com/p/CabcDEF123/')
    const measure = JSON.stringify({ details: { height: 999 }, type: 'MEASURE' })

    post(frame, 'https://platform.twitter.com', measure)
    post(frame, 'https://www.instagram.com', measure, window)

    expect(frame.style.height).toBe('450px')
  })
})

describe('reportedHeight', () => {
  it('reads only a positive finite height', () => {
    expect(reportedHeight('{"type":"MEASURE","details":{"height":608.2}}')).toBe(609)
    expect(reportedHeight({ 'twttr.embed': { method: 'twttr.private.resize', params: [{ height: 225 }] } })).toBe(225)
    expect(
      reportedHeight({ 'twttr.embed': { method: 'twttr.private.rendered', params: [{ height: 225 }] } })
    ).toBeNull()
    expect(reportedHeight('{"type":"MEASURE","details":{"height":"9e9"}}')).toBeNull()
    expect(reportedHeight({ type: 'MEASURE', details: { height: -1 } })).toBeNull()
    expect(reportedHeight('not json')).toBeNull()
    expect(reportedHeight(null)).toBeNull()
  })
})
