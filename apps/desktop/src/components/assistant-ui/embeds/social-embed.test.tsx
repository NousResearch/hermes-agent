import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { detectEmbed } from './providers'
import SocialEmbedRenderer from './social-embed'

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

  it('ignores height messages from another origin or another window', () => {
    const frame = renderEmbed('https://www.instagram.com/p/CabcDEF123/')
    const measure = JSON.stringify({ details: { height: 999 }, type: 'MEASURE' })

    post(frame, 'https://platform.twitter.com', measure)
    post(frame, 'https://www.instagram.com', measure, window)

    expect(frame.style.height).toBe('450px')

    // Positive control: the same message from the frame's own origin and window resizes it.
    post(frame, 'https://www.instagram.com', measure)

    expect(frame.style.height).toBe('999px')
  })
})
