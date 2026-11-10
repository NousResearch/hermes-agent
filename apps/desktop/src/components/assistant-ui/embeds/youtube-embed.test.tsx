import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { detectEmbed } from './providers'
import type { FrameEmbed } from './providers/types'
import YouTubeEmbedRenderer, { wrappedYoutubeSrc } from './youtube-embed'

afterEach(cleanup)

const descriptor = detectEmbed('https://www.youtube.com/watch?v=M7lc1UVf-VE&t=42') as FrameEmbed

describe('YouTubeEmbedRenderer', () => {
  it('embeds directly with its own origin on an http renderer, never asking for the host', () => {
    const getEmbedHostOrigin = vi.fn()
    window.hermesDesktop = { ...window.hermesDesktop, getEmbedHostOrigin }

    const { container } = render(<YouTubeEmbedRenderer descriptor={descriptor} />)
    const src = new URL(container.querySelector('iframe')!.src)

    expect(src.origin).toBe('https://www.youtube-nocookie.com')
    expect(src.searchParams.get('origin')).toBe(window.location.origin)
    expect(getEmbedHostOrigin).not.toHaveBeenCalled()
  })

  it('maps the player URL onto the loopback wrapper with the same video and params', () => {
    const src = new URL(wrappedYoutubeSrc(descriptor.embedUrl, 'http://127.0.0.1:5555'))

    expect(src.origin).toBe('http://127.0.0.1:5555')
    expect(src.pathname).toBe('/youtube/M7lc1UVf-VE')
    expect(src.searchParams.get('start')).toBe('42')
    expect(src.searchParams.get('rel')).toBe('0')
  })
})
