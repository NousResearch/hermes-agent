import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $embedMode } from '@/store/embed-consent'

import { detectEmbed } from './providers'
import { UrlEmbed } from './url-embed'

// Embed mode "off" promises plain links ("Off keeps plain links"). A plain link
// must not make the desktop fetch the provider page for a title: that is the
// exact outbound request the privacy gate exists to prevent.
describe('UrlEmbed with embeds off', () => {
  const fetchLinkTitle = vi.fn().mockResolvedValue('Some video title')

  beforeEach(() => {
    fetchLinkTitle.mockClear()
    $embedMode.set('off')
    ;(window as unknown as { hermesDesktop: object }).hermesDesktop = {
      fetchLinkTitle,
      openExternal: vi.fn().mockResolvedValue(undefined)
    }
  })

  afterEach(() => {
    cleanup()
    $embedMode.set('ask')
    delete (window as unknown as { hermesDesktop?: object }).hermesDesktop
  })

  it('renders the source URL as a plain link without fetching the provider page', async () => {
    const url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    const descriptor = detectEmbed(url)

    expect(descriptor).not.toBeNull()
    render(<UrlEmbed descriptor={descriptor!} />)

    const anchor = (await screen.findByRole('link')) as HTMLAnchorElement

    expect(anchor.getAttribute('href')).toBe(url)
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(fetchLinkTitle).not.toHaveBeenCalled()
  })
})
