import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  __resetLinkTitleCache,
  ExternalLink,
  fetchLinkTitle,
  hostPathLabel,
  isTitleFetchable,
  LinkifiedText,
  PrettyLink,
  urlSlugTitleLabel
} from './external-link'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

function installDesktopBridge(partial: Partial<Window['hermesDesktop']> = {}) {
  desktopWindow.hermesDesktop = {
    fetchLinkTitle: vi.fn().mockResolvedValue(''),
    openExternal: vi.fn().mockResolvedValue(undefined),
    ...partial
  } as unknown as Window['hermesDesktop']
}

afterEach(() => {
  __resetLinkTitleCache()
  vi.restoreAllMocks()
  cleanup()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('external link helpers', () => {
  it('formats URL fallbacks as host + path', () => {
    expect(
      hostPathLabel(
        'https://www.getyourguide.com/culebra-island-l145468/from-fajardo-full-day-cordillera-islands-catamaran-tour-t19894/'
      )
    ).toBe('getyourguide.com/culebra-island-l145468/from-fajardo-full-day-cordillera-islands-catamaran-tour-t19894')
  })

  it('derives readable title fallbacks from URL slugs', () => {
    expect(
      urlSlugTitleLabel(
        'https://www.getyourguide.com/fajardo-l882/from-fajardo-icacos-island-full-day-catamaran-trip-t19891/'
      )
    ).toBe('From Fajardo Icacos Island Full Day Catamaran Trip')
  })

  it('filters out local/non-http targets for title fetches', () => {
    expect(isTitleFetchable('https://www.expedia.com/things-to-do/foo')).toBe(true)
    expect(isTitleFetchable('http://localhost:5174')).toBe(false)
    expect(isTitleFetchable('http://127')).toBe(false)
    expect(isTitleFetchable('http://0.0.0.127/')).toBe(false)
    expect(isTitleFetchable('http://127.1')).toBe(false)
    expect(isTitleFetchable('http://2130706433')).toBe(false)
    expect(isTitleFetchable('http://10.0.0.1')).toBe(false)
    expect(isTitleFetchable('http://192.168.1.1')).toBe(false)
    expect(isTitleFetchable('https://192.0.0.9')).toBe(true)
    expect(isTitleFetchable('https://192.0.0.10')).toBe(true)
    expect(isTitleFetchable('http://[::1]')).toBe(false)
    expect(isTitleFetchable('http://[fc00::1]')).toBe(false)
    expect(isTitleFetchable('https://[::ffff:8.8.8.8]')).toBe(true)
    expect(isTitleFetchable('http://[::ffff:127.0.0.1]')).toBe(false)
    expect(isTitleFetchable('https://[64:ff9b::808:808]')).toBe(true)
    expect(isTitleFetchable('http://[64:ff9b::7f00:1]')).toBe(false)
    // IANA's globally reachable AMT assignment inside 2001::/23.
    expect(isTitleFetchable('https://[2001:3::1]')).toBe(true)
    expect(isTitleFetchable('https://8.8.8.8')).toBe(true)
    expect(isTitleFetchable('file:///tmp/demo.html')).toBe(false)
    expect(isTitleFetchable('mailto:hello@example.com')).toBe(false)
  })

  it('does not ask Electron to fetch titles for blocked local targets', async () => {
    const bridge = vi.fn().mockResolvedValue('unexpected')
    installDesktopBridge({ fetchLinkTitle: bridge as unknown as Window['hermesDesktop']['fetchLinkTitle'] })

    await expect(fetchLinkTitle('http://127')).resolves.toBe('')
    expect(bridge).not.toHaveBeenCalled()
  })

  it('deduplicates in-flight title fetches and caches results', async () => {
    const bridge = vi.fn().mockResolvedValue('El Yunque Tour Water Slide, Rope Swing & Pickup')
    installDesktopBridge({ fetchLinkTitle: bridge as unknown as Window['hermesDesktop']['fetchLinkTitle'] })

    const url =
      'https://www.expedia.com/things-to-do/puerto-rico-el-yunque-rainforest-adventure-with-transport.a46272756.activity-details'

    const [first, second] = await Promise.all([fetchLinkTitle(url), fetchLinkTitle(url)])

    expect(first).toBe('El Yunque Tour Water Slide, Rope Swing & Pickup')
    expect(second).toBe('El Yunque Tour Water Slide, Rope Swing & Pickup')
    expect(bridge).toHaveBeenCalledTimes(1)

    const third = await fetchLinkTitle(url)

    expect(third).toBe('El Yunque Tour Water Slide, Rope Swing & Pickup')
    expect(bridge).toHaveBeenCalledTimes(1)
  })

  it('shares cache across protocol/www URL variants', async () => {
    const bridge = vi.fn().mockResolvedValue('Shared Canonical Title')
    installDesktopBridge({ fetchLinkTitle: bridge as unknown as Window['hermesDesktop']['fetchLinkTitle'] })

    const first = 'https://www.getyourguide.com/san-juan-puerto-rico-l355/sunset-tours-tc306/'
    const second = 'http://getyourguide.com/san-juan-puerto-rico-l355/sunset-tours-tc306/'

    const [a, b] = await Promise.all([fetchLinkTitle(first), fetchLinkTitle(second)])

    expect(a).toBe('Shared Canonical Title')
    expect(b).toBe('Shared Canonical Title')
    expect(bridge).toHaveBeenCalledTimes(1)
  })

  it('opens links via the desktop bridge', () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)
    installDesktopBridge({ openExternal: openExternal as unknown as Window['hermesDesktop']['openExternal'] })

    render(<ExternalLink href="https://example.com/path/to/resource">Example link</ExternalLink>)

    fireEvent.click(screen.getByRole('link', { name: 'Example link' }))
    expect(openExternal).toHaveBeenCalledWith('https://example.com/path/to/resource')
  })

  it('shows a trailing external-link icon', () => {
    installDesktopBridge()

    render(<ExternalLink href="https://example.com/path/to/resource">Example link</ExternalLink>)

    const link = screen.getByRole('link', { name: 'Example link' })
    expect(link.querySelector('svg')).toBeTruthy()
  })

  it('renders pretty links with fetched titles and no host suffix', async () => {
    const bridge = vi.fn().mockResolvedValue('From Fajardo: Full-Day Culebra Islands Catamaran Tour')
    installDesktopBridge({ fetchLinkTitle: bridge as unknown as Window['hermesDesktop']['fetchLinkTitle'] })

    const url =
      'https://www.getyourguide.com/culebra-island-l145468/from-fajardo-full-day-cordillera-islands-catamaran-tour-t19894/'

    render(<LinkifiedText text={`Read ${url}`} />)

    const link = screen.getByTitle(url)
    expect(link.textContent).toContain('From Fajardo Full Day Cordillera Islands Catamaran Tour')

    await waitFor(() => {
      expect(link.textContent).toContain('From Fajardo: Full-Day Culebra Islands Catamaran Tour')
    })
    expect(link.textContent).not.toContain('getyourguide.com')
  })

  it('shows host/path fallback when title is unavailable', () => {
    installDesktopBridge()
    const url = 'https://www.expedia.com/things-to-do/puerto-rico-el-yunque'

    render(<PrettyLink href={url} />)

    const link = screen.getByTitle(url)

    expect(link.textContent).toBe('Puerto Rico El Yunque')
  })

  it('ignores error-like fetched titles and falls back to slug label', async () => {
    const bridge = vi.fn().mockResolvedValue('GetYourGuide – Error')
    installDesktopBridge({ fetchLinkTitle: bridge as unknown as Window['hermesDesktop']['fetchLinkTitle'] })

    const url =
      'https://www.getyourguide.com/culebra-island-l145468/from-fajardo-full-day-cordillera-islands-catamaran-tour-t19894/'

    render(<PrettyLink href={url} />)

    const link = screen.getByTitle(url)
    await waitFor(() => {
      expect(link.textContent).toBe('From Fajardo Full Day Cordillera Islands Catamaran Tour')
    })
  })

  it('normalizes scheme-less links before opening', () => {
    installDesktopBridge()

    render(<LinkifiedText text="Source expedia.com/things-to-do/puerto-rico-el-yunque-rainforest-adventure" />)

    const link = screen.getByRole('link')
    expect(link.getAttribute('href')).toBe(
      'https://expedia.com/things-to-do/puerto-rico-el-yunque-rainforest-adventure'
    )
  })

  it('explicitOnly skips bare filename/domain tokens and only links explicit URLs', () => {
    installDesktopBridge()

    render(
      <LinkifiedText
        explicitOnly
        pretty={false}
        text={'Report  https://paste.rs/abc\nagent.log  https://paste.rs/def\nerrors.log'}
      />
    )

    const links = screen.getAllByRole('link')
    expect(links.map(a => a.getAttribute('href'))).toEqual(['https://paste.rs/abc', 'https://paste.rs/def'])
    // Bare filename-shaped tokens stay as plain text, not links.
    expect(screen.queryByText(content => content.includes('agent.log'))).toBeTruthy()
    expect(links.some(a => (a.textContent ?? '').includes('.log'))).toBe(false)
  })

  it('without explicitOnly, bare filename tokens are still linkified (default behavior)', () => {
    installDesktopBridge()

    render(<LinkifiedText pretty={false} text="open agent.log please" />)

    const link = screen.getByRole('link', { name: 'agent.log' })
    expect(link.getAttribute('href')).toBe('https://agent.log')
  })
})
