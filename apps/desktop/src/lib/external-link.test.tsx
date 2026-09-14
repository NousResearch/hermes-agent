import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { IS_MAC } from '@/lib/keybinds/combo'
import { $previewTabs, closeRightRail } from '@/store/preview'

import {
  __resetLinkTitleCache,
  ExternalLink,
  fetchLinkTitle,
  hostPathLabel,
  hudForcesNativeLinks,
  isTitleFetchable,
  LinkifiedText,
  MarkdownLinkText,
  normalizeExternalUrl,
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

const FORGEJO_URL = 'https://forgejo.home.example/homelab/homelab-ops/issues/101'

function installTitleBridge(title: string) {
  const bridge = vi.fn().mockResolvedValue(title)

  installDesktopBridge({ fetchLinkTitle: bridge as unknown as Window['hermesDesktop']['fetchLinkTitle'] })

  return bridge
}

afterEach(() => {
  __resetLinkTitleCache()
  closeRightRail()
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
    expect(isTitleFetchable('file:///tmp/demo.html')).toBe(false)
    expect(isTitleFetchable('mailto:hello@example.com')).toBe(false)
  })

  // #93893: wire-format directive tokens reaching this module verbatim must
  // unwrap to their inner URL, never surface the wrapper as a fetchable
  // "host" (`@url` from `new URL('@url:`…`')`).
  describe('normalizeExternalUrl unwraps directive wrappers (#93893)', () => {
    it('unwraps a backtick-quoted @url: token', () => {
      expect(normalizeExternalUrl('@url:`https://oauth2:%s@example.internal.host`')).toBe(
        'https://oauth2:%s@example.internal.host'
      )
    })

    it('unwraps double-quote and bare quoted forms', () => {
      expect(normalizeExternalUrl('@url:"https://example.com/a b"')).toBe('https://example.com/a b')
      expect(normalizeExternalUrl('@url:https://example.com/x')).toBe('https://example.com/x')
    })

    it('peels trailing punctuation from the bare form (review follow-up)', () => {
      expect(normalizeExternalUrl('@url:https://example.com.')).toBe('https://example.com')
      expect(normalizeExternalUrl('@url:https://example.com/x,')).toBe('https://example.com/x')
    })

    it('does NOT unwrap @file/@folder — local paths must not become links', () => {
      // The wrapper keeps these inert downstream rather than promoting a bare
      // path to a link value (#93893 review, point 1).
      expect(normalizeExternalUrl('@file:`/home/user/report.md`')).toBe('@file:`/home/user/report.md`')
      expect(isTitleFetchable('@file:`/home/user/report.md`')).toBe(false)
    })

    it('leaves plain URLs and non-directive text untouched', () => {
      expect(normalizeExternalUrl('https://example.com')).toBe('https://example.com')
      expect(normalizeExternalUrl('not a directive but @url: is mentioned mid-sentence')).toBe(
        'not a directive but @url: is mentioned mid-sentence'
      )
    })

    it('makes wrapped URLs title-fetchable and keeps placeholder URLs un-fetched at the main boundary', () => {
      expect(isTitleFetchable('@url:`https://example.com/docs`')).toBe(true)
      // The wrapper itself still parses as garbage — nothing about the unwrap
      // turns junk into a URL.
      expect(isTitleFetchable('@url:`not-a-url`')).toBe(false)
    })
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

  // A web link belongs in the in-app browser now; the OS browser is the
  // ⌘/Ctrl-click escape hatch.
  it('opens a web link in the in-app browser', async () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)
    installDesktopBridge({ openExternal: openExternal as unknown as Window['hermesDesktop']['openExternal'] })

    render(<ExternalLink href="https://example.com/path/to/resource">Example link</ExternalLink>)

    fireEvent.click(screen.getByRole('link', { name: 'Example link' }))

    expect(openExternal).not.toHaveBeenCalled()
    await waitFor(() => expect($previewTabs.get().at(-1)?.target.url).toBe('https://example.com/path/to/resource'))
  })

  // Platform-specific on purpose (same rule as terminal links / middle-click):
  // ⌘ on macOS, Ctrl elsewhere. The suite runs as non-mac.
  it('escapes to the OS browser on the platform open-elsewhere modifier', () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)
    installDesktopBridge({ openExternal: openExternal as unknown as Window['hermesDesktop']['openExternal'] })

    render(<ExternalLink href="https://example.com/path/to/resource">Example link</ExternalLink>)

    fireEvent.click(screen.getByRole('link', { name: 'Example link' }), IS_MAC ? { metaKey: true } : { ctrlKey: true })

    expect(openExternal).toHaveBeenCalledWith('https://example.com/path/to/resource')
    expect($previewTabs.get()).toHaveLength(0)
  })

  it('treats only the HUD renderer as a native-link surface', () => {
    expect(hudForcesNativeLinks('')).toBe(false)
    expect(hudForcesNativeLinks('?win=secondary')).toBe(false)
    expect(hudForcesNativeLinks('?win=browser&tab=1')).toBe(false)
    expect(hudForcesNativeLinks('?win=hud')).toBe(true)
    expect(hudForcesNativeLinks('?profile=work&win=hud')).toBe(true)
  })

  // The HUD has no in-app browser. A click that opened a preview tile would
  // try to paint a webview into the transparent overlay (OAuth, consoles).
  it('sends every HUD web link to the OS browser', () => {
    const originalLocation = window.location

    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { ...originalLocation, search: '?win=hud' }
    })

    try {
      const openExternal = vi.fn().mockResolvedValue(undefined)
      installDesktopBridge({ openExternal: openExternal as unknown as Window['hermesDesktop']['openExternal'] })

      render(<ExternalLink href="https://accounts.google.com/o/oauth2/auth">Sign in</ExternalLink>)

      fireEvent.click(screen.getByRole('link', { name: 'Sign in' }))

      expect(openExternal).toHaveBeenCalledWith('https://accounts.google.com/o/oauth2/auth')
      expect($previewTabs.get()).toHaveLength(0)
    } finally {
      Object.defineProperty(window, 'location', { configurable: true, value: originalLocation })
    }
  })

  // A setup step sends you to a console you are signed into in your own
  // browser, to fill in a form and copy a secret back. The in-app pane has
  // none of that session and is the wrong destination even for web URLs.
  it('sends a setup-step link straight to the OS browser', () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)
    installDesktopBridge({ openExternal: openExternal as unknown as Window['hermesDesktop']['openExternal'] })

    render(<MarkdownLinkText text="Enable the [Docs API](https://console.cloud.google.com/apis/library) first." />)

    fireEvent.click(screen.getByRole('link', { name: 'Docs API' }))

    expect(openExternal).toHaveBeenCalledWith('https://console.cloud.google.com/apis/library')
    expect($previewTabs.get()).toHaveLength(0)
  })

  // A webview can't do anything useful with these, so they always hand off.
  it('hands a non-web scheme to the OS', () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)
    installDesktopBridge({ openExternal: openExternal as unknown as Window['hermesDesktop']['openExternal'] })

    render(<ExternalLink href="mailto:hi@example.com">Mail</ExternalLink>)

    fireEvent.click(screen.getByRole('link', { name: 'Mail' }))

    expect(openExternal).toHaveBeenCalledWith('mailto:hi@example.com')
  })

  it('hides the trailing external-link icon by default', () => {
    installDesktopBridge()

    render(<ExternalLink href="https://example.com/path/to/resource">Example link</ExternalLink>)

    const link = screen.getByRole('link', { name: 'Example link' })
    expect(link.querySelector('svg')).toBeNull()
  })

  it('shows a trailing external-link icon when opted in', () => {
    installDesktopBridge()

    render(
      <ExternalLink href="https://example.com/path/to/resource" showExternalIcon>
        Example link
      </ExternalLink>
    )

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

  it('treats not-found fetched titles as unusable', async () => {
    const bridge = installTitleBridge('Page not found - Forgejo')

    await expect(fetchLinkTitle(FORGEJO_URL)).resolves.toBe('')
    expect(bridge).toHaveBeenCalledTimes(1)
  })

  it('keeps an authored fallbackLabel ahead of a fetched title, and skips the fetch', async () => {
    const bridge = installTitleBridge('Kinkolino Forgejo')

    // Chat markdown passes authored link text as `fallbackLabel`, not `label`.
    render(<PrettyLink fallbackLabel="FJ #101" href={FORGEJO_URL} />)

    const link = screen.getByTitle(FORGEJO_URL)

    await waitFor(() => {
      expect(link.textContent).toContain('FJ #101')
    })
    expect(link.textContent).not.toContain('Kinkolino Forgejo')
    expect(bridge).not.toHaveBeenCalled()
  })

  it('still resolves a title when no label was authored', async () => {
    const bridge = installTitleBridge('Homelab Ops Issue 101')

    render(<PrettyLink href={FORGEJO_URL} />)

    await waitFor(() => {
      expect(screen.getByTitle(FORGEJO_URL).textContent).toContain('Homelab Ops Issue 101')
    })
    expect(bridge).toHaveBeenCalledTimes(1)
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

  it('prefixes a pretty link to a known host with its brand glyph', () => {
    installDesktopBridge()

    const url = 'https://github.com/NousResearch/hermes-agent/pull/123'

    render(<PrettyLink fallbackLabel="#123" href={url} />)

    const link = screen.getByTitle(url)

    expect(link.querySelector('svg')).toBeTruthy()
    // The glyph is decorative — it must not pollute the link's accessible name.
    expect(link.textContent).toBe('#123')
  })

  it('renders no brand glyph for an unknown host', () => {
    installDesktopBridge()

    const url = 'https://example.com/some/page'

    render(<PrettyLink fallbackLabel="Some Page" href={url} />)

    expect(screen.getByTitle(url).querySelector('svg')).toBeNull()
  })
})
