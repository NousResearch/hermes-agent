import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ConnectorLogo } from '@/components/ui/connector-logo'
import { connectorIconUrl } from '@/lib/connector-tools'

const ICON = connectorIconUrl('acme')

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('the connector mark ladder', () => {
  it('derives the vendor logo from the toolkit slug', () => {
    expect(connectorIconUrl('gmail')).toBe('https://logos.composio.dev/api/gmail')
  })

  it('draws the vendor icon as a plain image, ahead of any favicon lookup', () => {
    const resolveFavicon = vi.fn()
    // SAFETY: the favicon rung reads only `resolveFavicon` from the preload bridge.
    window.hermesDesktop = { resolveFavicon } as never

    const { container } = render(<ConnectorLogo connector={{ homepage: 'https://acme.test', iconUrl: ICON, name: 'acme' }} />)

    expect(container.querySelector('img')?.getAttribute('src')).toBe(ICON)
    expect(resolveFavicon).not.toHaveBeenCalled()
  })

  it('keeps the curated glyph for a brand the app ships, even when the vendor sends an icon', () => {
    const { container } = render(<ConnectorLogo connector={{ iconUrl: ICON, name: 'linear' }} />)

    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('svg')).not.toBeNull()
  })

  it('falls to the monogram when there is no glyph, no icon and no site', () => {
    const { container } = render(<ConnectorLogo connector={{ name: 'acme' }} />)

    expect(container.querySelector('img')).toBeNull()
    expect(container.textContent).toBe('A')
  })
})
