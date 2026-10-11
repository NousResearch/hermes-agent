import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ConnectorLogo } from './connector-logo'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('connector vendor marks', () => {
  it('keeps an unknown provider readable when its public logo cannot load', () => {
    const { container } = render(
      <ConnectorLogo
        connector={{ iconUrl: 'https://logos.example.test/resend', name: 'unlisted-provider', title: 'Resend' }}
      />
    )

    const icon = container.querySelector('img')!
    fireEvent.error(icon)

    expect(container.querySelector('img')).toBeNull()
    expect(screen.getByText('R')).toBeTruthy()
  })

  it('tries a new vendor URL when the same component changes providers', () => {
    const { container, rerender } = render(
      <ConnectorLogo connector={{ iconUrl: 'https://logos.example.test/first', name: 'first-provider' }} />
    )

    fireEvent.error(container.querySelector('img')!)
    rerender(<ConnectorLogo connector={{ iconUrl: 'https://logos.example.test/second', name: 'second-provider' }} />)

    expect(container.querySelector('img')?.getAttribute('src')).toBe('https://logos.example.test/second')
  })

  it('does not let a late failure from the previous URL hide the replacement', () => {
    const { container, rerender } = render(
      <ConnectorLogo connector={{ iconUrl: 'https://logos.example.test/old', name: 'old-provider' }} />
    )

    const oldIcon = container.querySelector('img')!
    rerender(<ConnectorLogo connector={{ iconUrl: 'https://logos.example.test/new', name: 'new-provider' }} />)
    fireEvent.error(oldIcon)

    expect(container.querySelector('img')).not.toBe(oldIcon)
    expect(container.querySelector('img')?.getAttribute('src')).toBe('https://logos.example.test/new')
  })

  it('uses the existing favicon lookup after the vendor image fails', async () => {
    const resolveFavicon = vi.fn().mockResolvedValue('')
    vi.stubGlobal('hermesDesktop', { resolveFavicon })

    const { container } = render(
      <ConnectorLogo
        connector={{
          homepage: 'https://fallback-brand.example.test',
          iconUrl: 'https://logos.example.test/fallback',
          name: 'fallback-provider'
        }}
      />
    )

    fireEvent.error(container.querySelector('img')!)

    await act(async () => {
      await vi.waitFor(() => expect(resolveFavicon).toHaveBeenCalledWith('https://fallback-brand.example.test'))
    })
    await vi.waitFor(() => expect(screen.getByText('F')).toBeTruthy())
  })
})
