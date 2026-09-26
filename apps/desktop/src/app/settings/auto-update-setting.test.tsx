import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopAutoUpdateView } from '@/global'
import { en } from '@/i18n/en'
import { _resetAutoUpdateForTests } from '@/store/auto-update'

import { AutoUpdateSetting } from './auto-update-setting'

const u = en.updates.autoUpdate

function installBridge(initial: DesktopAutoUpdateView) {
  let view = initial

  const auto = {
    get: vi.fn(async () => view),
    set: vi.fn(async (enabled: boolean) => {
      view = { ...view, enabled }

      return view
    }),
    claim: vi.fn(),
    report: vi.fn()
  }

  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { updates: { auto } }

  return auto
}

describe('AutoUpdateSetting', () => {
  beforeEach(() => {
    _resetAutoUpdateForTests()
  })

  afterEach(() => {
    cleanup()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('renders an off-by-default switch with the explanation', async () => {
    installBridge({ enabled: false, supported: true, lastAttempt: null })
    render(<AutoUpdateSetting />)

    const toggle = await screen.findByRole('switch', { name: u.title })

    expect(toggle.getAttribute('aria-checked')).toBe('false')
    expect(screen.getByText(u.desc)).toBeTruthy()
  })

  it('persists the choice through the Electron bridge', async () => {
    const auto = installBridge({ enabled: false, supported: true, lastAttempt: null })
    render(<AutoUpdateSetting />)

    fireEvent.click(await screen.findByRole('switch', { name: u.title }))

    await waitFor(() => expect(auto.set).toHaveBeenCalledWith(true))
    await waitFor(() => expect(screen.getByRole('switch', { name: u.title }).getAttribute('aria-checked')).toBe('true'))
  })

  it('shows the last automatic attempt while enabled', async () => {
    installBridge({
      enabled: true,
      supported: true,
      lastAttempt: { sessionKey: 'k', at: Date.now(), outcome: 'up-to-date' }
    })
    render(<AutoUpdateSetting />)

    expect(await screen.findByText(u.last(u.outcomes['up-to-date'], en.updates.justNow))).toBeTruthy()
  })

  it('is disabled with a platform note where unsupported', async () => {
    installBridge({ enabled: false, supported: false, lastAttempt: null })
    render(<AutoUpdateSetting />)

    const toggle = await screen.findByRole('switch', { name: u.title })

    expect(toggle.hasAttribute('disabled')).toBe(true)
    expect(screen.getByText(u.unsupported)).toBeTruthy()
  })

  it('renders nothing outside Electron', () => {
    const { container } = render(<AutoUpdateSetting />)

    expect(container.innerHTML).toBe('')
  })
})
