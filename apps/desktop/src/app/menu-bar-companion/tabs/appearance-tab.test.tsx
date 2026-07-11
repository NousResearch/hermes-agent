// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const setMode = vi.fn()
const setTheme = vi.fn()
const setTranslucency = vi.fn()
const setMenuBarTransparency = vi.fn()
const setZoomPercent = vi.fn()
const request = vi.fn()
const retry = vi.fn()
const broadcastDesktopStateChange = vi.fn()

const stores = vi.hoisted(() => ({
  activeProfile: null as any,
  menuBarTransparency: null as any,
  translucency: null as any,
  zoomPercent: null as any
}))

const gateway = vi.hoisted(() => ({ ready: true, error: '' }))

vi.mock('@/themes', () => ({
  useTheme: () => ({
    mode: 'dark',
    resolvedMode: 'dark',
    themeName: 'nous',
    availableThemes: [
      { name: 'nous', label: 'Nous' },
      { name: 'cyberpunk', label: 'Cyberpunk' }
    ],
    setMode,
    setTheme
  })
}))

vi.mock('@/store/translucency', async () => {
  const { atom } = await import('nanostores')
  stores.translucency = atom(35)

  return {
    $translucency: stores.translucency,
    setTranslucency: (value: number) => setTranslucency(value)
  }
})

vi.mock('@/store/menu-bar-transparency', async () => {
  const { atom } = await import('nanostores')
  stores.menuBarTransparency = atom(20)

  return {
    $menuBarTransparency: stores.menuBarTransparency,
    setMenuBarTransparency: (value: number) => setMenuBarTransparency(value)
  }
})

vi.mock('@/store/zoom', async () => {
  const { atom } = await import('nanostores')
  stores.zoomPercent = atom(100)

  return {
    $zoomPercent: stores.zoomPercent,
    setZoomPercent: (value: number) => setZoomPercent(value)
  }
})

vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')
  stores.activeProfile = atom('work')

  return {
    $activeGatewayProfile: stores.activeProfile,
    normalizeProfileKey: (value: string) => value || 'default'
  }
})

vi.mock('@/lib/desktop-state-sync', () => ({
  broadcastDesktopStateChange: (domain: string, options?: unknown) => broadcastDesktopStateChange(domain, options),
  onDesktopStateSync: () => () => {}
}))

vi.mock('../hooks/use-companion-gateway', () => ({
  useCompanionGateway: () => ({ ready: gateway.ready, error: gateway.error, request, retry })
}))

import { AppearanceTab } from './appearance-tab'

beforeEach(() => {
  gateway.ready = true
  gateway.error = ''
  request.mockImplementation(async method => {
    if (method === 'pet.info') {
      return { enabled: true, slug: 'boba', displayName: 'Boba' }
    }

    if (method === 'pet.gallery') {
      return {
        active: 'boba',
        enabled: true,
        pets: [
          { slug: 'boba', displayName: 'Boba', installed: true },
          { slug: 'pixel', displayName: 'Pixel', installed: true }
        ]
      }
    }

    return { ok: true }
  })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      focusMainWindow: vi.fn().mockResolvedValue({ ok: true }),
      settings: {
        getMenuBarCompanionEnabled: vi.fn().mockResolvedValue({ enabled: true }),
        setMenuBarCompanionEnabled: vi.fn(async enabled => ({ enabled }))
      }
    }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('AppearanceTab Desktop integration', () => {
  it('controls shared appearance, pet, Desktop focus, and tray state without exposing the gateway-only PII setting', async () => {
    render(<AppearanceTab />)
    await screen.findByRole('button', { name: 'Turn pet off' })

    expect(screen.queryByText(/Privacy \(local\)/)).toBeNull()
    expect(screen.queryByText(/Redact PII/)).toBeNull()
    expect(screen.getByText(/Active profile: work/)).toBeTruthy()

    fireEvent.change(screen.getByLabelText('Desktop mode'), { target: { value: 'system' } })
    expect(setMode).toHaveBeenCalledWith('system')

    fireEvent.change(screen.getByLabelText('Menu bar and Desktop color scheme'), {
      target: { value: 'cyberpunk' }
    })
    expect(setTheme).toHaveBeenCalledWith('cyberpunk')

    fireEvent.change(screen.getByLabelText(/Desktop window transparency/), { target: { value: '50' } })
    expect(setTranslucency).toHaveBeenCalledWith(50)

    fireEvent.change(screen.getByLabelText(/Menu bar window transparency/), { target: { value: '45' } })
    expect(setMenuBarTransparency).toHaveBeenCalledWith(45)

    fireEvent.change(screen.getByLabelText('Desktop UI scale'), { target: { value: '125' } })
    expect(setZoomPercent).toHaveBeenCalledWith(125)

    fireEvent.click(screen.getByRole('button', { name: 'Open Hermes Desktop' }))
    expect(window.hermesDesktop.focusMainWindow).toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Next pet' }))
    await waitFor(() => expect(request).toHaveBeenCalledWith('pet.select', { profile: 'work', slug: 'pixel' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next pet' }).hasAttribute('disabled')).toBe(false))

    fireEvent.click(screen.getByRole('button', { name: 'Prev pet' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Prev pet' }).hasAttribute('disabled')).toBe(false))

    fireEvent.click(screen.getByRole('button', { name: 'Pixel' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Pixel' }).hasAttribute('disabled')).toBe(false))

    fireEvent.click(screen.getByRole('button', { name: 'Turn pet off' }))
    await waitFor(() => expect(request).toHaveBeenCalledWith('pet.disable', { profile: 'work' }))
    expect(broadcastDesktopStateChange).toHaveBeenCalledWith('pet', { profile: 'work' })

    fireEvent.click(screen.getByLabelText('Show menu bar companion'))
    await waitFor(() => expect(window.hermesDesktop.settings.setMenuBarCompanionEnabled).toHaveBeenCalledWith(false))
  })

  it('does not offer an invalid pet action when no pet is installed', async () => {
    request.mockImplementation(async method => {
      if (method === 'pet.info') {
        return { enabled: false }
      }

      if (method === 'pet.gallery') {
        return { enabled: false, pets: [] }
      }

      return { ok: true }
    })

    render(<AppearanceTab />)

    const toggle = await screen.findByRole('button', { name: 'Turn pet on' })
    expect(toggle.hasAttribute('disabled')).toBe(true)
    expect(screen.getByText(/No installed pets/)).toBeTruthy()
  })

  it('enables the active installed pet', async () => {
    request.mockImplementation(async method => {
      if (method === 'pet.info') {
        return { enabled: false, slug: 'boba', displayName: 'Boba' }
      }

      if (method === 'pet.gallery') {
        return {
          active: 'boba',
          enabled: false,
          pets: [{ slug: 'boba', displayName: 'Boba', installed: true }]
        }
      }

      return { ok: true }
    })

    render(<AppearanceTab />)
    fireEvent.click(await screen.findByRole('button', { name: 'Turn pet on' }))

    await waitFor(() => expect(request).toHaveBeenCalledWith('pet.select', { profile: 'work', slug: 'boba' }))
  })

  it('retries a failed gateway connection', async () => {
    gateway.ready = false
    gateway.error = 'Gateway offline'

    render(<AppearanceTab />)
    fireEvent.click(await screen.findByRole('button', { name: 'Retry gateway' }))

    expect(retry).toHaveBeenCalled()
  })
})
