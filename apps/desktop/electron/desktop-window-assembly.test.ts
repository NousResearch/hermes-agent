import { expect, test, vi } from 'vitest'

const calls = vi.hoisted(() => [] as string[])

vi.mock('./desktop-window-wiring-runtime', () => ({ createDesktopWindowWiringRuntime: () => {
  calls.push('wiring')

  return { wireCommonWindowHandlers: vi.fn(), installPreviewGuestPreload: vi.fn(), wireWindowReveal: vi.fn() }
} }))
vi.mock('./desktop-secondary-window-runtime', () => ({ createDesktopSecondaryWindowRuntime: () => {
  calls.push('secondary')

  return { minimizeToTray: vi.fn(), focusWindow: vi.fn(), createSessionWindow: vi.fn(),
    createBrowserWindow: vi.fn(), createInstanceWindow: vi.fn() }
} }))
vi.mock('./wake-indicator-window', () => ({ createWakeIndicatorWindowController: () => {
  calls.push('wake')

  return {}
} }))
vi.mock('./intro-reveal-window', () => ({ createIntroRevealWindowController: () => {
  calls.push('intro')

  return {}
} }))
vi.mock('./chat-onboarding-window', () => ({ registerChatOnboardingWindow: () => calls.push('onboarding') }))
vi.mock('./desktop-pet-overlay-runtime', () => ({ createDesktopPetOverlayRuntime: () => {
  calls.push('pet')

  return { getPetOverlayWindow: vi.fn(), openPetOverlay: vi.fn(), closePetOverlay: vi.fn() }
} }))
vi.mock('./desktop-shell-overlay-runtime', () => ({ createDesktopShellOverlayRuntime: () => {
  calls.push('shell')

  return { applyQuickEntrySettings: vi.fn(), closeHudWindow: vi.fn(), closeQuickEntryWindow: vi.fn(),
    hideQuickEntryWindow: vi.fn(), openHudWindow: vi.fn(), readQuickEntrySettings: vi.fn(),
    resetHudWindowLayout: vi.fn(), writeQuickEntrySettings: vi.fn() }
} }))
vi.mock('./desktop-primary-window-runtime', () => ({ createDesktopPrimaryWindowRuntime: () => {
  calls.push('primary')

  return { createWindow: () => 'primary-window' }
} }))

import { createDesktopWindowAssembly } from './desktop-window-assembly'

test('window owners assemble in the original reveal and primary-window order', () => {
  const deps = new Proxy({}, { get: () => vi.fn() })
  const windows = createDesktopWindowAssembly(deps as any)

  expect(calls).toEqual(['wiring', 'secondary', 'wake', 'intro', 'onboarding', 'pet', 'shell', 'primary'])
  expect(windows.createWindow()).toBe('primary-window')
})
