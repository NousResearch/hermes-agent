import { afterEach, expect, it, vi } from 'vitest'

// Cold launches must re-evaluate persisted state; a static import would reuse
// the first launch's singleton and cannot exercise this module-loading boundary.

vi.mock('./desktop-metrics', () => ({
  DESKTOP_BUTTON_ACTIONS: { sidebarEnableSettings: 'enable', sidebarDisable: 'disable' },
  recordAction: vi.fn(),
  recordFeatureToggle: vi.fn()
}))

const shellKey = 'hermes.desktop.agentShell.v1'
const paintedShell = () => document.documentElement.getAttribute('data-shell')

let uninstall: (() => void) | undefined

async function launch(search = '') {
  uninstall?.()
  vi.resetModules()
  window.history.replaceState(null, '', `/${search}`)
  const shell = await import('./shell-variant')
  uninstall = shell.installShellAttribute()

  return shell
}

afterEach(() => {
  uninstall?.()
  uninstall = undefined
  localStorage.removeItem(shellKey)
  window.history.replaceState(null, '', '/')
  delete document.documentElement.dataset.shell
  vi.resetModules()
})

it('boots the portal shell when no choice is stored', async () => {
  const { $portalPreferred, $portalShell } = await launch()
  expect($portalPreferred.get()).toBe(true)
  expect($portalShell.get()).toBe(true)
})

it('keeps an explicit classic rollback across launches and can restore the portal shell', async () => {
  const first = await launch()
  first.setPortalPreferred(false)
  const next = await launch()
  expect(next.$portalPreferred.get()).toBe(false)
  next.setPortalPreferred(true)
  const restored = await launch()
  expect(restored.$portalPreferred.get()).toBe(true)
})

it('marks <html> on load and follows runtime toggles both ways; classic leaves no shell attribute', async () => {
  localStorage.setItem(shellKey, 'false')
  const classic = await launch()
  expect(paintedShell()).toBeNull()
  classic.setPortalPreferred(true)
  expect(paintedShell()).toBe('portal')
  classic.setPortalPreferred(false)
  expect(paintedShell()).toBeNull()

  localStorage.removeItem(shellKey)
  await launch()
  expect(paintedShell()).toBe('portal')
})

it('keeps the popped-out Browser window classic without changing the preference', async () => {
  const browser = await launch('?win=browser&tab=t1')
  expect(browser.$portalPreferred.get()).toBe(true)
  expect(browser.$portalShell.get()).toBe(false)
  expect(paintedShell()).toBeNull()
})

it('renders the redesign in the chromeless HUD window, and classic there once it is rolled back', async () => {
  const hud = await launch('?win=hud')
  expect(hud.$portalShell.get()).toBe(true)
  expect(paintedShell()).toBe('portal')

  hud.setPortalPreferred(false)
  expect(hud.$portalShell.get()).toBe(false)
  expect(paintedShell()).toBeNull()
})
