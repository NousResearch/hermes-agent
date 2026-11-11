import { afterEach, expect, it, vi } from 'vitest'

// Cold launches must re-evaluate persisted state; a static import would reuse
// the first launch's singleton and cannot exercise this module-loading boundary.

vi.mock('./desktop-metrics', () => ({
  DESKTOP_BUTTON_ACTIONS: { sidebarEnableSettings: 'enable', sidebarDisable: 'disable' },
  recordAction: vi.fn(),
  recordFeatureToggle: vi.fn()
}))

const legacyKey = 'hermes.desktop.simplifiedAgentMode.v1'
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
  localStorage.removeItem(legacyKey)
  localStorage.removeItem(shellKey)
  window.history.replaceState(null, '', '/')
  delete document.documentElement.dataset.shell
  vi.resetModules()
})

it.each([null, 'false', 'true'])('boots the portal shell and retires prototype preference %s', async legacy => {
  if (legacy !== null) {
    localStorage.setItem(legacyKey, legacy)
  }

  const { $portalPreferred, $portalShell } = await launch()
  expect($portalPreferred.get()).toBe(true)
  expect($portalShell.get()).toBe(true)
  expect(localStorage.getItem(legacyKey)).toBeNull()
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

it.each(['?win=hud', '?win=browser&tab=t1'])(
  'keeps the chromeless %s window classic without changing the preference',
  async search => {
    const hud = await launch(search)
    expect(hud.$portalPreferred.get()).toBe(true)
    expect(hud.$portalShell.get()).toBe(false)
    expect(paintedShell()).toBeNull()
  }
)
