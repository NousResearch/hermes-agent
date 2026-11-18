import { afterEach, expect, it, vi } from 'vitest'

// Cold launches must re-evaluate persisted state; a static import would reuse
// the first launch's singleton and cannot exercise this module-loading boundary.

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

it('boots the classic shell when no choice is stored', async () => {
  const { $portalPreferred, $portalShell } = await launch()
  expect($portalPreferred.get()).toBe(false)
  expect($portalShell.get()).toBe(false)
  expect(paintedShell()).toBeNull()
})

it('keeps an explicit portal choice across launches and can roll it back', async () => {
  const first = await launch()
  first.$shellChoice.set(true)
  const next = await launch()
  expect(next.$portalPreferred.get()).toBe(true)
  next.$shellChoice.set(false)
  const restored = await launch()
  expect(restored.$portalPreferred.get()).toBe(false)
})

it('marks <html> on load and follows runtime toggles both ways; classic leaves no shell attribute', async () => {
  localStorage.setItem(shellKey, 'true')
  const portal = await launch()
  expect(paintedShell()).toBe('portal')
  portal.$shellChoice.set(false)
  expect(paintedShell()).toBeNull()
  portal.$shellChoice.set(true)
  expect(paintedShell()).toBe('portal')

  localStorage.removeItem(shellKey)
  await launch()
  expect(paintedShell()).toBeNull()
})

it('keeps the popped-out Browser window classic without changing the preference', async () => {
  localStorage.setItem(shellKey, 'true')
  const browser = await launch('?win=browser&tab=t1')
  expect(browser.$portalPreferred.get()).toBe(true)
  expect(browser.$portalShell.get()).toBe(false)
  expect(paintedShell()).toBeNull()
})

it('renders the redesign in the chromeless HUD window, and classic there once it is rolled back', async () => {
  localStorage.setItem(shellKey, 'true')
  const hud = await launch('?win=hud')
  expect(hud.$portalShell.get()).toBe(true)
  expect(paintedShell()).toBe('portal')

  hud.$shellChoice.set(false)
  expect(hud.$portalShell.get()).toBe(false)
  expect(paintedShell()).toBeNull()
})
