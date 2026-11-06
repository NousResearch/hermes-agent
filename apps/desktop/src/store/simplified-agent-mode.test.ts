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

afterEach(() => {
  localStorage.removeItem(legacyKey)
  localStorage.removeItem(shellKey)
  vi.resetModules()
})

it.each([null, 'false', 'true'])('boots the normal shell with prototype preference %s', async legacy => {
  vi.resetModules()
  localStorage.removeItem(shellKey)
  if (legacy === null) {
    localStorage.removeItem(legacyKey)
  } else {
    localStorage.setItem(legacyKey, legacy)
  }

  const { $simplifiedAgentMode } = await import('./simplified-agent-mode')
  expect($simplifiedAgentMode.get()).toBe(true)
  expect(localStorage.getItem(legacyKey)).toBe(legacy)
})

it('keeps an explicit classic rollback across launches and can restore the normal shell', async () => {
  vi.resetModules()
  localStorage.removeItem(shellKey)
  const first = await import('./simplified-agent-mode')
  first.setSimplifiedAgentMode(false)
  vi.resetModules()
  const next = await import('./simplified-agent-mode')
  expect(next.$simplifiedAgentMode.get()).toBe(false)
  next.setSimplifiedAgentMode(true)
  vi.resetModules()
  const restored = await import('./simplified-agent-mode')
  expect(restored.$simplifiedAgentMode.get()).toBe(true)
})
