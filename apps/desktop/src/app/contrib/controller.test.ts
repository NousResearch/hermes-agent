import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@/contrib/plugins', () => ({ discoverBundledPlugins: vi.fn() }))
vi.mock('@/contrib/runtime-loader', () => ({ discoverRuntimePlugins: vi.fn() }))

afterEach(() => vi.unstubAllGlobals())

it('keeps the main layout set with onboarding disabled', async () => {
  vi.stubGlobal('hermesDesktop', { guestOnboardingEnabled: false })
  await import('./controller')
  const { registry } = await import('@/contrib/registry')
  const layouts = registry.getArea('layouts').map(preset => preset.id)
  expect(new Set(layouts)).toEqual(new Set(['default', 'focus', 'terminal-deck', 'quad']))
})
