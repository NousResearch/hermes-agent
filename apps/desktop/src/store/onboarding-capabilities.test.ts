import { afterEach, expect, it, vi } from 'vitest'

import { buildFirstTaskSeedMessages } from '@/components/onboarding-chat/setup-profile'
import { DEFAULT_ANSWERS } from '@/store/onboarding-answers'
import { readOnboardingCapabilities } from '@/store/onboarding-capabilities'
import { buildChatOnboardingSeedMessages } from '@/store/onboarding-script'

const api = vi.fn()

it('carries fresh catalog evidence into the guide and the exact working-profile handoff seed', async () => {
  vi.stubGlobal('window', { hermesDesktop: { api } })

  const entry = {
    name: 'future-studio', auth_type: 'none', installed: false, enabled: false,
    detected_apps: ['Future Studio'],
    suggest: { keywords: ['future studio'], applications: ['Future Studio'], examples: ['Make a scene in Future Studio'], requires_app: true }
  }

  api.mockResolvedValue({ entries: [entry], diagnostics: [], discovery: { scope: 'backend', status: 'ok', platform: 'darwin' } })
  const scope = { connectionId: 'remote-studio', profile: 'default' }
  const evidence = await readOnboardingCapabilities(scope)
  const guide = buildChatOnboardingSeedMessages('Hi', true, evidence)
  expect(guide[0].content).toContain('"name":"future-studio"')
  expect(guide[0].content).toContain('"readiness":"setup_required"')
  expect(guide[0].display_kind).toBe('hidden')

  const handoff = await buildFirstTaskSeedMessages('Make a scene in Future Studio', DEFAULT_ANSWERS, 'build', scope)
  expect(handoff[0].content).toContain('"name":"future-studio"')
  expect(handoff[0].content).toContain('not necessarily the desktop computer')
  expect(api).toHaveBeenLastCalledWith(expect.objectContaining(scope))
  expect(api.mock.calls.every(([request]) => request.path === '/api/mcp/catalog?detect_apps=true')).toBe(true)
})

afterEach(() => {
  vi.unstubAllGlobals()
  api.mockReset()
})

it('reads only the pinned backend and degrades safely on old or unavailable discovery', async () => {
  vi.stubGlobal('window', { hermesDesktop: { api } })
  const scope = { connectionId: 'remote-studio', profile: 'hermes-setup' }
  api.mockResolvedValueOnce({ entries: [], diagnostics: [] })
  expect(await readOnboardingCapabilities(scope)).toBe('')
  expect(api).toHaveBeenCalledWith(expect.objectContaining({
    connectionId: scope.connectionId,
    profile: scope.profile,
    path: '/api/mcp/catalog?detect_apps=true',
    timeoutMs: 5000
  }))

  api.mockRejectedValueOnce(new Error('Offline'))
  expect(await readOnboardingCapabilities(scope)).toBe('')
  expect(api).toHaveBeenCalledTimes(2)
})
