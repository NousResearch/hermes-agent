import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  checkHermesUpdate,
  getActionStatus,
  getStatus,
  getStudySettings,
  restartGateway,
  setApiRequestProfile,
  updateHermes,
  updateStudySettings
} from './hermes'

// Contract: every backend-targeted action helper must carry the active gateway
// profile, so a multi-profile / global-remote user's restart, status poll, and
// update hit the backend they're actually on — not the primary/default. The
// System-panel "restart does nothing" bug was these helpers dropping it.
describe('backend action helpers are profile-scoped', () => {
  const api = vi.fn(async (_req: { path: string; profile?: string }) => ({}) as never)

  beforeEach(() => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { api }
    api.mockClear()
  })

  afterEach(() => {
    setApiRequestProfile(null)
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  const lastProfile = () => api.mock.calls.at(-1)?.[0].profile

  it('omits profile when none is active (single-profile users unaffected)', () => {
    void getStatus()
    expect(lastProfile()).toBeUndefined()
  })

  it('forwards the active profile to every backend action', () => {
    setApiRequestProfile('coder')

    void getStatus()
    void restartGateway()
    void updateHermes()
    void checkHermesUpdate()
    void getActionStatus('gateway-restart')

    for (const call of api.mock.calls) {
      expect(call[0].profile).toBe('coder')
    }
  })
})

describe('StudyOS settings are profile-scoped', () => {
  const vaults = new Map<string, string>()

  const api = vi.fn(
    async (req: { body?: { vault_path?: string }; method?: string; path: string; profile?: string }) => {
      const profile = req.profile ?? 'default'

      if (req.method === 'PUT' && req.body?.vault_path) {
        vaults.set(profile, req.body.vault_path)
      }

      return {
        configured: vaults.has(profile),
        vault_path: vaults.get(profile) ?? null
      } as never
    }
  )

  beforeEach(() => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { api }
    vaults.clear()
    api.mockClear()
  })

  afterEach(() => {
    setApiRequestProfile(null)
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('keeps vault changes isolated when switching Hermes profiles', async () => {
    setApiRequestProfile('study-a')
    await updateStudySettings('/vaults/study-a')

    setApiRequestProfile('study-b')
    await updateStudySettings('/vaults/study-b')

    setApiRequestProfile('study-a')
    const studyA = await getStudySettings()
    setApiRequestProfile('study-b')
    const studyB = await getStudySettings()

    expect(studyA.vault_path).toBe('/vaults/study-a')
    expect(studyB.vault_path).toBe('/vaults/study-b')
    expect(api.mock.calls.every(([request]) => request.path.startsWith('/api/plugins/study_os/'))).toBe(true)
  })
})
