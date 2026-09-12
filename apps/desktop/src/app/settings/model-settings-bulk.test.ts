import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getProfiles, getGlobalModelInfo, getGlobalModelOptions, setMainModelAssignment } = vi.hoisted(() => ({
  getProfiles: vi.fn(),
  getGlobalModelInfo: vi.fn(),
  getGlobalModelOptions: vi.fn(),
  setMainModelAssignment: vi.fn()
}))

vi.mock('@/hermes', () => ({ getProfiles, getGlobalModelInfo, getGlobalModelOptions }))
vi.mock('@/store/cron-model-impact', () => ({ setMainModelAssignment }))
vi.mock('@/store/notifications', () => ({ readableError: (error: Error) => ({ message: error.message }) }))

import { applyModelToProfiles } from './model-settings-bulk'

const options = () => ({
  connectionId: 'remote-1', provider: 'custom:team', model: 'shared-model',
  isCurrent: () => true, onResult: vi.fn(), unavailable: 'Not configured',
  interrupted: 'Scope changed', failed: 'Save failed'
})

beforeEach(() => {
  vi.resetAllMocks()
  getProfiles.mockResolvedValue({ profiles: [{ name: 'default' }, { name: 'writer' }] })
  getGlobalModelOptions.mockImplementation((_, scope) => Promise.resolve({
    providers: [{ slug: 'custom:team', authenticated: true, models: ['shared-model'], api_url: `https://${scope.profile}.example/v1` }]
  }))
  getGlobalModelInfo.mockResolvedValue({ provider: 'custom:team', model: 'shared-model' })
  setMainModelAssignment.mockResolvedValue({ ok: true, provider: 'custom:team', model: 'shared-model' })
})

describe('one-time model assignment to profiles', () => {
  it('pins requests and uses each target endpoint without copying unrelated config or credentials', async () => {
    const result = await applyModelToProfiles(options())
    expect(getProfiles).toHaveBeenCalledWith({ connectionId: 'remote-1' })
    expect(result).toEqual([{ profile: 'default', ok: true }, { profile: 'writer', ok: true }])

    for (const [index, profile] of ['default', 'writer'].entries()) {
      expect(getGlobalModelOptions).toHaveBeenNthCalledWith(index + 1, undefined, { connectionId: 'remote-1', profile })
      expect(setMainModelAssignment).toHaveBeenNthCalledWith(index + 1,
        { provider: 'custom:team', model: 'shared-model', base_url: `https://${profile}.example/v1` },
        { connectionId: 'remote-1', profile }, { skipConfirmPrompt: true })
    }
  })

  it('reports per-profile failure and continues after missing providers or a failed save', async () => {
    getProfiles.mockResolvedValue({ profiles: [{ name: 'missing' }, { name: 'failed' }, { name: 'working' }] })
    getGlobalModelOptions.mockResolvedValueOnce({ providers: [] })
    setMainModelAssignment.mockRejectedValueOnce(new Error('Not authorized'))
    const settings = options()
    expect(await applyModelToProfiles(settings)).toEqual([
      { profile: 'missing', ok: false, error: 'Not configured' },
      { profile: 'failed', ok: false, error: 'Not authorized' },
      { profile: 'working', ok: true }
    ])
    expect(setMainModelAssignment).toHaveBeenCalledTimes(2)
    expect(settings.onResult).toHaveBeenCalledTimes(3)
  })

  it('reports an unverified write when readback differs, rather than claiming success', async () => {
    getGlobalModelInfo.mockResolvedValue({ provider: 'custom:team', model: 'old-model' })
    const result = await applyModelToProfiles(options())
    expect(result.every(row => !row.ok && row.error === 'Save failed')).toBe(true)
  })

  it('never writes when scope changes during a catalog read', async () => {
    let current = true
    getGlobalModelOptions.mockImplementation(async () => {
      current = false

      return { providers: [{ slug: 'custom:team', authenticated: true, models: ['shared-model'] }] }
    })
    const result = await applyModelToProfiles({ ...options(), isCurrent: () => current })
    expect(setMainModelAssignment).not.toHaveBeenCalled()
    expect(getGlobalModelOptions).toHaveBeenCalledTimes(1)
    expect(result.every(row => !row.ok && row.error === 'Scope changed')).toBe(true)
  })

  it('keeps completed writes and stops remaining profiles when scope changes during a save', async () => {
    let current = true
    setMainModelAssignment.mockImplementation(async () => {
      current = false

      return { ok: true }
    })
    expect(await applyModelToProfiles({ ...options(), isCurrent: () => current })).toEqual([
      { profile: 'default', ok: true }, { profile: 'writer', ok: false, error: 'Scope changed' }
    ])
    expect(setMainModelAssignment).toHaveBeenCalledTimes(1)
  })

  it.each([
    { slug: 'custom:team', authenticated: false, models: ['shared-model'] },
    { slug: 'custom:team', authenticated: true, models: ['other-model'] }
  ])('skips unconfigured provider/model', async provider => {
    getGlobalModelOptions.mockResolvedValue({ providers: [provider] })
    const result = await applyModelToProfiles(options())
    expect(setMainModelAssignment).not.toHaveBeenCalled()
    expect(result.every(row => !row.ok)).toBe(true)
  })
})
