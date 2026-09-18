import { describe, expect, it, vi } from 'vitest'

import { fetchAgentPluginDesktopHalf } from './agent-plugins'

describe('fetchAgentPluginDesktopHalf', () => {
  it('asks plugins.manage for the desktop half by key, scoped to the profile', async () => {
    const request = vi.fn(async () => ({ bytes: 12, key: 'mission-control', name: 'mission-control', ok: true, sha256: 'abc', source: 'git', text: 'export default {}' }))

    const result = await fetchAgentPluginDesktopHalf(request as never, {
      key: 'mission-control',
      name: 'mission-control',
      profile: 'work'
    })

    expect(request).toHaveBeenCalledWith('plugins.manage', {
      action: 'desktop_half',
      key: 'mission-control',
      name: 'mission-control',
      profile: 'work'
    })
    expect(result).toMatchObject({ ok: true, sha256: 'abc', text: 'export default {}' })
  })

  it('requires an identifier and never calls the backend without one', async () => {
    const request = vi.fn()

    const result = await fetchAgentPluginDesktopHalf(request as never, { name: '  ' })

    expect(result.ok).toBe(false)
    expect(request).not.toHaveBeenCalled()
  })

  it('treats an ok response with no text as no half at all', async () => {
    const request = vi.fn(async () => ({ ok: true }))

    const result = await fetchAgentPluginDesktopHalf(request as never, { name: 'mission-control' })

    expect(result).toMatchObject({ ok: false, error: 'the backend returned no desktop half' })
  })

  it('tells the user to update a backend that predates the action', async () => {
    const request = vi.fn(async () => {
      throw new Error('unknown plugins action: desktop_half')
    })

    const result = await fetchAgentPluginDesktopHalf(request as never, { name: 'mission-control' })

    expect(result.ok).toBe(false)
    expect(result.error).toMatch(/update Hermes on that machine/)
  })

  it('relays any other backend error verbatim', async () => {
    const request = vi.fn(async () => {
      throw new Error("'mission-control' ships no desktop half")
    })

    const result = await fetchAgentPluginDesktopHalf(request as never, { name: 'mission-control' })

    expect(result.error).toBe("'mission-control' ships no desktop half")
  })
})
