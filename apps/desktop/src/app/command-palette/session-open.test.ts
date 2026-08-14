import { beforeEach, describe, expect, it, vi } from 'vitest'

const { openSession, resolveSessionProfile } = vi.hoisted(() => ({
  openSession: vi.fn(),
  resolveSessionProfile: vi.fn()
}))

vi.mock('../open-session', async () => ({
  ...(await vi.importActual<Record<string, unknown>>('../open-session')),
  openSession
}))

vi.mock('../session/hooks/use-session-actions/utils', () => ({ resolveSessionProfile }))

import { openCommandPaletteSession } from './session-open'

describe('openCommandPaletteSession', () => {
  const navigate = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('passes a listed session owner to a modifier-opened window', async () => {
    await openCommandPaletteSession('session-1', 'life', { metaKey: true, shiftKey: true }, navigate)

    expect(resolveSessionProfile).not.toHaveBeenCalled()
    expect(openSession).toHaveBeenCalledWith('session-1', navigate, 'window', 'life')
  })

  it('resolves a direct session id owner before opening a window', async () => {
    resolveSessionProfile.mockResolvedValue('life')

    await openCommandPaletteSession('session-2', undefined, { metaKey: true, shiftKey: true }, navigate)

    expect(resolveSessionProfile).toHaveBeenCalledWith('session-2')
    expect(openSession).toHaveBeenCalledWith('session-2', navigate, 'window', 'life')
  })

  it('does not delay an in-workspace open for profile discovery', async () => {
    await openCommandPaletteSession('session-3', undefined, undefined, navigate)

    expect(resolveSessionProfile).not.toHaveBeenCalled()
    expect(openSession).toHaveBeenCalledWith('session-3', navigate, 'stack', undefined)
  })
})
