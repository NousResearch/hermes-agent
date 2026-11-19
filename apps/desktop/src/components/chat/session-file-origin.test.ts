import { beforeEach, describe, expect, it, vi } from 'vitest'

const resolveSessionOwner = vi.fn()

vi.mock('@/app/session/hooks/use-session-actions/utils', () => ({
  resolveSessionOwner: (id: string) => resolveSessionOwner(id)
}))

const { sessionFileOrigin } = await import('./session-file-origin')

describe('sessionFileOrigin', () => {
  beforeEach(() => resolveSessionOwner.mockReset())

  it("names the stored session's owning profile so the gateway finds the session", async () => {
    resolveSessionOwner.mockResolvedValue('finance')

    expect(await sessionFileOrigin('local-abc', 'rt-1')).toEqual({ profile: 'finance', sessionId: 'local-abc' })
    expect(resolveSessionOwner).toHaveBeenCalledWith('local-abc')
  })

  it('pins a connection-routed owner as the exact owner route', async () => {
    resolveSessionOwner.mockResolvedValue({ connectionId: 'c1', profile: 'finance' })

    expect(await sessionFileOrigin('local-abc', null)).toEqual({
      owner: { connectionId: 'c1', profile: 'finance' },
      sessionId: 'local-abc'
    })
  })

  it('falls back to the runtime session alone before the chat is stored', async () => {
    expect(await sessionFileOrigin(null, 'rt-1')).toEqual({ sessionId: 'rt-1' })
    expect(resolveSessionOwner).not.toHaveBeenCalled()
  })

  it('has no origin without a session', async () => {
    expect(await sessionFileOrigin(null, null)).toBeUndefined()
  })
})
