import { afterEach, describe, expect, it, vi } from 'vitest'

const proposal = {
  actionId: 'doc-42:replace-section',
  revision: 'r7',
  profile: 'default',
  serviceInstance: 'docsvc@3f9a',
  nonce: 'n0nce-AAAAAAAAAAAAAAAA',
  title: 'Replace section 2'
}

/** The preload's one-shot door, then a fresh load of the plugin host. */
async function loadHost(door?: unknown) {
  let claimed = false

  const claimPluginConfirmation =
    door === undefined
      ? undefined
      : () => {
          if (claimed) {
            return null
          }

          claimed = true

          return door
        }

  ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = { claimPluginConfirmation }
  vi.resetModules()

  return import('./plugin')
}

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('ctx.confirmAction', () => {
  it('an undeclared plugin is refused and main is never asked', async () => {
    const door = vi.fn()
    const { createPluginContext } = await loadHost(door)

    const result = await createPluginContext('doc-sync').confirmAction(proposal)

    expect(result.status).toBe('refused')
    expect(door).not.toHaveBeenCalled()
  })

  it('plugin code cannot reach main around its ctx: the host already holds the door', async () => {
    const door = vi.fn()
    await loadHost(door)

    expect(window.hermesDesktop?.claimPluginConfirmation?.()).toBeNull()
  })

  it('a declared plugin forwards its own id and the proposal to main, and gets main’s answer', async () => {
    const answer = { status: 'approved', receipt: { payload: '{}', signature: 'sig', keyId: 'k' } }
    const door = vi.fn().mockResolvedValue(answer)
    const { createPluginContext } = await loadHost(door)

    const ctx = createPluginContext('doc-sync', undefined, ['confirm.action'])

    expect(await ctx.confirmAction(proposal)).toBe(answer)
    expect(door).toHaveBeenCalledWith('doc-sync', proposal)
  })

  it('an older shell without the door resolves unavailable instead of throwing', async () => {
    const { createPluginContext } = await loadHost()

    const ctx = createPluginContext('doc-sync', undefined, ['confirm.action'])

    expect((await ctx.confirmAction(proposal)).status).toBe('unavailable')
  })
})
