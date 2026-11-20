import { afterEach, describe, expect, it, vi } from 'vitest'

import { createPluginContext } from './plugin'

const proposal = {
  actionId: 'doc-42:replace-section',
  revision: 'r7',
  profile: 'default',
  serviceInstance: 'docsvc@3f9a',
  nonce: 'n0nce-AAAAAAAAAAAAAAAA',
  title: 'Replace section 2'
}

const bridge = (confirmPluginAction?: unknown) => {
  ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = { confirmPluginAction }
}

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('ctx.confirmAction', () => {
  it('an undeclared plugin is refused and main is never asked', async () => {
    const confirm = vi.fn()
    bridge(confirm)

    const result = await createPluginContext('doc-sync').confirmAction(proposal)

    expect(result.status).toBe('refused')
    expect(confirm).not.toHaveBeenCalled()
  })

  it('a declared plugin forwards its own id and the proposal to main, and gets main’s answer', async () => {
    const answer = { status: 'approved', receipt: { payload: '{}', signature: 'sig', keyId: 'k' } }
    const confirm = vi.fn().mockResolvedValue(answer)
    bridge(confirm)

    const ctx = createPluginContext('doc-sync', undefined, ['confirm.action'])

    expect(await ctx.confirmAction(proposal)).toBe(answer)
    expect(confirm).toHaveBeenCalledWith('doc-sync', proposal)
  })

  it('an older shell without the bridge resolves unavailable instead of throwing', async () => {
    bridge(undefined)

    const ctx = createPluginContext('doc-sync', undefined, ['confirm.action'])

    expect((await ctx.confirmAction(proposal)).status).toBe('unavailable')
  })
})
