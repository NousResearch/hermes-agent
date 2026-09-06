/**
 * `applyAdvancedConfig` — the advanced editor's save, which persists ONLY the
 * dirty sections and reports per-section outcomes back to the dialog.
 *
 * Two contracts live here:
 *
 *  - **Inherit clears the pin.** A dirty model section with both fields empty
 *    means "inherit the gateway default", which is a `config unset model`, not
 *    a configure payload. Its success/failure has to merge into the same
 *    `applied` map as the RPC sections, or the dialog toasts a contradictory
 *    "saved" over a failed clear.
 *  - **#95293 on this surface.** The gateway now guards data-policy /
 *    expensive models for `profiles.configure` too, answering
 *    `confirm_required` + `confirm_message`. Bots does NOT route through
 *    use-model-controls, so before the fix that response matched no branch and
 *    the pick silently dropped. It must go through the SAME shared handler the
 *    core picker uses (one applier, no forked confirm UI), must not count as a
 *    FAILED section while it is merely pending, and Confirm must resend ONLY
 *    the model section with `confirm_expensive_model: true`.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { applyAdvancedConfig, emptyAdvancedState } from './profile-config'
import type { RosterRow } from './types'

type AdvancedConfigState = ReturnType<typeof emptyAdvancedState>

/** The payload the shared core confirm flow receives — the same object the
 *  model picker hands it. */
interface ModelSwitchConfirmArgs {
  confirmMessage: string
  finish: () => void
  requestConfirmed: () => Promise<{ applied?: Record<string, boolean> } | undefined>
}

const { confirmMock, hostMock, invalidateMock } = vi.hoisted(() => ({
  confirmMock: vi.fn((_args: ModelSwitchConfirmArgs) => 'notification-1'),
  hostMock: {
    getGateway: () => 'ambient-gateway',
    request: vi.fn(),
    requestProfile: vi.fn(),
    state: { connectionId: { get: () => 'local' }, gateway: { get: () => 'open' }, profile: { get: () => 'default' } }
  },
  invalidateMock: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    Button: () => null,
    Checkbox: () => null,
    GlyphSpinner: () => null,
    host: hostMock,
    Input: () => null,
    McpTab: undefined,
    queryClient: { invalidateQueries: invalidateMock },
    ScrollArea: () => null,
    Select: () => null,
    SelectContent: () => null,
    SelectItem: () => null,
    SelectTrigger: () => null,
    SelectValue: () => null,
    SkillsView: undefined,
    surfaceModelSwitchConfirm: confirmMock,
    Textarea: () => null,
    ToolsetConfigPanel: undefined,
    useQuery: vi.fn(() => ({ data: undefined, error: null, isLoading: false })),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))
// The SOUL protocol append has its own suite; here it must not rewrite the
// text under the assertions.
vi.mock('./soul', () => ({ ensureMessagingProtocol: (soul: string) => soul }))

const bot = { name: 'zeta' } as RosterRow

const dirtyModel = (patch: Partial<AdvancedConfigState> = {}): AdvancedConfigState => ({
  ...emptyAdvancedState(),
  dirtyModel: true,
  loaded: true,
  model: 'muse-spark-1.2-contributor',
  provider: 'opencode-go',
  ...patch
})

/** Every RPC in order, with params frozen at call time. */
const routed: Array<{ method: string; params: Record<string, unknown> }> = []

function respondWith(handler: (method: string, params: Record<string, unknown>) => unknown) {
  hostMock.request.mockImplementation(async (method: string, params: Record<string, unknown>) => {
    routed.push({ method, params: structuredClone(params ?? {}) })

    return handler(method, params)
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  routed.length = 0
  respondWith(() => ({}))
})

describe('selecting Inherit', () => {
  it('clears the profile model assignment through the CLI', async () => {
    respondWith(() => ({ blocked: false, code: 0, output: 'Unset model' }))

    const result = await applyAdvancedConfig(bot, dirtyModel({ model: '', provider: '' }))

    expect(routed).toEqual([
      { method: 'cli.exec', params: { argv: ['--profile', 'zeta', 'config', 'unset', 'model'] } }
    ])
    expect(result).toEqual({ applied: { model: true }, ok: true })
  })

  it('merges the clear with the other dirty sections’ outcomes', async () => {
    respondWith(method => (method === 'cli.exec' ? { blocked: false, code: 0 } : { applied: { soul: true } }))

    const result = await applyAdvancedConfig(
      bot,
      dirtyModel({ dirtySoul: true, model: '', provider: '', soul: '# Ops' })
    )

    expect(routed.map(call => call.method)).toEqual(['cli.exec', 'profiles.configure'])
    expect(routed[1].params).toEqual({ name: 'zeta', soul: '# Ops' })
    expect(result).toMatchObject({ applied: { model: true, soul: true }, ok: true })
  })

  it('reports a rejected clear as a failed section', async () => {
    respondWith(() => ({ blocked: false, code: 1, output: 'Config key not set' }))

    await expect(applyAdvancedConfig(bot, dirtyModel({ model: '', provider: '' }))).resolves.toEqual({
      applied: { model: false },
      ok: false
    })
  })

  it('treats a thrown clear the same as a rejected one', async () => {
    hostMock.request.mockRejectedValue(new Error('gateway down'))

    await expect(applyAdvancedConfig(bot, dirtyModel({ model: '', provider: '' }))).resolves.toEqual({
      applied: { model: false },
      ok: false
    })
  })

  it('refuses a half-filled model section outright', async () => {
    // A provider with no model (or the reverse) is neither a pin nor a clear.
    await expect(applyAdvancedConfig(bot, dirtyModel({ model: '' }))).resolves.toEqual({
      applied: { model: false },
      ok: false
    })
    expect(routed).toHaveLength(0)
  })
})

describe('a guarded model switch (#95293)', () => {
  it('surfaces the SHARED confirm flow instead of silently dropping the pick', async () => {
    respondWith(() => ({
      applied: {},
      confirm_message: 'CONTRIBUTOR TIER: this model may train on your data.',
      confirm_required: true,
      ok: true
    }))

    const result = await applyAdvancedConfig(bot, dirtyModel())

    expect(routed).toHaveLength(1)
    expect(routed[0].method).toBe('profiles.configure')
    expect(routed[0].params.model).toBe('muse-spark-1.2-contributor')
    expect(routed[0].params.confirm_expensive_model).toBeFalsy()

    expect(confirmMock).toHaveBeenCalledTimes(1)
    expect(confirmMock.mock.calls[0][0].confirmMessage).toMatch(/CONTRIBUTOR TIER/)

    // Pending confirmation is NOT a failed section — the editor must not toast
    // "Some sections failed: model" while the confirm toast is still up.
    expect(result.applied?.model).not.toBe(false)
    expect(result.ok).toBe(true)
  })

  it('resends ONLY the model section on Confirm', async () => {
    let calls = 0

    respondWith(() => {
      calls += 1

      return calls === 1
        ? { applied: {}, confirm_message: 'guarded', confirm_required: true, ok: true }
        : { applied: { model: true }, ok: true }
    })

    await applyAdvancedConfig(bot, dirtyModel({ dirtySoul: true, soul: '# Zeta' }))

    const confirmed = await confirmMock.mock.calls[0][0].requestConfirmed()

    expect(routed).toHaveLength(2)
    expect(routed[1].method).toBe('profiles.configure')
    expect(Object.keys(routed[1].params).sort()).toEqual(['confirm_expensive_model', 'model', 'name', 'provider'])
    expect(routed[1].params).toMatchObject({
      confirm_expensive_model: true,
      model: 'muse-spark-1.2-contributor',
      name: 'zeta',
      provider: 'opencode-go'
    })
    expect(confirmed?.applied?.model).toBe(true)
  })

  it('refreshes the roster once the confirmed switch lands', async () => {
    respondWith(() => ({ applied: {}, confirm_message: 'guarded', confirm_required: true, ok: true }))

    await applyAdvancedConfig(bot, dirtyModel())
    confirmMock.mock.calls[0][0].finish()

    expect(invalidateMock).toHaveBeenCalled()
  })

  it('leaves an unguarded save single-shot', async () => {
    respondWith(() => ({ applied: { model: true }, ok: true }))

    const result = await applyAdvancedConfig(bot, dirtyModel())

    expect(routed).toHaveLength(1)
    expect(confirmMock).not.toHaveBeenCalled()
    expect(result).toMatchObject({ applied: { model: true }, ok: true })
  })
})
