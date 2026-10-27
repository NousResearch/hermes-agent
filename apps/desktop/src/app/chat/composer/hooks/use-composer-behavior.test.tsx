import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { useComposerBehavior } from './use-composer-behavior'
const api = vi.hoisted(() => vi.fn())
vi.mock('@/api/client', () => ({
  hermesApiAs: api,
  resolveOwnerNow: ({ profile }: { profile?: string }) => ({ connectionId: 'ga401', profile: profile || 'default' })
}))

let dispose = () => {}

const options = {
  gateway: {},
  profile: 'default',
  sessionId: 'runtime',
  storedId: 'stored',
  busy: false,
  enabled: true,
  empty: true,
  revision: 'model:high'
}

const install = (protocol = 2) => {
  dispose = registry.register({
    id: 'native-bridge-test',
    source: 'plugin:neo-next-prompt',
    area: 'composer.behavior.v2',
    data: {
      protocol,
      create: (env: any) => {
        let started = false

        return {
          update: (s: any) => {
            if (s.enabled && s.empty && !started) {
              started = true
              void env.request('/suggestion', {}).then((r: any) => env.publish(r.text))
            }
          },
          decide: () => '运行回归测试',
          refresh: () => {},
          dispose: () => {}
        }
      }
    }
  })
}

beforeEach(() => {
  api.mockReset()
  api.mockResolvedValue({ text: '运行回归测试' })
  install()
})
afterEach(() => {
  cleanup()
  dispose()
})
describe('scoped plugin composer host', () => {
  it('captures profile/connection and inserts no text automatically', async () => {
    const v = renderHook(p => useComposerBehavior(p), { initialProps: options })
    await waitFor(() => expect(v.result.current.text).toBe('运行回归测试'))
    expect(api.mock.calls[0][0]).toEqual({ connectionId: 'ga401', profile: 'default' })
    expect(api.mock.calls[0][1].path).toBe('/api/plugins/neo-next-prompt/suggestion')
    v.rerender({ ...options, empty: false })
    expect(v.result.current.text).toBeNull()
    v.rerender(options)
    expect(api).toHaveBeenCalledTimes(1)
  })
  it('drops an old scope result after changing profile', async () => {
    let finish: (r: unknown) => void = () => {}
    api.mockImplementationOnce(
      () =>
        new Promise(r => {
          finish = r
        })
    )
    const v = renderHook(p => useComposerBehavior(p), { initialProps: options })
    v.rerender({ ...options, profile: 'other' })
    await waitFor(() => expect(v.result.current.text).toBe('运行回归测试'))
    await act(async () => finish({ text: '旧会话建议' }))
    expect(v.result.current.text).toBe('运行回归测试')
    expect(api.mock.calls[1][0].profile).toBe('other')
  })
  it('does not execute an incompatible provider or publish unsafe text', async () => {
    dispose()
    install(3)
    const bad = renderHook(() => useComposerBehavior(options))
    expect(api).not.toHaveBeenCalled()
    expect(bad.result.current.text).toBeNull()
    bad.unmount()
    dispose()
    install()
    api.mockResolvedValue({ text: 'ab\u202ecd' })
    const v = renderHook(() => useComposerBehavior(options))
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1))
    expect(v.result.current.text).toBeNull()
  })
})
