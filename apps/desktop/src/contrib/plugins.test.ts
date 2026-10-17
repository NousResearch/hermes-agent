import { describe, expect, it, vi } from 'vitest'

import { unwrapPluginDefault } from './plugins'

vi.mock('./runtime-loader', () => ({ watchRuntimePlugins: vi.fn() }))
vi.mock('./plugins-store', () => ({
  publishPlugin: vi.fn(),
  pluginActive: vi.fn().mockReturnValue(false)
}))

describe('unwrapPluginDefault', () => {
  it('returns the plugin object when default is the ESM object form', () => {
    const plugin = { id: 'accent', name: 'Accent', register: vi.fn() }
    // Normal Vite/ESM glob result: `{ default: HermesPlugin }`
    expect(unwrapPluginDefault({ default: plugin })).toBe(plugin)
  })

  it('invokes the default getter and returns the plugin when it is a Rolldown CJS-interop wrapper', () => {
    const plugin = { id: 'hermes-bots', name: 'Bots', register: vi.fn() }
    const getter = vi.fn(() => plugin)
    // Rolldown CJS-interop: `{ default: () => HermesPlugin }`
    const result = unwrapPluginDefault({ default: getter })

    expect(getter).toHaveBeenCalledTimes(1)
    expect(result).toBe(plugin)
  })

  it('returns null when the module shape has no default export', () => {
    expect(unwrapPluginDefault({})).toBeNull()
    expect(unwrapPluginDefault({ default: undefined })).toBeNull()
    expect(unwrapPluginDefault(null)).toBeNull()
    expect(unwrapPluginDefault(undefined)).toBeNull()
  })

  it('returns null (never throws) when the default getter throws', () => {
    const broken = { default: () => { throw new Error('boom') } }
    expect(unwrapPluginDefault(broken)).toBeNull()
  })

  it('returns null when the default getter resolves to a non-object value', () => {
    // Defensive: a misbehaving wrapper could resolve to undefined/null
    expect(unwrapPluginDefault({ default: () => undefined })).toBeNull()
    expect(unwrapPluginDefault({ default: () => null })).toBeNull()
  })
})
