import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import * as sdk from './index'
import { installPluginSdk, pluginSdkGlobalsForTest, sdkImportMap } from './runtime'

const GLOBAL_KEYS = [
  '__HERMES_PLUGIN_SDK__',
  '__HERMES_REACT__',
  '__HERMES_REACT_JSX__',
  '__HERMES_REACT_JSX_DEV__'
] as const

const SPECIFIERS = [
  '@hermes/plugin-sdk',
  'react',
  'react/jsx-runtime',
  'react/jsx-dev-runtime'
] as const

function liveSdkNamedExport(): string {
  const name = Object.keys(sdk).find((key) => key !== 'default' && /^[A-Za-z_$][\w$]*$/.test(key))
  expect(name).toEqual(expect.any(String))
  return name as string
}

const shimBlobs = new Map<string, Blob>()

beforeAll(() => {
  const originalCreateObjectURL = URL.createObjectURL.bind(URL)
  vi.spyOn(URL, 'createObjectURL').mockImplementation((obj) => {
    const url = originalCreateObjectURL(obj)
    if (obj instanceof Blob) shimBlobs.set(url, obj)
    return url
  })
})

async function readShimSource(url: string): Promise<string> {
  const blob = shimBlobs.get(url)
  expect(blob).toBeInstanceOf(Blob)
  return blob!.text()
}

function emitShimSource(globalKey: string, names: string[]): string {
  return (
    `const m = globalThis.${globalKey};\n` +
    `export default m.default ?? m;\n` +
    (names.length ? `export const { ${names.join(', ')} } = m;\n` : '')
  )
}

afterEach(() => {
  for (const key of GLOBAL_KEYS) {
    delete (globalThis as Record<string, unknown>)[key]
  }
})

describe('plugin SDK runtime globals', () => {
  it('exposes the four host namespaces as late-bound accessors', () => {
    for (const key of GLOBAL_KEYS) {
      const descriptor = Object.getOwnPropertyDescriptor(pluginSdkGlobalsForTest, key)
      expect(descriptor?.get).toEqual(expect.any(Function))
    }
  })

  it('builds a cached import map for the four runtime specifiers', () => {
    const first = sdkImportMap()
    expect(() => sdkImportMap()).not.toThrow()
    const second = sdkImportMap()
    expect(second).toBe(first)
    expect(Object.keys(first).sort()).toEqual([...SPECIFIERS].sort())
  })

  it('installs a live SDK namespace that Object.keys can enumerate', async () => {
    installPluginSdk()
    const installed = (globalThis as Record<string, unknown>).__HERMES_PLUGIN_SDK__
    expect(installed).toEqual(expect.any(Object))
    expect(() => Object.keys(installed as object)).not.toThrow()

    const named = liveSdkNamedExport()
    expect(Object.keys(installed as object)).toContain(named)

    const source = await readShimSource(sdkImportMap()['@hermes/plugin-sdk'])
    expect(source).toContain(`const m = globalThis.__HERMES_PLUGIN_SDK__`)
    expect(source).toContain('export default m.default ?? m')
    expect(source).toContain(named)
  })

  it('late-binds so Object.keys sees the namespace after it is assigned', () => {
    const ns = pluginSdkGlobalsForTest.__HERMES_PLUGIN_SDK__
    expect(ns).toEqual(expect.any(Object))
    expect(() => Object.keys(ns)).not.toThrow()
    expect(Object.keys(ns)).toContain(liveSdkNamedExport())
    expect(ns).toBe(sdk)
  })

  it('does not emit empty named-export destructuring', async () => {
    expect(emitShimSource('__HERMES_PLUGIN_SDK__', [])).not.toContain('export const { }')

    const source = await readShimSource(sdkImportMap()['@hermes/plugin-sdk'])
    expect(source).not.toMatch(/export const \{\s*\} = m/)
  })
})
