import { afterEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { loadRuntimePlugin } from '@/contrib/runtime-loader'

import { namespaceExportNames, resetSdkImportMapCache, sdkImportMap } from './runtime'

// Evidence: since the 2026-09-10 desktop build, every disk-door plugin fails
// at sdk-*.js module-graph prep with
//   TypeError: Cannot convert undefined or null to object
// before register() runs — including a zero-import plugin. The production
// throw is Object.keys(GLOBALS[key]) inside shimUrl (via sdkImportMap), which
// loadRuntimePlugin always invokes. The live slot that goes null in a
// production Electron graph is `import * as jsxDevRuntime from
// 'react/jsx-dev-runtime'` (__HERMES_REACT_JSX_DEV__): Vite/Rolldown interop
// or a missing jsx-dev-runtime in the packaged renderer. A naive
// loadRuntimePlugin(zero-import) test is NOT red here — vitest's graph has a
// real jsx-dev-runtime namespace. The helper IS the Object.keys call.

function stubBlobAsDataUrl(): () => void {
  const createObjectURL = vi
    .spyOn(URL, 'createObjectURL')
    .mockImplementation(
      blob =>
        `data:text/javascript;base64,${Buffer.from((blob as unknown as { parts: string[] }).parts.join('')).toString('base64')}`
    )
  const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const RealBlob = globalThis.Blob
  vi.stubGlobal(
    'Blob',
    class {
      parts: string[]
      constructor(parts: string[]) {
        this.parts = parts
      }
    }
  )

  return () => {
    createObjectURL.mockRestore()
    revokeObjectURL.mockRestore()
    vi.stubGlobal('Blob', RealBlob)
  }
}

describe('namespaceExportNames (disk-door sdk shim prep)', () => {
  it('does not throw TypeError when a live namespace is null/undefined', () => {
    // Arrange the failing GLOBALS slot the production graph actually hits.
    expect(() => namespaceExportNames(undefined)).not.toThrow(TypeError)
    expect(() => namespaceExportNames(null)).not.toThrow(TypeError)
    expect(namespaceExportNames(undefined)).toEqual([])
    expect(namespaceExportNames(null)).toEqual([])
  })

  it('does not throw TypeError for a non-object interop binding', () => {
    expect(() => namespaceExportNames(42)).not.toThrow(TypeError)
    expect(namespaceExportNames(42)).toEqual([])
  })

  it('still emits named exports on a healthy import-star namespace', () => {
    const names = namespaceExportNames({
      cn: () => undefined,
      default: {},
      host: {},
      PALETTE_AREA: 'palette',
      ROUTES_AREA: 'routes',
      SIDEBAR_NAV_AREA: 'sidebar-nav'
    })

    expect(names).toEqual(expect.arrayContaining(['host', 'cn', 'ROUTES_AREA', 'SIDEBAR_NAV_AREA', 'PALETTE_AREA']))
    expect(names).not.toContain('default')
  })
})

describe('sdkImportMap', () => {
  it('keeps every specifier key (including jsx-dev-runtime) so imports stay supported', () => {
    const map = sdkImportMap()

    expect(map['@hermes/plugin-sdk']).toMatch(/^(blob:|data:)/)
    expect(map.react).toMatch(/^(blob:|data:)/)
    expect(map['react/jsx-runtime']).toMatch(/^(blob:|data:)/)
    expect(map['react/jsx-dev-runtime']).toMatch(/^(blob:|data:)/)
  })
})

describe('loadRuntimePlugin through SDK graph prep', () => {
  let restore: (() => void) | undefined

  afterEach(() => {
    restore?.()
    restore = undefined
    resetSdkImportMapCache()
  })

  it('zero-import disk plugin still loads when the SDK graph is prepared', async () => {
    restore = stubBlobAsDataUrl()
    resetSdkImportMapCache()
    ;(globalThis as unknown as { __diagRan?: boolean }).__diagRan = false

    const source = `export default { id: 'diag-noimport', name: 'Diag', defaultEnabled: true, register() { globalThis.__diagRan = true } }`
    const id = await loadRuntimePlugin(source, 'diag-noimport')

    expect(id).toBe('diag-noimport')
    expect((globalThis as unknown as { __diagRan?: boolean }).__diagRan).toBe(true)
    expect($pluginRecords.get()['diag-noimport']).toMatchObject({ status: 'loaded' })
  })

  it('rewrites a @hermes/plugin-sdk named import and still runs register()', async () => {
    restore = stubBlobAsDataUrl()
    resetSdkImportMapCache()
    ;(globalThis as unknown as { __sdkHostRan?: boolean }).__sdkHostRan = false

    const source = `import { host, cn } from '@hermes/plugin-sdk'
export default {
  id: 'diag-sdk-import',
  name: 'Diag SDK',
  defaultEnabled: true,
  register() {
    if (typeof host !== 'object' || typeof cn !== 'function') throw new Error('sdk names missing')
    globalThis.__sdkHostRan = true
  }
}`
    const id = await loadRuntimePlugin(source, 'diag-sdk-import')

    expect(id).toBe('diag-sdk-import')
    expect((globalThis as unknown as { __sdkHostRan?: boolean }).__sdkHostRan).toBe(true)
  })

  it('invalid plugin (no register) still errors', async () => {
    restore = stubBlobAsDataUrl()
    resetSdkImportMapCache()

    const id = await loadRuntimePlugin(`export default { id: 'diag-invalid', name: 'Nope' }`, 'diag-invalid')

    expect(id).toBeNull()
    expect($pluginRecords.get()['diag-invalid']).toMatchObject({ status: 'error' })
  })
})
