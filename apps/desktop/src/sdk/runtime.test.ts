import { describe, expect, it } from 'vitest'

import { shimSource } from './runtime'

// The shim re-exports a global namespace's live members as an ESM blob that only ever runs inside
// Chromium. A bad name in the generated `export const { … } = ns` is a SyntaxError surfaced as a
// bare "Unexpected token ')'" pointing at generated code, with nothing naming the plugin at fault
// -- so these assert the emitted source directly instead of waiting for a runtime crash.

/** The names inside the generated destructuring statement. */
function exportedNames(source: string): string[] {
  const match = /\{([^}]*)\}/.exec(source)
  if (!match) return []
  return match[1].split(',').map(name => name.trim()).filter(Boolean)
}

describe('shimSource', () => {
  it('never emits a reserved word as an export binding', () => {
    // `in` is the proven leak: it matches the identifier shape test, and
    // `export const { in } = ns` is a SyntaxError.
    const source = shimSource('__HERMES_PLUGIN_SDK__', { in: 1, ok: 2, default: 3 })

    expect(exportedNames(source)).toEqual(['ok'])
    expect(source).not.toContain('in,')
  })

  it('keeps the namespace binding out of the destructure list', () => {
    // The binding is unlikely AND excluded from the destructure list, so a namespace exporting a
    // name that collides with it cannot produce `Identifier … has already been declared`.
    const source = shimSource('__HERMES_PLUGIN_SDK__', { __hermes_shim_ns__: 1, ok: 2 })

    expect(exportedNames(source)).toEqual(['ok'])
    expect(source).toContain('const __hermes_shim_ns__ = globalThis.__HERMES_PLUGIN_SDK__;')
  })

  it('allows a namespace to export the name the shim used to bind locally', () => {
    // `m` was the old local binding, and the session-list-density chunk exports `m` -- which made
    // `export const { m } = m` a hard "Identifier 'm' has already been declared". Moving the
    // binding off `m` is what makes this namespace loadable at all.
    const source = shimSource('__HERMES_PLUGIN_SDK__', { m: 1, ok: 2 })

    expect(exportedNames(source)).toEqual(['m', 'ok'])
  })

  it('omits the destructure statement entirely for a namespace with no named exports', () => {
    // `export const {  } = ns` is itself a syntax error, so `default`-only and empty namespaces
    // must emit the default export alone.
    const source = shimSource('__HERMES_REACT__', { default: 1 })

    expect(source).not.toContain('{')
    expect(source).toContain('export default __hermes_shim_ns__.default ?? __hermes_shim_ns__;')
  })

  it('drops names that are not valid binding identifiers', () => {
    const source = shimSource('__HERMES_REACT__', { 'not-a-name': 1, '9lives': 2, ok: 3 })

    expect(exportedNames(source)).toEqual(['ok'])
  })
})
