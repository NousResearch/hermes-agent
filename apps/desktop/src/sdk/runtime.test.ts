/**
 * Guards the plugin-SDK global installation against a bundler-ordering
 * regression that source-level tests cannot see.
 *
 * The bug: `sdk/runtime.ts` used to hoist its namespaces into a module-level
 * `const GLOBALS = { __HERMES_PLUGIN_SDK__: sdk, … }`. Rolldown is free to
 * merge `sdk/runtime` and `sdk/index` into a single chunk and emit the
 * GLOBALS statement BEFORE the statement that assigns the SDK namespace
 * object. `__HERMES_PLUGIN_SDK__` then froze as `undefined`, the runtime
 * loader handed plugins an empty namespace, and every plugin (Follow-up
 * included) rendered as bare text with no components.
 *
 * Two layers of defence:
 *   1. a behavioural test that the namespaces really land on globalThis, and
 *   2. an AST check over the SHIPPED chunk proving the SDK namespace is
 *      initialised before it is read.
 */

import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

import { parse } from 'acorn'
import { describe, expect, it } from 'vitest'

import { installPluginSdk, sdkImportMap } from '@/sdk/runtime'

const GLOBAL_KEYS = ['__HERMES_PLUGIN_SDK__', '__HERMES_REACT__', '__HERMES_REACT_JSX__', '__HERMES_REACT_JSX_DEV__'] as const

describe('installPluginSdk', () => {
  it('installs every namespace as a real object, never undefined', () => {
    installPluginSdk()

    for (const key of GLOBAL_KEYS) {
      const value = (globalThis as Record<string, unknown>)[key]

      expect(value, `${key} must be installed`).toBeTypeOf('object')
      expect(value, `${key} must not be undefined`).not.toBeUndefined()
    }
  })

  it('exposes the SDK components plugins actually render with', () => {
    installPluginSdk()

    const sdk = (globalThis as unknown as Record<string, Record<string, unknown>>).__HERMES_PLUGIN_SDK__

    // A representative slice: if the namespace were the empty/undefined
    // object of the regression, none of these would exist.
    for (const name of ['Tip', 'Codicon', 'Button', 'cn', 'haptic', 'host']) {
      expect(sdk[name], `SDK must export ${name}`).toBeDefined()
    }
  })

  it('builds shim modules that re-export named members', () => {
    const map = sdkImportMap()

    expect(map['@hermes/plugin-sdk']).toMatch(/^blob:/)
    expect(map.react).toMatch(/^blob:/)
    expect(map['react/jsx-runtime']).toMatch(/^blob:/)
  })
})

/** The shipped chunk that defines `__HERMES_PLUGIN_SDK__`, if the app has been built. */
function findBuiltSdkChunk(): { file: string; source: string } | null {
  const assets = join(__dirname, '..', '..', 'dist', 'assets')

  if (!existsSync(assets)) {
    return null
  }

  for (const name of readdirSync(assets)) {
    if (!name.endsWith('.js')) {
      continue
    }

    const source = readFileSync(join(assets, name), 'utf8')

    if (source.includes('__HERMES_PLUGIN_SDK__:')) {
      return { file: name, source }
    }
  }

  return null
}

describe('source shape', () => {
  // This guard runs everywhere, including a fresh CI checkout with no dist/.
  // It encodes the invariant the regression violated: the namespaces must not
  // be captured by a MODULE-LEVEL object literal, because the bundler may
  // evaluate that literal before the SDK namespace object exists.
  it('never captures the namespaces in a module-level object literal', () => {
    const source = readFileSync(join(__dirname, 'runtime.ts'), 'utf8')

    // Strip the type annotations acorn cannot parse, then look for a
    // top-level (non-function) property named __HERMES_PLUGIN_SDK__.
    const stripped = source
      .replace(/^import .*$/gm, '')
      .replace(/\bas const\b/g, '')
      .replace(/: Record<string, string> \| null/g, '')
      .replace(/: Record<string, string>/g, '')
      .replace(/: GlobalKey/g, '')
      .replace(/: string/g, '')
      .replace(/: void/g, '')
      .replace(/\bexport /g, '')
      .replace(/type GlobalKey =[^\n]*\n/g, '')

    const ast = parse(stripped, { ecmaVersion: 'latest', sourceType: 'module' }) as unknown as {
      body: Record<string, unknown>[]
    }

    const FUNCTION_NODES = new Set(['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression'])
    let eager = false

    const walk = (node: unknown, insideFunction: boolean): void => {
      if (!node || typeof node !== 'object') {
        return
      }

      const n = node as Record<string, unknown>
      const nested = insideFunction || FUNCTION_NODES.has(n.type as string)

      if (n.type === 'Property' && !insideFunction) {
        const key = n.key as { name?: string; value?: string }

        if ((key?.name ?? key?.value) === '__HERMES_PLUGIN_SDK__') {
          eager = true
        }
      }

      for (const value of Object.values(n)) {
        if (Array.isArray(value)) {
          value.forEach(child => walk(child, nested))
        } else if (value && typeof value === 'object') {
          walk(value, nested)
        }
      }
    }

    ast.body.forEach(stmt => walk(stmt, false))

    expect(
      eager,
      'runtime.ts must build its globals inside a function. A module-level ' +
        '`{ __HERMES_PLUGIN_SDK__: sdk }` can be emitted before the SDK ' +
        'namespace is assigned, freezing it as undefined and making every ' +
        'plugin render as plain text.'
    ).toBe(false)
  })
})

describe('built bundle', () => {
  const chunk = findBuiltSdkChunk()

  // Skipped VISIBLY (reported by the runner) when the app has not been built,
  // rather than passing silently as a green test that never asserted.
  it.skipIf(!chunk)('initialises the SDK namespace before __HERMES_PLUGIN_SDK__ reads it', () => {
    if (!chunk) {
      return
    }

    const ast = parse(chunk.source, { ecmaVersion: 'latest', sourceType: 'module' }) as unknown as {
      body: Record<string, unknown>[]
    }

    // Statement index at which each top-level `var X = …` is initialised.
    const declaredAt = new Map<string, number>()

    ast.body.forEach((stmt, i) => {
      if (stmt.type !== 'VariableDeclaration') {
        return
      }

      for (const d of stmt.declarations as { id: { type: string; name?: string }; init: unknown }[]) {
        if (d.id.type === 'Identifier' && d.init && !declaredAt.has(d.id.name as string)) {
          declaredAt.set(d.id.name as string, i)
        }
      }
    })

    // Locate the object literal carrying __HERMES_PLUGIN_SDK__.
    //
    // Only an EAGER read is dangerous: one evaluated as part of the chunk's
    // top-level statement sequence. A read inside a function body (the fix:
    // `() => ({ __HERMES_PLUGIN_SDK__: sdk, … })`) runs when the loader
    // calls it, long after every top-level `var` is assigned — so we track
    // whether the walk has descended through a function.
    const FUNCTION_NODES = new Set(['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression'])

    let eagerRead: { stmtIndex: number; identifier: string | null } | null = null
    let lazyRead = false

    const walk = (node: unknown, stmtIndex: number, insideFunction: boolean): void => {
      if (!node || typeof node !== 'object') {
        return
      }

      const n = node as Record<string, unknown>
      const nested = insideFunction || FUNCTION_NODES.has(n.type as string)

      if (n.type === 'Property') {
        const key = n.key as { name?: string; value?: string }

        if ((key?.name ?? key?.value) === '__HERMES_PLUGIN_SDK__') {
          if (nested) {
            lazyRead = true
          } else {
            const value = n.value as { type: string; name?: string }

            eagerRead = { stmtIndex, identifier: value.type === 'Identifier' ? (value.name as string) : null }
          }
        }
      }

      for (const key of Object.keys(n)) {
        const child = n[key]

        if (Array.isArray(child)) {
          child.forEach(c => walk(c, stmtIndex, nested))
        } else if (child && typeof child === 'object') {
          walk(child, stmtIndex, nested)
        }
      }
    }

    ast.body.forEach((stmt, i) => walk(stmt, i, false))

    expect(eagerRead !== null || lazyRead, 'chunk must define __HERMES_PLUGIN_SDK__').toBe(true)

    // Deferred behind a function — safe by construction.
    if (eagerRead === null) {
      return
    }

    const { stmtIndex, identifier } = eagerRead

    // Produced inline (call expression, spread…) rather than referencing a
    // hoisted binding — nothing can be read before it exists.
    if (identifier === null) {
      return
    }

    const initIndex = declaredAt.get(identifier)

    expect(
      initIndex === undefined || initIndex < stmtIndex,
      `${chunk.file}: SDK namespace '${identifier}' is initialised at statement #${initIndex} but ` +
        `read eagerly at #${stmtIndex} — __HERMES_PLUGIN_SDK__ would be undefined and every ` +
        `plugin would render as plain text`
    ).toBe(true)
  })
})
