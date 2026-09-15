/**
 * Guards `installPluginSdk()` against the initialisation-order regression
 * fixed in 6c3d4a4af7.
 *
 * `sdk/runtime` sits in an import cycle (`sdk/index` -> `contrib/*` ->
 * `contrib/runtime-loader` -> `sdk/runtime`). A module-scope literal such as
 * `const GLOBALS = { __HERMES_PLUGIN_SDK__: sdk }` is therefore evaluated
 * before `sdk/index` has run. Unbundled that is harmless (live namespace
 * object); bundled, the namespace is a hoisted `var` assigned later, so the
 * capture froze as `undefined` and `Object.keys()` in `shimUrl()` threw --
 * breaking EVERY runtime-loaded plugin, content-independently.
 *
 * Two layers, because each misses what the other catches:
 *   1. source shape  -- runs on any checkout, including CI with no build.
 *   2. emitted chunk -- catches a bundler/config change that reintroduces the
 *      hazard while the source still looks correct.
 */

import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

import { parse } from 'acorn'
import { transpileModule } from 'typescript'
import { describe, expect, it } from 'vitest'

import { installPluginSdk, sdkImportMap } from '@/sdk/runtime'

type AstNode = Record<string, unknown>

const GLOBAL_KEYS = [
  '__HERMES_PLUGIN_SDK__',
  '__HERMES_REACT__',
  '__HERMES_REACT_JSX__',
  '__HERMES_REACT_JSX_DEV__'
] as const

/** Node types whose body runs on call, not during chunk evaluation. */
const FUNCTION_NODES = new Set([
  'ArrowFunctionExpression',
  'FunctionDeclaration',
  'FunctionExpression'
])

/**
 * Walk an AST, reporting every `__HERMES_PLUGIN_SDK__` property and whether it
 * sits inside a function. Deliberately shape-agnostic: an arrow
 * (`() => ({ ... })`) and a declaration (`function f() { return { ... } }`)
 * are both lazy, and a guard that only recognises one silently passes the
 * other.
 */
function findNamespaceReads(
  root: AstNode[]
): { eager: { identifier: null | string; stmtIndex: number }[]; lazy: number } {
  const eager: { identifier: null | string; stmtIndex: number }[] = []
  let lazy = 0

  const walk = (node: unknown, stmtIndex: number, insideFunction: boolean): void => {
    if (!node || typeof node !== 'object') {
      return
    }

    const n = node as AstNode
    const nested = insideFunction || FUNCTION_NODES.has(n.type as string)

    if (n.type === 'Property') {
      const key = n.key as { name?: string; value?: string } | undefined

      if ((key?.name ?? key?.value) === '__HERMES_PLUGIN_SDK__') {
        if (nested) {
          lazy += 1
        } else {
          const value = n.value as { name?: string; type: string }

          eager.push({
            identifier: value.type === 'Identifier' ? (value.name as string) : null,
            stmtIndex
          })
        }
      }
    }

    for (const child of Object.values(n)) {
      if (Array.isArray(child)) {
        child.forEach(item => walk(item, stmtIndex, nested))
      } else if (child && typeof child === 'object') {
        walk(child, stmtIndex, nested)
      }
    }
  }

  root.forEach((stmt, i) => walk(stmt, i, false))

  return { eager, lazy }
}

function parseProgram(source: string): AstNode[] {
  return (parse(source, { ecmaVersion: 'latest', sourceType: 'module' }) as unknown as { body: AstNode[] })
    .body
}

describe('installPluginSdk', () => {
  it('installs every namespace as a real object, never undefined', () => {
    installPluginSdk()

    for (const key of GLOBAL_KEYS) {
      const value = (globalThis as Record<string, unknown>)[key]

      expect(value, `${key} must be installed`).toBeTypeOf('object')
      expect(value, `${key} must not be undefined`).not.toBeUndefined()
    }
  })

  it('exposes the SDK members plugins actually render with', () => {
    installPluginSdk()

    const sdk = (globalThis as unknown as Record<string, Record<string, unknown>>)
      .__HERMES_PLUGIN_SDK__

    // Had the namespace frozen as undefined, none of these would resolve.
    for (const name of ['Tip', 'Codicon', 'cn', 'haptic']) {
      expect(sdk[name], `SDK must export ${name}`).toBeDefined()
    }
  })

  it('builds shim modules for every specifier plugins may import', () => {
    const map = sdkImportMap()

    for (const specifier of ['@hermes/plugin-sdk', 'react', 'react/jsx-runtime']) {
      expect(map[specifier], `${specifier} must map to a shim`).toMatch(/^blob:/)
    }
  })
})

describe('source shape', () => {
  // Runs everywhere, including a fresh CI checkout with no dist/.
  it('never captures the namespaces in a module-scope object literal', () => {
    const source = readFileSync(join(__dirname, 'runtime.ts'), 'utf8')

    // Transpile rather than regex-stripping types: a hand-rolled stripper
    // silently mis-parses new syntax and turns this guard into a no-op.
    const { outputText } = transpileModule(source, {
      compilerOptions: { target: 99 }
    })

    const { eager, lazy } = findNamespaceReads(parseProgram(outputText))

    expect(lazy, 'runtime.ts must build the namespaces inside a function').toBeGreaterThan(0)
    expect(
      eager,
      'runtime.ts must not capture the namespaces at module scope: the bundler ' +
        'can emit that literal before the SDK namespace is assigned, freezing it ' +
        'as undefined and breaking every runtime-loaded plugin.'
    ).toEqual([])
  })
})

/** The shipped chunk defining `__HERMES_PLUGIN_SDK__`, if the app was built. */
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

describe('emitted chunk', () => {
  const chunk = findBuiltSdkChunk()

  // Skipped VISIBLY when the app has not been built, rather than passing as a
  // green test that asserted nothing.
  it.skipIf(!chunk)('assigns the SDK namespace before anything reads it', () => {
    if (!chunk) {
      return
    }

    const body = parseProgram(chunk.source)

    // Statement index at which each top-level binding receives its value.
    const declaredAt = new Map<string, number>()

    body.forEach((stmt, i) => {
      if (stmt.type !== 'VariableDeclaration') {
        return
      }

      const declarations = stmt.declarations as {
        id: { name?: string; type: string }
        init: unknown
      }[]

      for (const d of declarations) {
        if (d.id.type === 'Identifier' && d.init && !declaredAt.has(d.id.name as string)) {
          declaredAt.set(d.id.name as string, i)
        }
      }
    })

    const { eager, lazy } = findNamespaceReads(body)

    expect(eager.length + lazy, 'chunk must define __HERMES_PLUGIN_SDK__').toBeGreaterThan(0)

    for (const { identifier, stmtIndex } of eager) {
      // Built inline (call, spread, ...) rather than from a hoisted binding:
      // there is no earlier binding that could still be undefined.
      if (identifier === null) {
        continue
      }

      const initIndex = declaredAt.get(identifier)

      expect(
        initIndex === undefined || initIndex < stmtIndex,
        `${chunk.file}: namespace '${identifier}' is assigned at statement #${initIndex} but ` +
          `read at #${stmtIndex} -- __HERMES_PLUGIN_SDK__ would be undefined and every ` +
          `runtime-loaded plugin would fail to load`
      ).toBe(true)
    }
  })
})
