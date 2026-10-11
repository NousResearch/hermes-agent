/**
 * A native package that a manifest declares optional must stay optional in the lockfile.
 *
 * npm only forgives a failed install script when the lockfile marks the package
 * ``"optional": true``. That flag is computed from EVERY edge into the package: one
 * plain ``dependencies``/``devDependencies`` edge from any workspace demotes it to
 * ``devOptional`` (or nothing), and a failing ``node-pre-gyp --fallback-to-build``
 * then aborts the whole ``npm ci`` — even when the declaring edge's workspace is the
 * only one selected. That is #134246: ``tests-js`` listed ``get-windows`` as a
 * devDependency, so the Desktop build's ``npm ci --include=optional`` died on
 * Windows ARM64, where get-windows has no prebuilt binding and its source build fails.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { describe, test } from 'vitest'

type LockEntry = {
  link?: boolean
  optional?: boolean
  hasInstallScript?: boolean
  dependencies?: Record<string, string>
  devDependencies?: Record<string, string>
  optionalDependencies?: Record<string, string>
}

const LOCK = JSON.parse(fs.readFileSync(path.resolve(__dirname, '..', 'package-lock.json'), 'utf8')) as {
  packages: Record<string, LockEntry>
}

// Node's lookup: the nearest node_modules/<name> walking up from the declarer.
// Workspace dirs (apps/desktop) fall through to the root tree.
function resolveFrom(packages: Record<string, LockEntry>, from: string, name: string): string | undefined {
  let base = from

  for (;;) {
    const candidate = `${base ? `${base}/` : ''}node_modules/${name}`

    if (packages[candidate]) {
      return candidate
    }

    if (!base) {
      return undefined
    }

    const nested = base.lastIndexOf('/node_modules/')
    base = nested >= 0 ? base.slice(0, nested) : ''
  }
}

/** Native packages declared optional somewhere, mapped to the non-optional edges that defeat it. */
function optionalNativeEdges(packages: Record<string, LockEntry>) {
  const edges = new Map<string, { optional: string[]; required: string[] }>()

  for (const [from, entry] of Object.entries(packages)) {
    const kinds = [
      ['optional', entry.optionalDependencies],
      ['required', entry.dependencies],
      ['required', entry.devDependencies],
    ] as const

    for (const [kind, deps] of kinds) {
      for (const name of Object.keys(deps ?? {})) {
        // npm lets optionalDependencies override a same-name dependencies entry.
        if (kind === 'required' && entry.optionalDependencies?.[name] !== undefined) {
          continue
        }

        const target = resolveFrom(packages, from, name)

        if (!target || !packages[target].hasInstallScript) {
          continue
        }

        const seen = edges.get(target) ?? { optional: [], required: [] }
        seen[kind].push(from || '<root>')
        edges.set(target, seen)
      }
    }
  }

  return [...edges].filter(([, seen]) => seen.optional.length > 0)
}

describe('package-lock optional native dependencies', () => {
  test('every native package declared optional is optional in the lock', () => {
    const declared = optionalNativeEdges(LOCK.packages)
    assert.ok(declared.length > 0, 'expected at least one optional native dependency in the lock')

    const broken = declared
      .filter(([target]) => LOCK.packages[target].optional !== true)
      .map(([target, seen]) => `${target}: optional via ${seen.optional.join(', ')}, `
        + `but required via ${seen.required.join(', ') || '(a transitive non-optional path)'}`)

    assert.deepEqual(broken, [],
      'npm ci aborts when these install scripts fail; declare them under optionalDependencies everywhere')
  })
})
