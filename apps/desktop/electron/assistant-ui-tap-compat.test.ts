/**
 * Invariant: the @assistant-ui dependency cluster agrees on one tap version.
 *
 * The Hermes desktop app (``apps/desktop``) is built from source on every
 * install/update via ``scripts/install.ps1`` → ``npm ci``/``npm install`` →
 * ``tsc -b && vite build``. The ``@assistant-ui`` packages share an internal
 * reactivity lib, ``@assistant-ui/tap``, and they only interoperate when they
 * all resolve the *same* tap version:
 *
 * - ``@assistant-ui/react@0.12.28`` and ``@assistant-ui/core`` pin
 *   ``@assistant-ui/tap@^0.5.x`` (which exports ``.`` and ``./react``).
 * - ``@assistant-ui/store@0.2.18`` bumped its tap peer to ``^0.9.0`` and started
 *   importing ``@assistant-ui/tap/react-shim`` — an entry point that only exists
 *   in the tap ``0.9.x`` line.
 *
 * Because ``react@0.12.28`` requests ``store@^0.2.9`` (a caret range), a fresh
 * install silently floated ``store`` up to ``0.2.18``, which then could not find
 * ``./react-shim`` in the hoisted ``tap@0.5.x`` and crashed ``vite build`` with:
 *
 *     "./react-shim" is not exported ... from package @assistant-ui/tap
 *
 * i.e. the opaque "apps/desktop build failed (exit 1)" every user hit when
 * updating. The fix pins ``@assistant-ui/store`` (via root ``overrides``) to the
 * last release that targets ``tap@^0.5.x``.
 *
 * This is a *contract* test, not a snapshot: it does not assert specific version
 * numbers, only that the cluster resolves a single shared tap (wherever npm
 * places it — hoisted to root, or nested under the ``apps/desktop`` workspace
 * since the 0.14 bump dropped the ``store`` override) and that this tap satisfies
 * every ``@assistant-ui/*`` package's declared requirement. It fails if any
 * future bump reintroduces a split tap version or requirement across the cluster.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..')
const LOCK_PATH = path.join(REPO_ROOT, 'package-lock.json')
const TAP = '@assistant-ui/tap'

/**
 * Minimal npm semver check for the ranges this cluster actually uses.
 *
 * Supports exact versions, ``^x.y.z`` (with correct 0.x semantics), and
 * ``||`` unions. Pre-release tags are ignored (none are used here).
 */
function caretSatisfies(version: string, spec: string): boolean {
  function parse(v: string): [number, number, number] {
    const core = v.replace(/^[^0-9]+/, '').split('-')[0].split('+')[0]
    const parts = core.split('.').slice(0, 3)
    while (parts.length < 3) parts.push('0')
    return [parseInt(parts[0], 10), parseInt(parts[1], 10), parseInt(parts[2], 10)]
  }

  const ver = parse(version)
  for (const clause of spec.split('||')) {
    const trimmed = clause.trim()
    if (!trimmed) continue
    if (trimmed.startsWith('^')) {
      const lo = parse(trimmed)
      if (ver[0] < lo[0] || (ver[0] === lo[0] && ver[1] < lo[1]) || (ver[0] === lo[0] && ver[1] === lo[1] && ver[2] < lo[2])) continue
      let hi: [number, number, number]
      if (lo[0] > 0) {
        hi = [lo[0] + 1, 0, 0]
      } else if (lo[1] > 0) {
        hi = [0, lo[1] + 1, 0]
      } else {
        hi = [0, 0, lo[2] + 1]
      }
      if (
        (ver[0] < hi[0]) ||
        (ver[0] === hi[0] && ver[1] < hi[1]) ||
        (ver[0] === hi[0] && ver[1] === hi[1] && ver[2] < hi[2])
      ) {
        return true
      }
    } else if (trimmed[0].match(/\d/) || trimmed.startsWith('v')) {
      if (ver[0] === parse(trimmed)[0] && ver[1] === parse(trimmed)[1] && ver[2] === parse(trimmed)[2]) {
        return true
      }
    }
  }
  return false
}

interface LockPackage {
  version?: string
  dependencies?: Record<string, string>
  peerDependencies?: Record<string, string>
  peerDependenciesMeta?: Record<string, { optional?: boolean }>
}

function lockPackages(): Record<string, LockPackage> {
  const lock = JSON.parse(fs.readFileSync(LOCK_PATH, 'utf-8'))
  return (lock.packages ?? {}) as Record<string, LockPackage>
}

function sharedTapVersion(packages: Record<string, LockPackage>): string {
  /** The one tap version every install site resolves to. */
  const versions = new Set<string>()
  for (const [key, meta] of Object.entries(packages)) {
    const idx = key.lastIndexOf('node_modules/')
    const name = idx >= 0 ? key.slice(idx + 'node_modules/'.length) : key
    if (name === TAP) {
      if (meta.version) versions.add(meta.version)
    }
  }
  assert.ok(versions.size > 0, 'package-lock.json has no @assistant-ui/tap entry — the @assistant-ui cluster should resolve a single shared tap version.')
  assert.ok(versions.size === 1, `@assistant-ui/tap resolves to multiple versions ${[...versions].sort()} — the cluster must share one tap line (see this test's docstring).`)
  return [...versions][0]!
}

test('every @assistant-ui/* package\'s tap requirement is satisfiable', () => {
  const packages = lockPackages()

  const tapVersion = sharedTapVersion(packages)

  const offenders: string[] = []
  for (const [key, meta] of Object.entries(packages)) {
    const idx = key.lastIndexOf('node_modules/')
    const name = idx >= 0 ? key.slice(idx + 'node_modules/'.length) : key
    if (!name.startsWith('@assistant-ui/') || name === TAP) continue
    const peerMeta = (meta.peerDependenciesMeta ?? {})[TAP]
    if (peerMeta?.optional) continue
    const spec = (meta.dependencies ?? {})[TAP] || (meta.peerDependencies ?? {})[TAP]
    if (!spec) continue
    if (!caretSatisfies(tapVersion, spec)) {
      offenders.push(`${name} requires ${TAP}"${spec}"`)
    }
  }

  assert.deepEqual(
    offenders,
    [],
    `Hoisted ${TAP}@${tapVersion} does not satisfy: ` +
      offenders.join('; ') +
      '. The @assistant-ui cluster has split tap requirements — pin the ' +
      'offending package (e.g. via root package.json `overrides`) so the ' +
      'whole cluster shares one tap line. See this test\'s module docstring.'
  )
})

test('caretSatisfies helper', () => {
  assert.ok(caretSatisfies('0.5.14', '^0.5.10'))
  assert.ok(caretSatisfies('0.5.14', '^0.5.14'))
  assert.ok(!caretSatisfies('0.5.14', '^0.9.0'))
  assert.ok(!caretSatisfies('0.5.14', '^0.6.0'))
  assert.ok(caretSatisfies('1.2.5', '^1.2.0'))
  assert.ok(!caretSatisfies('2.0.0', '^1.2.0'))
  assert.ok(caretSatisfies('0.5.14', '^0.5.0 || ^0.9.0'))
})
