/**
 * Invariant: the @assistant-ui dependency cluster agrees on one tap version.
 *
 * The Hermes desktop app (``apps/desktop``) is built from source on every
 * install/update via ``scripts/install.ps1`` → a locked dependency install →
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
 * numbers, only that ``pnpm-lock.yaml`` resolves a single shared tap and that
 * this tap satisfies every ``@assistant-ui/*`` package's declared peer
 * requirement (pnpm's lock records peer ranges; a dependency that resolved a
 * different tap would add a second tap version). It fails if any future bump
 * reintroduces a split tap version or requirement across the cluster.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { load } from 'js-yaml'
import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')
const LOCK_PATH = path.join(REPO_ROOT, 'pnpm-lock.yaml')
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

    while (parts.length < 3) {parts.push('0')}

    return [parseInt(parts[0], 10), parseInt(parts[1], 10), parseInt(parts[2], 10)]
  }

  const ver = parse(version)

  for (const clause of spec.split('||')) {
    const trimmed = clause.trim()

    if (!trimmed) {continue}

    if (trimmed.startsWith('^')) {
      const lo = parse(trimmed)

      if (ver[0] < lo[0] || (ver[0] === lo[0] && ver[1] < lo[1]) || (ver[0] === lo[0] && ver[1] === lo[1] && ver[2] < lo[2])) {continue}
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
  peerDependencies?: Record<string, string>
  peerDependenciesMeta?: Record<string, { optional?: boolean }>
}

function lockPackages(): Record<string, LockPackage> {
  const lock = load(fs.readFileSync(LOCK_PATH, 'utf-8')) as { packages?: Record<string, LockPackage> }

  return lock.packages ?? {}
}

/** ``name@version`` keys; a scoped name carries a leading @. */
function splitKey(key: string): { name: string, version: string } {
  const at = key.lastIndexOf('@')

  return { name: key.slice(0, at), version: key.slice(at + 1) }
}

function sharedTapVersion(packages: Record<string, LockPackage>): string {
  /** The one tap version every install site resolves to. */
  const versions = new Set<string>()

  for (const key of Object.keys(packages)) {
    const { name, version } = splitKey(key)

    if (name === TAP) {versions.add(version)}
  }

  assert.ok(versions.size > 0, 'pnpm-lock.yaml has no @assistant-ui/tap entry — the @assistant-ui cluster should resolve a single shared tap version.')
  assert.ok(versions.size === 1, `@assistant-ui/tap resolves to multiple versions ${[...versions].sort()} — the cluster must share one tap line (see this test's docstring).`)

  return [...versions][0]!
}

test('every @assistant-ui/* package\'s tap requirement is satisfiable', () => {
  const packages = lockPackages()
  const tapVersion = sharedTapVersion(packages)

  const offenders: string[] = []
  let checked = 0

  for (const [key, meta] of Object.entries(packages)) {
    const { name } = splitKey(key)

    if (!name.startsWith('@assistant-ui/') || name === TAP) {continue}
    const peerMeta = (meta.peerDependenciesMeta ?? {})[TAP]

    if (peerMeta?.optional) {continue}
    const spec = (meta.peerDependencies ?? {})[TAP]

    if (!spec) {continue}
    checked++

    if (!caretSatisfies(tapVersion, spec)) {
      offenders.push(`${name} requires ${TAP}"${spec}"`)
    }
  }

  assert.ok(checked > 0, `no @assistant-ui/* package in pnpm-lock.yaml declares a ${TAP} peer`)
  assert.deepEqual(
    offenders,
    [],
    `Locked ${TAP}@${tapVersion} does not satisfy: ` +
      offenders.join('; ') +
      '. The @assistant-ui cluster has split tap requirements — pin the ' +
      'offending package (e.g. via pnpm-workspace.yaml `overrides`) so the ' +
      'whole cluster shares one tap line. See this test\'s module docstring.'
  )
})
