/**
 * Invariant: every workspace that renders React ships one react/react-dom pair.
 *
 * React validates at import time that ``react`` and ``react-dom`` come from the
 * same installed copy. When they don't, it throws "Minified React error #527"
 * *before* the first paint — the Electron window just stays blank white, with
 * the only clue buried in the devtools console of a packaged build.
 *
 * npm never warns about this. ``apps/desktop`` pins both packages to one exact
 * version, but a root dependency whose react peer is a loose range (e.g.
 * ``^18.0.0 || ^19.0.0``, and with no react-dom peer to keep the two in step)
 * makes npm hoist the newest react to the monorepo root while react-dom stays
 * at the pinned version. react-dom's own peer range is a caret, so the newer
 * react still "satisfies" it and the install reports success.
 *
 * ``apps/desktop/vite.config.ts`` used to alias both packages to a hardcoded
 * ``../../node_modules/<pkg>`` — i.e. straight into the split — so the bundle
 * shipped the mismatched pair. It now resolves them from the workspace itself,
 * where npm guarantees the declared versions are reachable.
 *
 * This is a *contract* test: it asserts no specific version, only that the two
 * halves of the pair can never drift apart again — neither through a loosened
 * manifest spec, nor by re-pinning the bundler at the hoisted copy.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')
const DESKTOP_VITE_CONFIG = path.join(REPO_ROOT, 'apps', 'desktop', 'vite.config.ts')

interface Manifest {
  dependencies?: Record<string, string>
  devDependencies?: Record<string, string>
  workspaces?: string[]
}

function readManifest(file: string): Manifest {
  return JSON.parse(fs.readFileSync(file, 'utf-8')) as Manifest
}

/** Every workspace manifest, resolved from the root ``workspaces`` globs. */
function workspaceManifests(): { name: string, manifest: Manifest }[] {
  const patterns = readManifest(path.join(REPO_ROOT, 'package.json')).workspaces ?? []
  const found: { name: string, manifest: Manifest }[] = []

  for (const pattern of patterns) {
    // The globs in use are plain paths or a single trailing ``/*``.
    const parent = pattern.endsWith('/*') ? path.join(REPO_ROOT, pattern.slice(0, -2)) : null
    const dirs = parent === null
      ? [pattern]
      : fs.existsSync(parent)
        ? fs.readdirSync(parent).map((entry) => `${pattern.slice(0, -2)}/${entry}`)
        : []

    for (const dir of dirs) {
      const file = path.join(REPO_ROOT, dir, 'package.json')

      if (fs.existsSync(file)) {found.push({ name: dir, manifest: readManifest(file) })}
    }
  }

  return found
}

test('workspaces declaring react and react-dom pin them to the same exact version', () => {
  const offenders: string[] = []

  for (const { name, manifest } of workspaceManifests()) {
    const deps = { ...manifest.devDependencies, ...manifest.dependencies }
    const react = deps['react']
    const reactDom = deps['react-dom']

    if (!react || !reactDom) {continue}

    if (react !== reactDom) {
      offenders.push(`${name} declares react"${react}" but react-dom"${reactDom}"`)
      continue
    }

    if (!/^\d/.test(react)) {
      offenders.push(`${name} declares a floating range react/react-dom"${react}"`)
    }
  }

  assert.deepEqual(
    offenders,
    [],
    'react and react-dom must be pinned to the same exact version per workspace, ' +
      'otherwise npm can hoist a newer react next to the older react-dom and React ' +
      `throws error #527 (blank window): ${offenders.join('; ')}`
  )
})

test('the desktop bundler does not alias react at the hoisted root copy', () => {
  const config = fs.readFileSync(DESKTOP_VITE_CONFIG, 'utf-8')

  assert.ok(
    !config.includes('node_modules/react'),
    'apps/desktop/vite.config.ts hardcodes a node_modules path for react/react-dom. ' +
      'That pins the bundle to the hoisted copies, which npm is free to resolve to a ' +
      'different version than the pinned react-dom. Resolve both from the workspace ' +
      "instead (see this test's module docstring)."
  )
})
