import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import { BUILD_CRITICAL_PACKAGES as BUILD_CRITICAL, checkRootInstall, requiredPackages } from '../scripts/assert-root-install.mjs'

// Build a throwaway repo shaped like this one: an app workspace whose
// dependencies are hoisted to the repo root, which is what the guard walks.
// `manifest` is merged into the app's package.json so tests can declare
// dependencies the guard is expected to read.
function makeTree({ rootPackages = BUILD_CRITICAL, react = '19.2.7', reactDom = '19.2.7', manifest = {} } = {}) {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-assert-root-'))
  const appDir = path.join(tempRoot, 'apps', 'desktop')
  fs.mkdirSync(appDir, { recursive: true })
  fs.writeFileSync(path.join(appDir, 'package.json'), JSON.stringify({ name: 'desktop', ...manifest }), 'utf8')

  const writePackage = (name, version) => {
    const dir = path.join(tempRoot, 'node_modules', name)
    fs.mkdirSync(dir, { recursive: true })
    fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ name, version }), 'utf8')
  }
  for (const name of rootPackages) writePackage(name, '1.0.0')
  if (react !== null) writePackage('react', react)
  if (reactDom !== null) writePackage('react-dom', reactDom)

  return { tempRoot, appDir }
}

test('checkRootInstall passes on a complete root install', () => {
  const { tempRoot, appDir } = makeTree()
  try {
    assert.deepEqual(checkRootInstall(appDir, tempRoot), { ok: true })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// The regression this guard was widened for: the updater's partial `npm install`
// left katex out while vite was present, so the old vite-only check passed and
// the build died on an unresolved `katex/dist/katex.min.css` (#86443).
test('checkRootInstall fails when katex is missing but vite is present', () => {
  const { tempRoot, appDir } = makeTree({
    rootPackages: BUILD_CRITICAL.filter(name => name !== 'katex')
  })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /katex/)
    assert.match(result.error, /npm ci/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkRootInstall reports every missing package at once', () => {
  const { tempRoot, appDir } = makeTree({ rootPackages: ['vite'] })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    for (const name of ['katex', 'electron', 'electron-builder']) {
      assert.match(result.error, new RegExp(name))
    }
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// The original guard's only check — kept, so widening coverage cannot silently
// drop the case it already handled.
test('checkRootInstall still fails when vite is missing', () => {
  const { tempRoot, appDir } = makeTree({
    rootPackages: BUILD_CRITICAL.filter(name => name !== 'vite')
  })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /vite/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkRootInstall fails on a react/react-dom version split', () => {
  const { tempRoot, appDir } = makeTree({ react: '19.2.7', reactDom: '19.1.0' })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /#527/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// A package installed into the app's own node_modules rather than hoisted to the
// root is still installed. The guard walks upward like Node does, so it must not
// insist on the hoisted location.
test('checkRootInstall accepts a package nested in the app workspace', () => {
  const { tempRoot, appDir } = makeTree({
    rootPackages: BUILD_CRITICAL.filter(name => name !== 'katex')
  })
  const nested = path.join(appDir, 'node_modules', 'katex')
  fs.mkdirSync(nested, { recursive: true })
  fs.writeFileSync(path.join(nested, 'package.json'), JSON.stringify({ name: 'katex' }), 'utf8')
  try {
    assert.deepEqual(checkRootInstall(appDir, tempRoot), { ok: true })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// The class, not the four instances: the floor list is what a partial install
// has been *seen* to drop, but any declared non-optional package can be the one
// missing next (`vite.config.ts` imports `@rolldown/plugin-babel`, which the
// floor never named). The guard must read the manifest so the list cannot drift
// behind a new import.
test('checkRootInstall fails when a declared devDependency outside the floor is missing', () => {
  const { tempRoot, appDir } = makeTree({
    manifest: { devDependencies: { '@rolldown/plugin-babel': '1.0.0', esbuild: '1.0.0' } },
    rootPackages: [...BUILD_CRITICAL, 'esbuild']
  })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /@rolldown\/plugin-babel/)
    assert.doesNotMatch(result.error, /esbuild/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkRootInstall fails when a declared runtime dependency is missing', () => {
  const { tempRoot, appDir } = makeTree({
    manifest: { dependencies: { '@vscode/codicons': '1.0.0' } }
  })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /@vscode\/codicons/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// npm skips optionalDependencies legitimately (platform-gated natives), so an
// absent optional package is not a partial install.
test('checkRootInstall ignores missing optionalDependencies', () => {
  const { tempRoot, appDir } = makeTree({
    manifest: { optionalDependencies: { 'get-windows': '9.3.0' } }
  })
  try {
    assert.deepEqual(checkRootInstall(appDir, tempRoot), { ok: true })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkRootInstall passes when every declared package is installed', () => {
  const { tempRoot, appDir } = makeTree({
    manifest: { dependencies: { '@scope/pkg': '1.0.0' }, devDependencies: { esbuild: '1.0.0' } },
    rootPackages: [...BUILD_CRITICAL, '@scope/pkg', 'esbuild']
  })
  try {
    assert.deepEqual(checkRootInstall(appDir, tempRoot), { ok: true })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// The floor is unconditional: a manifest the guard cannot parse must not turn
// the check off.
test('checkRootInstall keeps the floor when the manifest is unreadable', () => {
  const { tempRoot, appDir } = makeTree({ rootPackages: ['vite'] })
  fs.writeFileSync(path.join(appDir, 'package.json'), '{not json', 'utf8')
  try {
    assert.deepEqual(requiredPackages(appDir), [])
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /katex/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// A workspace member declared `file:../shared` is not a registry package: npm links
// it and `vite.config.ts` aliases it to the workspace source, so an untraversable
// `node_modules/@hermes/shared` (a reparse point an SSH token refuses to follow on
// Windows) must not be reported as a partial install — the tree can build.
test('checkRootInstall accepts a file: workspace member with no hoisted copy', () => {
  const { tempRoot, appDir } = makeTree({
    manifest: { dependencies: { '@hermes/shared': 'file:../shared' } }
  })
  const workspace = path.join(tempRoot, 'apps', 'shared')
  fs.mkdirSync(workspace, { recursive: true })
  fs.writeFileSync(path.join(workspace, 'package.json'), JSON.stringify({ name: '@hermes/shared' }), 'utf8')
  try {
    assert.deepEqual(checkRootInstall(appDir, tempRoot), { ok: true })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// The exact Windows shape: the hoisted entry EXISTS but cannot be traversed (a
// reparse point this token may not follow). A dangling symlink models that on
// POSIX — existsSync() answers false for the package.json inside it, which is the
// false negative that produced the misleading "partial install" verdict.
test('checkRootInstall accepts a file: member whose hoisted link is untraversable', () => {
  const { tempRoot, appDir } = makeTree({
    manifest: { dependencies: { '@hermes/shared': 'file:../shared' } }
  })
  const workspace = path.join(tempRoot, 'apps', 'shared')
  fs.mkdirSync(workspace, { recursive: true })
  fs.writeFileSync(path.join(workspace, 'package.json'), JSON.stringify({ name: '@hermes/shared' }), 'utf8')
  const hoisted = path.join(tempRoot, 'node_modules', '@hermes', 'shared')
  fs.mkdirSync(path.dirname(hoisted), { recursive: true })
  fs.symlinkSync(path.join(tempRoot, 'no-such-target'), hoisted)
  try {
    assert.equal(fs.existsSync(path.join(hoisted, 'package.json')), false)
    assert.deepEqual(checkRootInstall(appDir, tempRoot), { ok: true })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// A `file:` spec whose target workspace is genuinely absent is still a broken
// tree; skipping the check for local specifiers must not skip this.
test('checkRootInstall still fails when a file: dependency target is absent', () => {
  const { tempRoot, appDir } = makeTree({
    manifest: { dependencies: { '@hermes/shared': 'file:../missing' } }
  })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /@hermes\/shared/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// The hoisted copy is preferred when it is readable: a workspace member that IS
// installed keeps passing through the normal path, so the fallback cannot mask an
// install that lost its link while the workspace source is also gone.
test('checkRootInstall fails when a file: member is neither hoisted nor present', () => {
  const { tempRoot, appDir } = makeTree({
    manifest: { dependencies: { '@hermes/shared': 'file:../shared' } }
  })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /@hermes\/shared/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})
