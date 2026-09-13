import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import {
  BUILD_CRITICAL_PACKAGES as BUILD_CRITICAL,
  checkPnpResolution,
  checkRootInstall,
  findPnpManifest,
  requiredPackages
} from '../scripts/assert-root-install.mjs'
import { electronBundleOptions } from '../scripts/electron-bundle-options.mjs'

// Build a throwaway repo shaped like this one: an app workspace whose
// dependencies are hoisted to the repo root, which is what the guard walks.
// `manifest` is merged into the app's package.json so tests can declare
// dependencies the guard is expected to read. `parentDir` places the repo
// inside an existing directory so tests can plant files above the repo root.
function makeTree({
  rootPackages = BUILD_CRITICAL,
  react = '19.2.7',
  reactDom = '19.2.7',
  manifest = {},
  parentDir = os.tmpdir()
} = {}) {
  const tempRoot = fs.mkdtempSync(path.join(parentDir, 'hermes-assert-root-'))
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

test('checkRootInstall fails when electron is missing', () => {
  const { tempRoot, appDir } = makeTree({
    rootPackages: BUILD_CRITICAL.filter(name => name !== 'electron')
  })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /electron/)
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

// esbuild adopts a Yarn PnP manifest from any ancestor of its working directory,
// so a real manifest above the repo (typically a stray one in a home directory)
// breaks `bundle-electron-main.mjs` with `Could not resolve "..."` over a complete
// npm install. Only a manifest esbuild actually loads may fail the check: an empty
// file, a directory, an unrelated file or a corrupt manifest with the same name must
// not block a build. These fixtures run the real esbuild resolver.

// A minimal Yarn PnP state for a lone root workspace, the shape `yarn` leaves
// behind when run in a parent folder. `declared` lists packages the manifest maps
// to this repo's hoisted node_modules; anything else is forbidden.
function pnpState(declared = []) {
  const hard = name => [name, [['npm:1.0.0', { packageLocation: `./repo/node_modules/${name}/`, packageDependencies: [[name, 'npm:1.0.0']], linkType: 'HARD' }]]]
  const deps = declared.map(name => [name, 'npm:1.0.0'])
  return JSON.stringify({
    dependencyTreeRoots: [{ name: 'stray-root', reference: 'workspace:.' }],
    enableTopLevelFallback: true,
    fallbackExclusionList: [['stray-root', ['workspace:.']]],
    fallbackPool: [],
    ignorePatternData: null,
    packageRegistryData: [
      [null, [[null, { packageLocation: './', packageDependencies: deps, linkType: 'SOFT' }]]],
      ['stray-root', [['workspace:.', { packageLocation: './', packageDependencies: [['stray-root', 'workspace:.'], ...deps], linkType: 'SOFT' }]]],
      ...declared.map(hard)
    ]
  })
}

const PNP_FIXTURES = {
  'valid .pnp.data.json': outer => fs.writeFileSync(path.join(outer, '.pnp.data.json'), pnpState()),
  'valid .pnp.cjs': outer => fs.writeFileSync(path.join(outer, '.pnp.cjs'), [
    '"use strict";',
    `const RAW_RUNTIME_STATE =\n'${pnpState()}';`,
    'function $$SETUP_STATE(hydrateRuntimeState, basePath) {',
    '  return hydrateRuntimeState(JSON.parse(RAW_RUNTIME_STATE), {basePath: basePath || __dirname});',
    '}',
    ''
  ].join('\n')),
  'partial .pnp.data.json': outer => fs.writeFileSync(path.join(outer, '.pnp.data.json'), pnpState(['probe-pkg'])),
  'empty .pnp.cjs': outer => fs.writeFileSync(path.join(outer, '.pnp.cjs'), ''),
  '.pnp.cjs directory': outer => fs.mkdirSync(path.join(outer, '.pnp.cjs')),
  'unrelated .pnp.js': outer => fs.writeFileSync(path.join(outer, '.pnp.js'), 'module.exports = {}\n'),
  'corrupt .pnp.data.json': outer => fs.writeFileSync(path.join(outer, '.pnp.data.json'), '{"packageRegistryData": ['),
}

// An app workspace nested under `outer` whose entry imports two packages hoisted
// to the repo root, plus bundle options shaped like electron-bundle-options.mjs.
function makePnpTree(fixture) {
  const outer = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-assert-pnp-'))
  const appDir = path.join(outer, 'repo', 'apps', 'desktop')
  fs.mkdirSync(path.join(appDir, 'electron'), { recursive: true })
  fs.writeFileSync(path.join(outer, 'repo', 'package.json'), JSON.stringify({ name: 'repo', private: true, workspaces: ['apps/*'] }))
  fs.writeFileSync(path.join(appDir, 'package.json'), JSON.stringify({ name: 'desktop', dependencies: { 'probe-pkg': '1.0.0', 'second-pkg': '1.0.0' } }))
  for (const name of ['probe-pkg', 'second-pkg']) {
    const pkgDir = path.join(outer, 'repo', 'node_modules', name)
    fs.mkdirSync(pkgDir, { recursive: true })
    fs.writeFileSync(path.join(pkgDir, 'package.json'), JSON.stringify({ name, version: '1.0.0', main: 'index.js' }))
    fs.writeFileSync(path.join(pkgDir, 'index.js'), `module.exports = ${JSON.stringify(name)}\n`)
  }
  const entry = path.join(appDir, 'electron', 'main.ts')
  fs.writeFileSync(entry, "import probe from 'probe-pkg'\nimport second from 'second-pkg'\nconsole.log(probe, second)\n")
  if (fixture) PNP_FIXTURES[fixture](outer)
  const bundleOptions = [{ entryPoints: [entry], bundle: true, platform: 'node', format: 'esm', external: ['electron'] }]
  return { outer, appDir, bundleOptions }
}

test('checkPnpResolution passes without any PnP manifest', async () => {
  const { outer, appDir, bundleOptions } = makePnpTree(null)
  try {
    assert.deepEqual(await checkPnpResolution(appDir, { bundleOptions }), { ok: true })
  } finally {
    fs.rmSync(outer, { recursive: true, force: true })
  }
})

for (const fixture of ['valid .pnp.data.json', 'valid .pnp.cjs']) {
  test(`checkPnpResolution fails and names the manifest for a ${fixture} above the repo`, async () => {
    const { outer, appDir, bundleOptions } = makePnpTree(fixture)
    try {
      const result = await checkPnpResolution(appDir, { bundleOptions })
      assert.equal(result.ok, false)
      assert.ok(result.error.includes(path.join(fs.realpathSync(outer), fixture.split(' ')[1])), result.error)
      assert.match(result.error, /Plug'n'Play/)
      assert.match(result.error, /"probe-pkg"/)
      assert.match(result.error, /"second-pkg"/)
    } finally {
      fs.rmSync(outer, { recursive: true, force: true })
    }
  })
}

// A manifest can declare some bundle dependencies and omit others. Probing one
// package would pass here while the real bundle still fails, so every import
// reachable from the entrypoints must be resolved.
test('checkPnpResolution names the omitted package when the manifest declares only some bundle dependencies', async () => {
  const { outer, appDir, bundleOptions } = makePnpTree('partial .pnp.data.json')
  try {
    const result = await checkPnpResolution(appDir, { bundleOptions })
    assert.equal(result.ok, false)
    assert.match(result.error, /"second-pkg"/)
    assert.doesNotMatch(result.error, /"probe-pkg"/)
  } finally {
    fs.rmSync(outer, { recursive: true, force: true })
  }
})

for (const fixture of ['empty .pnp.cjs', '.pnp.cjs directory', 'unrelated .pnp.js', 'corrupt .pnp.data.json']) {
  test(`checkPnpResolution does not block the build for a stray ${fixture} above the repo`, async () => {
    const { outer, appDir, bundleOptions } = makePnpTree(fixture)
    try {
      assert.deepEqual(await checkPnpResolution(appDir, { bundleOptions }), { ok: true })
    } finally {
      fs.rmSync(outer, { recursive: true, force: true })
    }
  })
}

// Without explicit options the check resolves the real Electron bundles, so their
// entrypoints must exist where bundle-electron-main.mjs builds them from.
test('electronBundleOptions points at the real main and preload entrypoints', () => {
  const desktopDir = path.resolve(import.meta.dirname, '..')
  const options = electronBundleOptions(desktopDir)
  assert.deepEqual(options.map(o => o.format), ['esm', 'cjs'])
  for (const { entryPoints, external } of options) {
    assert.ok(fs.existsSync(entryPoints[0]), entryPoints[0])
    assert.ok(external.includes('electron'))
  }
})

test('findPnpManifest detects every manifest name and returns null when absent', () => {
  for (const name of ['.pnp.cjs', '.pnp.js', '.pnp.data.json']) {
    const outer = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-assert-pnp-'))
    const nested = path.join(outer, 'repo', 'apps', 'desktop')
    fs.mkdirSync(nested, { recursive: true })
    try {
      assert.equal(findPnpManifest(nested), null)
      fs.writeFileSync(path.join(outer, name), '', 'utf8')
      assert.equal(findPnpManifest(nested), path.join(outer, name))
    } finally {
      fs.rmSync(outer, { recursive: true, force: true })
    }
  }
})
