'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const {
  checkUpdaterDeps,
  UPDATER_DIR,
  VENDOR_NODE_MODULES,
  MANIFEST_NAME,
} = require('../scripts/assert-updater-deps.cjs')

// The real staged closure the CI log reports ("staged 16 packages"): the
// electron-updater@6.x production dependency tree. The fixture reproduces the
// packaged layout (Resources/updater-deps/vendor/node_modules/<pkg>) so the gate
// runs against a byte-faithful stand-in without any monorepo deps installed.
// builder-util-runtime is placed as a version-pinned NESTED copy to exercise the
// manifest `dir` carrying a deeper node_modules path.
const CLOSURE = [
  { name: 'electron-updater', dir: 'electron-updater', entry: 'electron-updater/out/main.js' },
  { name: 'semver', dir: 'semver', entry: 'semver/index.js' },
  { name: 'js-yaml', dir: 'js-yaml', entry: 'js-yaml/index.js' },
  { name: 'argparse', dir: 'argparse', entry: 'argparse/argparse.js' },
  { name: 'fs-extra', dir: 'fs-extra', entry: 'fs-extra/lib/index.js' },
  { name: 'graceful-fs', dir: 'graceful-fs', entry: 'graceful-fs/graceful-fs.js' },
  { name: 'jsonfile', dir: 'jsonfile', entry: 'jsonfile/index.js' },
  { name: 'universalify', dir: 'universalify', entry: 'universalify/index.js' },
  { name: 'lazy-val', dir: 'lazy-val', entry: 'lazy-val/out/main.js' },
  { name: 'lodash.isequal', dir: 'lodash.isequal', entry: 'lodash.isequal/index.js' },
  { name: 'tiny-typed-emitter', dir: 'tiny-typed-emitter', entry: 'tiny-typed-emitter/lib/index.js' },
  { name: 'lodash.escaperegexp', dir: 'lodash.escaperegexp', entry: 'lodash.escaperegexp/index.js' },
  {
    name: 'builder-util-runtime',
    dir: 'electron-updater/node_modules/builder-util-runtime',
    entry: 'electron-updater/node_modules/builder-util-runtime/out/index.js',
  },
  { name: 'sax', dir: 'sax', entry: 'sax/lib/sax.js' },
  { name: 'debug', dir: 'debug', entry: 'debug/src/index.js' },
  { name: 'ms', dir: 'ms', entry: 'ms/index.js' },
]

// Build a fake packaged Resources dir with the given closure fully materialized
// (each package's dir + its entry file). Returns the resources dir path.
function makeFixture(root, closure = CLOSURE, { writeManifest = true } = {}) {
  const resourcesDir = path.join(root, 'Resources')
  const vendorNm = path.join(resourcesDir, UPDATER_DIR, VENDOR_NODE_MODULES)
  for (const pkg of closure) {
    // entry may be null (stager: unresolvable main -> dir check alone).
    if (pkg.entry) {
      const entryAbs = path.join(vendorNm, ...pkg.entry.split('/'))
      fs.mkdirSync(path.dirname(entryAbs), { recursive: true })
      fs.writeFileSync(entryAbs, `module.exports = {} // ${pkg.name}\n`)
    }
    // A package.json so the dir is a plausible package (the gate checks the dir
    // exists; this keeps the fixture realistic).
    const pkgJson = path.join(vendorNm, ...pkg.dir.split('/'), 'package.json')
    fs.mkdirSync(path.dirname(pkgJson), { recursive: true })
    fs.writeFileSync(pkgJson, JSON.stringify({ name: pkg.name }))
  }
  if (writeManifest) {
    const manifestPath = path.join(resourcesDir, UPDATER_DIR, MANIFEST_NAME)
    fs.writeFileSync(
      manifestPath,
      JSON.stringify({ root: 'electron-updater', packages: closure }, null, 2)
    )
  }
  return resourcesDir
}

function withTempFixture(fn) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hc436-updater-gate-'))
  try {
    return fn(root)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}

test('PASS: full 16-package closure present -> ok, no missing', () => {
  withTempFixture((root) => {
    const resourcesDir = makeFixture(root)
    const res = checkUpdaterDeps(resourcesDir)
    assert.equal(res.ok, true, `expected ok, got missing=${JSON.stringify(res.missing)}`)
    assert.equal(res.checked, CLOSURE.length)
    assert.equal(res.checked, 16) // matches the CI "staged 16 packages" log
    assert.deepEqual(res.missing, [])
  })
})

test('FAIL: a dropped package -> not ok, and the error names it', () => {
  withTempFixture((root) => {
    const resourcesDir = makeFixture(root)
    // Simulate a stage that silently dropped one dep (the hc-435 failure mode,
    // but for a transitive package rather than the root).
    const dropped = 'js-yaml'
    const vendorNm = path.join(resourcesDir, UPDATER_DIR, VENDOR_NODE_MODULES)
    fs.rmSync(path.join(vendorNm, dropped), { recursive: true, force: true })

    const res = checkUpdaterDeps(resourcesDir)
    assert.equal(res.ok, false)
    assert.ok(
      res.missing.some((m) => m.includes(`/${dropped}`)),
      `expected a missing entry naming "${dropped}", got ${JSON.stringify(res.missing)}`
    )
    // Other packages still pass -- the report is specific, not a blanket fail.
    assert.ok(!res.missing.some((m) => m.includes('/semver')))
  })
})

test('FAIL: the root electron-updater entry dropped -> named explicitly', () => {
  withTempFixture((root) => {
    const resourcesDir = makeFixture(root)
    const vendorNm = path.join(resourcesDir, UPDATER_DIR, VENDOR_NODE_MODULES)
    // Remove ONLY the root entry file (dir + package.json remain) -- the exact
    // hc-435 regression: electron-updater present but its require()-main gone.
    fs.rmSync(path.join(vendorNm, 'electron-updater', 'out', 'main.js'), { force: true })

    const res = checkUpdaterDeps(resourcesDir)
    assert.equal(res.ok, false)
    assert.ok(
      res.missing.some((m) => m.includes('electron-updater/out/main.js')),
      `expected the root entry to be flagged, got ${JSON.stringify(res.missing)}`
    )
  })
})

test('FAIL: a version-pinned nested dep dropped -> named with its nested path', () => {
  withTempFixture((root) => {
    const resourcesDir = makeFixture(root)
    const vendorNm = path.join(resourcesDir, UPDATER_DIR, VENDOR_NODE_MODULES)
    fs.rmSync(path.join(vendorNm, 'electron-updater', 'node_modules', 'builder-util-runtime'), {
      recursive: true,
      force: true,
    })

    const res = checkUpdaterDeps(resourcesDir)
    assert.equal(res.ok, false)
    assert.ok(
      res.missing.some((m) => m.includes('electron-updater/node_modules/builder-util-runtime')),
      `expected nested dep flagged, got ${JSON.stringify(res.missing)}`
    )
  })
})

test('FAIL: manifest absent (whole updater-deps copy dropped) -> not ok', () => {
  withTempFixture((root) => {
    // Materialize packages but NOT the manifest: mimics extraResources dropping
    // the whole updater-deps subtree, or a build that never re-staged.
    const resourcesDir = makeFixture(root, CLOSURE, { writeManifest: false })
    const res = checkUpdaterDeps(resourcesDir)
    assert.equal(res.ok, false)
    assert.ok(res.missing.some((m) => m.includes(MANIFEST_NAME)))
  })
})

test('PASS: manifest entry null (unresolvable main) -> dir check alone suffices', () => {
  withTempFixture((root) => {
    const closure = CLOSURE.map((p) => (p.name === 'sax' ? { ...p, entry: null } : p))
    const resourcesDir = makeFixture(root, closure)
    const res = checkUpdaterDeps(resourcesDir)
    assert.equal(res.ok, true, JSON.stringify(res.missing))
  })
})

test('FAIL: manifest lists zero packages -> not ok', () => {
  withTempFixture((root) => {
    const resourcesDir = makeFixture(root)
    const manifestPath = path.join(resourcesDir, UPDATER_DIR, MANIFEST_NAME)
    fs.writeFileSync(manifestPath, JSON.stringify({ root: 'electron-updater', packages: [] }))
    const res = checkUpdaterDeps(resourcesDir)
    assert.equal(res.ok, false)
    assert.ok(res.missing.some((m) => m.includes('zero packages')))
  })
})
