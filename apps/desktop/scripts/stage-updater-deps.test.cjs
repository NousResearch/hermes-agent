// Regression tests for stage-updater-deps.cjs entry resolution — the exact
// class of bug that failed the 0.16.6 release gates on BOTH platforms: raw
// package.json mains like ms's "./index" / tiny-typed-emitter's "lib/index"
// were recorded verbatim into the manifest, and the integrity gate's literal
// isFile() couldn't see the files require() resolves. entryRelFor must return
// the REAL resolved file (relative to the package dir), or null when nothing
// resolves (the gate then relies on its dir-presence check).
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const { entryRelFor } = require('./stage-updater-deps.cjs')

function withTempPkg(pkgJson, files, fn) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hc436-entryres-'))
  try {
    if (pkgJson !== undefined) {
      fs.writeFileSync(path.join(root, 'package.json'), JSON.stringify(pkgJson))
    }
    for (const rel of files) {
      const abs = path.join(root, ...rel.split('/'))
      fs.mkdirSync(path.dirname(abs), { recursive: true })
      fs.writeFileSync(abs, 'module.exports = {}\n')
    }
    return fn(root)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}

test('main "./index" without extension (real-world: ms) -> resolved index.js', () => {
  withTempPkg({ name: 'ms', main: './index' }, ['index.js'], (root) => {
    assert.equal(entryRelFor(root), 'index.js')
  })
})

test('main "lib/index" without extension (real-world: tiny-typed-emitter) -> lib/index.js', () => {
  withTempPkg({ name: 'tte', main: 'lib/index' }, ['lib/index.js'], (root) => {
    assert.equal(entryRelFor(root), path.join('lib', 'index.js'))
  })
})

test('main with explicit extension -> unchanged real file', () => {
  withTempPkg({ name: 'semver', main: 'index.js' }, ['index.js'], (root) => {
    assert.equal(entryRelFor(root), 'index.js')
  })
})

test('no main field -> default index.js resolution', () => {
  withTempPkg({ name: 'plain' }, ['index.js'], (root) => {
    assert.equal(entryRelFor(root), 'index.js')
  })
})

test('main pointing at a directory -> its index.js', () => {
  withTempPkg({ name: 'dirmain', main: './lib' }, ['lib/index.js'], (root) => {
    assert.equal(entryRelFor(root), path.join('lib', 'index.js'))
  })
})

test('nothing resolvable (main missing, no index.js) -> null, never a lying path', () => {
  withTempPkg({ name: 'exotic', main: './does-not-exist' }, [], (root) => {
    assert.equal(entryRelFor(root), null)
  })
})

test('unreadable package dir -> null', () => {
  assert.equal(entryRelFor(path.join(os.tmpdir(), 'hc436-definitely-missing-dir')), null)
})
