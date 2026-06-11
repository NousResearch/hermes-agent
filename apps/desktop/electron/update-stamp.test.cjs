const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const { bundleNeedsRebuild, readInstallStamp } = require('./update-stamp.cjs')

function withBundle(fn) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-stamp-'))
  const resources = path.join(root, 'Contents', 'Resources')
  fs.mkdirSync(resources, { recursive: true })
  try {
    return fn({ root, stampPath: path.join(resources, 'install-stamp.json') })
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}

test('readInstallStamp returns null when the bundle or stamp is missing', () => {
  assert.equal(readInstallStamp(null), null)
  withBundle(({ root }) => {
    assert.equal(readInstallStamp(root), null)
  })
})

test('bundleNeedsRebuild is true when the baked stamp differs from HEAD', () => {
  withBundle(({ root, stampPath }) => {
    fs.writeFileSync(stampPath, JSON.stringify({ commit: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' }))

    assert.equal(bundleNeedsRebuild(root, 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'), true)
  })
})

test('bundleNeedsRebuild is false when the baked stamp matches HEAD', () => {
  withBundle(({ root, stampPath }) => {
    fs.writeFileSync(stampPath, JSON.stringify({ commit: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' }))

    assert.equal(bundleNeedsRebuild(root, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'), false)
  })
})

test('bundleNeedsRebuild treats malformed stamps as non-fatal', () => {
  withBundle(({ root, stampPath }) => {
    fs.writeFileSync(stampPath, '{bad json')

    assert.equal(bundleNeedsRebuild(root, 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'), false)
  })
})
