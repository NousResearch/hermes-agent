const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const {
  EMPTY_TRACKED_SOURCE_DIFF_HASH,
  bundleNeedsRebuild,
  hashText,
  readInstallStamp
} = require('./update-stamp.cjs')

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

test('bundleNeedsRebuild is false when the baked stamp and tracked source diff match HEAD', () => {
  withBundle(({ root, stampPath }) => {
    const diffHash = hashText('diff --git a/apps/desktop/src/a.ts b/apps/desktop/src/a.ts\n')
    fs.writeFileSync(
      stampPath,
      JSON.stringify({
        commit: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        trackedSourceDiffHash: diffHash
      })
    )

    assert.equal(
      bundleNeedsRebuild(root, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', fs, {
        currentTrackedSourceDiffHash: diffHash
      }),
      false
    )
  })
})

test('bundleNeedsRebuild is true when tracked source diff changed after the local build', () => {
  withBundle(({ root, stampPath }) => {
    fs.writeFileSync(
      stampPath,
      JSON.stringify({
        commit: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        trackedSourceDiffHash: hashText('old diff')
      })
    )

    assert.equal(
      bundleNeedsRebuild(root, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', fs, {
        currentTrackedSourceDiffHash: hashText('new diff')
      }),
      true
    )
  })
})

test('bundleNeedsRebuild keeps legacy stamps honest when checkout has tracked desktop edits', () => {
  withBundle(({ root, stampPath }) => {
    fs.writeFileSync(stampPath, JSON.stringify({ commit: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' }))

    assert.equal(
      bundleNeedsRebuild(root, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', fs, {
        currentTrackedSourceDiffHash: hashText('diff --git a/apps/desktop/src/a.ts b/apps/desktop/src/a.ts\n')
      }),
      true
    )
    assert.equal(
      bundleNeedsRebuild(root, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', fs, {
        currentTrackedSourceDiffHash: EMPTY_TRACKED_SOURCE_DIFF_HASH
      }),
      false
    )
  })
})

test('bundleNeedsRebuild treats malformed stamps as non-fatal', () => {
  withBundle(({ root, stampPath }) => {
    fs.writeFileSync(stampPath, '{bad json')

    assert.equal(bundleNeedsRebuild(root, 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'), false)
  })
})
