import assert from 'node:assert/strict'
import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'
import test from 'node:test'

const DESKTOP_ROOT = resolve(import.meta.dirname, '..')
const SCRIPT = join(DESKTOP_ROOT, 'scripts', 'write-build-stamp.mjs')
const INSTALL_STAMP = join(DESKTOP_ROOT, 'build', 'install-stamp.json')
const BUILD_METADATA = join(DESKTOP_ROOT, 'build', 'hermes_cli', '_build_metadata.json')

function preserve(path) {
  return existsSync(path) ? readFileSync(path) : null
}

function restore(path, original) {
  if (original === null) {
    rmSync(path, { force: true })
    return
  }
  writeFileSync(path, original)
}

test('desktop build embeds the same exact revision as its install pin', () => {
  const originalStamp = preserve(INSTALL_STAMP)
  const originalMetadata = preserve(BUILD_METADATA)
  const revision = spawnSync('git', ['rev-parse', 'HEAD'], {
    cwd: DESKTOP_ROOT,
    encoding: 'utf8'
  }).stdout.trim()

  try {
    const run = spawnSync(process.execPath, [SCRIPT], {
      cwd: DESKTOP_ROOT,
      env: {
        ...process.env,
        GITHUB_SHA: revision,
        GITHUB_REF_NAME: 'main'
      },
      encoding: 'utf8'
    })
    assert.equal(run.status, 0, run.stderr)

    const installStamp = JSON.parse(readFileSync(INSTALL_STAMP, 'utf8'))
    const buildMetadata = JSON.parse(readFileSync(BUILD_METADATA, 'utf8'))
    assert.equal(installStamp.commit, revision)
    assert.deepEqual(buildMetadata, { source_revision: revision })
  } finally {
    restore(INSTALL_STAMP, originalStamp)
    restore(BUILD_METADATA, originalMetadata)
  }
})

test('desktop build refuses exact identity when CI revision mismatches checkout', () => {
  const originalStamp = preserve(INSTALL_STAMP)
  const originalMetadata = preserve(BUILD_METADATA)
  const revision = '0123456789abcdef0123456789abcdef01234567'

  try {
    const run = spawnSync(process.execPath, [SCRIPT], {
      cwd: DESKTOP_ROOT,
      env: {
        ...process.env,
        GITHUB_SHA: revision,
        GITHUB_REF_NAME: 'main'
      },
      encoding: 'utf8'
    })
    assert.equal(run.status, 0, run.stderr)

    const installStamp = JSON.parse(readFileSync(INSTALL_STAMP, 'utf8'))
    const buildMetadata = JSON.parse(readFileSync(BUILD_METADATA, 'utf8'))
    assert.equal(installStamp.dirty, true)
    assert.deepEqual(buildMetadata, { source_revision: null })
  } finally {
    restore(INSTALL_STAMP, originalStamp)
    restore(BUILD_METADATA, originalMetadata)
  }
})
