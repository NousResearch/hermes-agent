import assert from 'node:assert/strict'
import test from 'node:test'

import afterPack from './after-pack.mjs'

const missingExecutableContext = {
  electronPlatformName: 'win32',
  appOutDir: 'Z:\\hermes-release-test\\missing',
  packager: {
    appInfo: {
      productFilename: 'Hermes'
    }
  }
}

test('Windows branding is best-effort locally and mandatory for releases', async () => {
  const previous = process.env.HERMES_DESKTOP_RELEASE_SIGNING

  try {
    delete process.env.HERMES_DESKTOP_RELEASE_SIGNING
    await assert.doesNotReject(afterPack(missingExecutableContext))

    process.env.HERMES_DESKTOP_RELEASE_SIGNING = '1'
    await assert.rejects(afterPack(missingExecutableContext), /release exe identity stamp failed/)
  } finally {
    if (previous === undefined) {
      delete process.env.HERMES_DESKTOP_RELEASE_SIGNING
    } else {
      process.env.HERMES_DESKTOP_RELEASE_SIGNING = previous
    }
  }
})
