import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import { stageMacLocalizedInfoPlist } from './after-pack.mjs'

test('stages the Arabic InfoPlist localization in the macOS app bundle', async () => {
  const appOutDir = await mkdtemp(path.join(os.tmpdir(), 'hermes-after-pack-'))

  try {
    await stageMacLocalizedInfoPlist({
      appOutDir,
      packager: { appInfo: { productFilename: 'Hermes Test' } }
    })

    const staged = await readFile(
      path.join(appOutDir, 'Hermes Test.app', 'Contents', 'Resources', 'ar.lproj', 'InfoPlist.strings'),
      'utf8'
    )

    assert.match(staged, /"CFBundleDisplayName" = "هرمس";/)
    assert.match(staged, /"NSMicrophoneUsageDescription" = "يستخدم هرمس الميكروفون/)
  } finally {
    await rm(appOutDir, { force: true, recursive: true })
  }
})
