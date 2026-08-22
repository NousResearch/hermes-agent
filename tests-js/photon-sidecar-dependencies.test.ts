import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')

const SIDECAR_PACKAGE = path.join(
  REPO_ROOT,
  'plugins',
  'platforms',
  'photon',
  'sidecar',
  'package.json'
)

test('Photon sidecar directly declares its native-voice transcoder', () => {
  const sidecarPackage = JSON.parse(fs.readFileSync(SIDECAR_PACKAGE, 'utf-8')) as {
    dependencies?: Record<string, string>
  }

  assert.ok(
    'ffmpeg-static' in (sidecarPackage.dependencies ?? {}),
    "Photon's Spectrum voice() path converts MP3 TTS to native AAC/M4A voice notes and requires ffmpeg-static as a direct sidecar dependency."
  )
})
