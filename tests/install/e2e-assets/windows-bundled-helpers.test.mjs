// windows-bundled-helpers — unit tests for the pure helpers of the
// Windows packaged-app E2E arm. Pure manifest validation plus real descriptor
// CLI calls through the release Python module; no Windows installation needed.
//
//   node --test tests/install/e2e-assets/windows-bundled-helpers.test.mjs
import { test } from 'node:test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { byteRange } from './windows-bundled-helpers.mjs'

test('native acceptance descriptor CLI uses explicit publisher and a stable subscription across upgrades', () => {
  const feed = fs.mkdtempSync(path.join(os.tmpdir(), 'wbh-descriptor-'))
  try {
    for (const version of ['1.2.3.0', '1.2.4.31']) {
      execFileSync(process.execPath, [fileURLToPath(new URL('./windows-bundled-helpers.mjs', import.meta.url)),
        'descriptor', '--feed', feed, '--base-url', 'http://127.0.0.1:9',
        '--identity', 'Fixture.App', '--publisher', 'CN=Fixture & "Team"',
        '--version', version, '--bundle', `${version}.msixbundle`, '--descriptor-filename', 'update.appinstaller',
      ])
      const xml = fs.readFileSync(path.join(feed, 'update.appinstaller'), 'utf8')
      assert.match(xml, /Publisher="CN=Fixture &amp; &quot;Team&quot;"/)
      assert.match(xml, /Uri="http:\/\/127.0.0.1:9\/update.appinstaller"/)
      assert.ok(xml.includes(`Uri="http://127.0.0.1:9/${version}.msixbundle"`))
      assert.equal((xml.match(new RegExp(`Version="${version.replaceAll('.', '\\.')}"`, 'g')) || []).length, 2)
      assert.match(xml, /HoursBetweenUpdateChecks="12"/)
    }
  } finally { fs.rmSync(feed, { recursive: true, force: true }) }
})

test('the feed honours byte ranges, which App Installer requires (0x80D05011 without them)', () => {
  assert.equal(byteRange(undefined, 1000), null)
  assert.deepEqual(byteRange('bytes=0-99', 1000), { start: 0, end: 99 })
  assert.deepEqual(byteRange('bytes=900-', 1000), { start: 900, end: 999 })
  assert.deepEqual(byteRange('bytes=-100', 1000), { start: 900, end: 999 })
  assert.deepEqual(byteRange('bytes=990-5000', 1000), { start: 990, end: 999 })
  assert.equal(byteRange('bytes=1000-', 1000), 'unsatisfiable')
  assert.equal(byteRange('bytes=0-1,5-9', 1000), null)
})
