/**
 * Contract coverage for the browser-owner's fail-closed send control allowlist.
 *
 * The owner runs against an authenticated browser profile, so this test stays
 * local: it verifies that the send-candidate block includes the exact visible
 * button label observed in the current Gemini DOM without exercising a live
 * service or credentials.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const OWNER_PATH = path.resolve(__dirname, '..', 'tools', 'web_gemini_owner.js')
const OWNER_SOURCE = fs.readFileSync(OWNER_PATH, 'utf8')

test('owner allowlists the visible Send message button when aria-label is empty', () => {
  const start = OWNER_SOURCE.indexOf('const sendCandidates = [')
  assert.ok(start >= 0, 'owner send-candidate block is missing')

  const end = OWNER_SOURCE.indexOf('  ];', start)
  assert.ok(end > start, 'owner send-candidate block is not terminated')

  const block = OWNER_SOURCE.slice(start, end)
  assert.match(
    block,
    /page\.locator\('button'\)\.filter\(\{\s*hasText:\s*\/\^Send message\$\/i\s*\}\)\.last\(\)/,
    'owner must retain an explicitly allowlisted exact visible-text fallback for the current DOM',
  )
})
