import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { describe, it } from 'vitest'
import { fileURLToPath } from 'node:url'

/**
 * Regression for #72530: body stream errors after `response` (e.g.
 * net::ERR_CONTENT_LENGTH_MISMATCH) must be handled on the IncomingMessage.
 * fetchJson/fetchPublicJson already do this; fetchJsonViaOauthSession must too.
 */
describe('fetchJsonViaOauthSession body error handling', () => {
  it('attaches res.on("error") inside the electronNet response handler', () => {
    const here = path.dirname(fileURLToPath(import.meta.url))
    const src = fs.readFileSync(path.join(here, 'main.ts'), 'utf8')
    const start = src.indexOf('function fetchJsonViaOauthSession')
    assert.ok(start >= 0, 'fetchJsonViaOauthSession not found')
    const end = src.indexOf('\nfunction ', start + 1)
    const fn = src.slice(start, end > start ? end : start + 2500)
    assert.match(
      fn,
      /request\.on\(\s*['"]response['"][\s\S]*?res\.on\(\s*['"]error['"]/,
      'response IncomingMessage must listen for error (#72530)'
    )
  })
})
