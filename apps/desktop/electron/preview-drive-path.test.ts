import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import { previewTargetCandidates } from './local-read-path'
import { setWslBridgeActive } from './wsl-path-bridge'

// The local PREVIEW path for a delivered Windows drive spelling.
//
// A `MEDIA:` value (and a chat link to a local file) reaches the desktop as
// `/C:/Users/...` — the renderer percent-encodes the transcript's
// `C:\Users\...`, so the leading slash belongs to the ENCODING. Two seams used
// to mangle it before the file could be opened:
//
//   1. the WSL bridge read the value as a POSIX path and rewrote it into
//      `\\wsl.localhost\<distro>\C:\Users\...` — a UNC path that points
//      nowhere;
//   2. `path.resolve` on the Windows host read the stray slash as a root
//      WITHOUT a drive and landed on `C:\C:\Users\...`.
//
// The file the user asked to open was reported missing even though it sat
// exactly where the transcript said (#134855). These tests pin the CHAIN, not
// one seam: the preview target must resolve to the REAL file, with the
// percent-encoded spaces and the non-ASCII characters a real filename has.

afterEach(() => {
  setWslBridgeActive(true)
})

test('the preview chain resolves a delivered slash-prefixed drive path to the real file', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-preview-drive-'))
  const name = 'arquivo com espaço e acentuação.md'
  const file = path.join(dir, name)

  fs.writeFileSync(file, '# conteúdo', 'utf8')

  try {
    // The delivered spelling: a slash-prefixed drive form, percent-encoded
    // exactly as the renderer encodes it into the `#media:` href.
    const delivered = `/${file.replace(/\\/g, '/').replace(/ /g, '%20')}`

    // The preview rail walks these candidates in order and opens the first
    // one that exists.
    const candidates = previewTargetCandidates(delivered, value => value)
    const resolved = candidates.find(candidate => fs.existsSync(candidate))

    assert.ok(resolved, `one candidate must exist: ${JSON.stringify(candidates)}`)
    assert.equal(path.resolve(resolved as string), path.resolve(file))
    assert.equal(fs.readFileSync(resolved as string, 'utf8'), '# conteúdo')
  } finally {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

test('the preview chain keeps a POSIX path on the POSIX path it already was', () => {
  // The repair is drive-scoped: `/home/me/notes.md` is a legitimate POSIX
  // path and must never be rewritten into a drive spelling. (On a Windows host
  // the WSL bridge turns it into its UNC form — that is the bridge's own
  // documented job — but it stays a POSIX-rooted path, never `C:\...`.)
  const candidates = previewTargetCandidates('/home/me/notes.md', value => value)

  for (const candidate of candidates) {
    assert.equal(/^[A-Za-z]:/.test(candidate), false, `a POSIX path must not become a drive path: ${candidate}`)
  }
})
