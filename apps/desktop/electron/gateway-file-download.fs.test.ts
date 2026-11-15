// Real-filesystem witnesses for the failure-atomic save contract (#96597).
//
// The unit tests in gateway-file-download.test.ts prove the pump's control flow
// against fakes. These run the exact production deps (`fsPumpDeps()`) against
// node:fs in a scratch directory and assert the user-visible invariants
// byte-for-byte: a pre-existing destination survives every failure mode this
// harness can force, a pre-existing file at the temp name survives a pre-open
// collision, and no owned `.part` file is ever left behind.

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Readable } from 'node:stream'

import { afterEach, beforeEach, test } from 'vitest'

import { fsPumpDeps, pumpStreamToFile, writeBufferToFile } from './gateway-file-download'

let dir = ''

beforeEach(async () => {
  dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-download-fs-'))
})

afterEach(async () => {
  await fs.promises.rm(dir, { force: true, recursive: true })
})

// A body that delivers `chunks` then fails with `error` (or ends cleanly when
// `error` is omitted). Readable satisfies the pump's ReadableLike shape.
function body(chunks: string[], error?: Error): Readable {
  let i = 0

  return new Readable({
    read() {
      if (i < chunks.length) {
        this.push(Buffer.from(chunks[i++]))

        return
      }

      if (error) {
        this.destroy(error)

        return
      }

      this.push(null)
    }
  })
}

async function listing(): Promise<string[]> {
  return (await fs.promises.readdir(dir)).sort()
}

test('a completed download replaces the destination and leaves no temp file', async () => {
  const dest = path.join(dir, 'report.bin')

  await fs.promises.writeFile(dest, 'OLD CONTENT')

  await pumpStreamToFile(body(['new ', 'content']), dest, fsPumpDeps())

  assert.equal(await fs.promises.readFile(dest, 'utf8'), 'new content')
  assert.deepEqual(await listing(), ['report.bin'])
})

test('a download that fails mid-stream leaves the pre-existing destination byte-for-byte and no temp file', async () => {
  const dest = path.join(dir, 'report.bin')
  const original = Buffer.from('OLD CONTENT THAT MUST SURVIVE')

  await fs.promises.writeFile(dest, original)

  await assert.rejects(
    pumpStreamToFile(body(['partial'], new Error('socket hang up')), dest, fsPumpDeps()),
    /socket hang up/
  )

  assert.ok(original.equals(await fs.promises.readFile(dest)), 'destination bytes must be unchanged')
  assert.deepEqual(await listing(), ['report.bin'], 'no .part file may remain')
})

test('a download into a name with no existing file that fails leaves nothing behind', async () => {
  const dest = path.join(dir, 'fresh.bin')

  await assert.rejects(pumpStreamToFile(body(['partial'], new Error('reset')), dest, fsPumpDeps()), /reset/)

  assert.deepEqual(await listing(), [])
})

// The reviewer-requested regression: seed the candidate temp path with known
// bytes, force the exclusive open to fail with EEXIST, and prove those bytes
// remain untouched and no rename occurred.
test('a pre-open EEXIST collision leaves the seeded temp file and the destination untouched', async () => {
  const dest = path.join(dir, 'report.bin')
  const pinnedTemp = path.join(dir, '.hermes-download-pinned.part')
  const seeded = Buffer.from('SOMEONE ELSES BYTES')
  const original = Buffer.from('OLD CONTENT')

  await fs.promises.writeFile(dest, original)
  await fs.promises.writeFile(pinnedTemp, seeded)

  const deps = { ...fsPumpDeps(), tempPathFor: () => pinnedTemp }

  await assert.rejects(pumpStreamToFile(body(['new content']), dest, deps), (err: NodeJS.ErrnoException) => {
    assert.equal(err.code, 'EEXIST')

    return true
  })

  assert.ok(seeded.equals(await fs.promises.readFile(pinnedTemp)), 'the colliding file must not be unlinked')
  assert.ok(original.equals(await fs.promises.readFile(dest)), 'destination must not be renamed over')
  assert.deepEqual(await listing(), ['.hermes-download-pinned.part', 'report.bin'])
})

test('a failed final rename removes the owned temp file and leaves the destination as it was', async () => {
  // A directory at the destination makes rename(2) fail on every platform.
  const dest = path.join(dir, 'report.bin')

  await fs.promises.mkdir(dest)
  await fs.promises.writeFile(path.join(dest, 'keep.txt'), 'inside')

  await assert.rejects(pumpStreamToFile(body(['new content']), dest, fsPumpDeps()))

  assert.ok((await fs.promises.stat(dest)).isDirectory(), 'destination directory must survive')
  assert.equal(await fs.promises.readFile(path.join(dest, 'keep.txt'), 'utf8'), 'inside')
  assert.deepEqual(await listing(), ['report.bin'], 'the owned temp file must be cleaned up')
})

test('writeBufferToFile replaces the destination atomically and leaves no temp file', async () => {
  const dest = path.join(dir, 'fallback.bin')

  await fs.promises.writeFile(dest, 'OLD CONTENT')

  await writeBufferToFile(Buffer.from('data-url payload'), dest, fsPumpDeps())

  assert.equal(await fs.promises.readFile(dest, 'utf8'), 'data-url payload')
  assert.deepEqual(await listing(), ['fallback.bin'])
})

test('writeBufferToFile into a missing directory fails without creating anything', async () => {
  const dest = path.join(dir, 'missing-subdir', 'fallback.bin')

  await assert.rejects(writeBufferToFile(Buffer.from('payload'), dest, fsPumpDeps()), (err: NodeJS.ErrnoException) => {
    assert.equal(err.code, 'ENOENT')

    return true
  })

  assert.deepEqual(await listing(), [])
})
