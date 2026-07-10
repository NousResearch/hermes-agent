import assert from 'node:assert/strict'
import test from 'node:test'

import { readTextFileBytes } from './read-file-text'

function shortReadHandle(source: Buffer, maxChunk: number) {
  const reads: { length: number; offset: number; position: number }[] = []

  return {
    reads,
    handle: {
      async read(buffer: Buffer, offset: number, length: number, position: number) {
        reads.push({ length, offset, position })
        const bytesRead = Math.min(maxChunk, length, Math.max(0, source.length - position))

        source.copy(buffer, offset, position, position + bytesRead)

        return { bytesRead }
      }
    }
  }
}

test('readTextFileBytes caps preview reads', async () => {
  const source = Buffer.from('0123456789')
  const { handle, reads } = shortReadHandle(source, 10)

  assert.equal((await readTextFileBytes(handle, source.length, 4, false)).toString(), '0123')
  assert.deepEqual(reads, [{ length: 4, offset: 0, position: 0 }])
})

test('readTextFileBytes returns complete content across short reads', async () => {
  const source = Buffer.from('# Complete report')
  const { handle, reads } = shortReadHandle(source, 3)

  assert.equal((await readTextFileBytes(handle, source.length, 4, true)).toString(), '# Complete report')
  assert.ok(reads.length > 1)
  assert.equal(reads.at(-1)?.position, 15)
})

test('readTextFileBytes stops cleanly if the file shrinks after stat', async () => {
  const source = Buffer.from('short')
  const { handle } = shortReadHandle(source, 2)

  assert.equal((await readTextFileBytes(handle, 20, 4, true)).toString(), 'short')
})
