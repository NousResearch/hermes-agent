import assert from 'node:assert/strict'
import test from 'node:test'

import { type FileReadSnapshot, readCompleteTextFileBytes, readTextFileBytes } from './read-file-text'

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

function snapshot(size: number, mtimeMs = 1, ctimeMs = 1): FileReadSnapshot {
  return { ctimeMs, isFile: () => true, mtimeMs, size }
}

function stableHandle(source: Buffer, maxChunk: number, snapshots: FileReadSnapshot[]) {
  const { handle, reads } = shortReadHandle(source, maxChunk)
  let statIndex = 0

  return {
    reads,
    handle: {
      ...handle,
      async stat() {
        return snapshots[Math.min(statIndex++, snapshots.length - 1)] as FileReadSnapshot
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

test('readTextFileBytes returns requested content across short reads', async () => {
  const source = Buffer.from('# Complete report')
  const { handle, reads } = shortReadHandle(source, 3)

  assert.equal((await readTextFileBytes(handle, source.length, 4, true)).toString(), '# Complete report')
  assert.ok(reads.length > 1)
  assert.equal(reads.at(-1)?.position, 15)
})

test('readCompleteTextFileBytes returns a stable complete snapshot', async () => {
  const source = Buffer.from('# Complete report')
  const stable = snapshot(source.length)
  const { handle } = stableHandle(source, 3, [stable, stable])
  const result = await readCompleteTextFileBytes(handle, 64)

  assert.equal(result.buffer.toString(), '# Complete report')
  assert.equal(result.byteSize, source.length)
})

test('readCompleteTextFileBytes rejects a file that shrinks while reading', async () => {
  const source = Buffer.from('short')
  const { handle } = stableHandle(source, 2, [snapshot(20), snapshot(source.length, 2, 2)])

  await assert.rejects(readCompleteTextFileBytes(handle, 64), /File changed while reading/)
})

test('readCompleteTextFileBytes rejects a file that grows while reading', async () => {
  const source = Buffer.from('abcdef')
  const { handle } = stableHandle(source, 6, [snapshot(3), snapshot(source.length, 2, 2)])

  await assert.rejects(readCompleteTextFileBytes(handle, 64), /File changed while reading/)
})

test('readCompleteTextFileBytes enforces the source cap on the opened handle', async () => {
  const source = Buffer.from('0123456789')
  const { handle, reads } = stableHandle(source, 10, [snapshot(source.length)])

  await assert.rejects(readCompleteTextFileBytes(handle, 4), /File too large/)
  assert.deepEqual(reads, [])
})
