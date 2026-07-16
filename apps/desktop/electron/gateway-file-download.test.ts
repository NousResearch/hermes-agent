import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import test from 'node:test'

import {
  filenameFromContentDisposition,
  gatewayFilePath,
  isNotFoundError,
  parseDataUrlToBuffer,
  pumpStreamToFile
} from './gateway-file-download'

// A Readable-like response driven manually in tests.
class FakeResponse extends EventEmitter {
  paused = false
  resumed = false
  destroyed = false

  pause() {
    this.paused = true
  }

  resume() {
    this.resumed = true
  }

  destroy() {
    this.destroyed = true
  }
}

// A write stream that records writes and lets tests control backpressure.
class FakeWriteStream extends EventEmitter {
  chunks: Buffer[] = []
  ended = false
  destroyed = false
  private writeReturns: boolean[]

  constructor(writeReturns: boolean[] = []) {
    super()
    this.writeReturns = writeReturns
  }

  write(chunk: Buffer): boolean {
    this.chunks.push(chunk)

    return this.writeReturns.length ? this.writeReturns.shift()! : true
  }

  end(cb: () => void) {
    this.ended = true
    cb()
  }

  destroy() {
    this.destroyed = true
  }
}

test('pumpStreamToFile streams chunks to the destination without buffering the whole body', async () => {
  const res = new FakeResponse()
  const ws = new FakeWriteStream()
  const unlinked: string[] = []

  const promise = pumpStreamToFile(res as never, '/tmp/out.bin', {
    createWriteStream: () => ws as never,
    unlink: async p => {
      unlinked.push(p)
    }
  })

  res.emit('data', Buffer.from('abc'))
  res.emit('data', Buffer.from('def'))
  res.emit('end')

  await promise

  assert.equal(Buffer.concat(ws.chunks).toString('utf8'), 'abcdef')
  assert.equal(ws.ended, true)
  assert.deepEqual(unlinked, []) // success -> no cleanup
})

test('pumpStreamToFile applies backpressure: pauses on a full buffer and resumes on drain', async () => {
  const res = new FakeResponse()
  const ws = new FakeWriteStream([false]) // first write signals "buffer full"

  const promise = pumpStreamToFile(res as never, '/tmp/out.bin', {
    createWriteStream: () => ws as never,
    unlink: async () => {}
  })

  res.emit('data', Buffer.from('big-chunk'))
  assert.equal(res.paused, true, 'source should be paused when write() returns false')
  assert.equal(res.resumed, false)

  ws.emit('drain')
  assert.equal(res.resumed, true, 'source should resume after the write stream drains')

  res.emit('end')
  await promise
})

test('pumpStreamToFile unlinks the partial file and rejects on a write error', async () => {
  const res = new FakeResponse()
  const ws = new FakeWriteStream()
  const unlinked: string[] = []

  const promise = pumpStreamToFile(res as never, '/tmp/partial.bin', {
    createWriteStream: () => ws as never,
    unlink: async p => {
      unlinked.push(p)
    }
  })

  res.emit('data', Buffer.from('abc'))
  ws.emit('error', new Error('ENOSPC: disk full'))

  await assert.rejects(promise, /disk full/)
  assert.deepEqual(unlinked, ['/tmp/partial.bin'])
  assert.equal(res.destroyed, true, 'source should be torn down on write failure')
})

test('pumpStreamToFile unlinks the partial file and rejects on a response error', async () => {
  const res = new FakeResponse()
  const ws = new FakeWriteStream()
  const unlinked: string[] = []

  const promise = pumpStreamToFile(res as never, '/tmp/partial.bin', {
    createWriteStream: () => ws as never,
    unlink: async p => {
      unlinked.push(p)
    }
  })

  res.emit('data', Buffer.from('abc'))
  res.emit('error', new Error('socket hang up'))

  await assert.rejects(promise, /socket hang up/)
  assert.deepEqual(unlinked, ['/tmp/partial.bin'])
})

test('parseDataUrlToBuffer decodes base64 payloads', () => {
  const buffer = parseDataUrlToBuffer('data:text/markdown;base64,IyByZXBvcnQ=')

  assert.equal(buffer.toString('utf8'), '# report')
})

test('parseDataUrlToBuffer decodes percent-encoded (non-base64) payloads', () => {
  const buffer = parseDataUrlToBuffer('data:text/plain,hello%20world')

  assert.equal(buffer.toString('utf8'), 'hello world')
})

test('parseDataUrlToBuffer throws on a malformed data URL', () => {
  assert.throws(() => parseDataUrlToBuffer('not-a-data-url'), /Malformed data URL/)
})

test('filenameFromContentDisposition prefers filename* and reduces to a basename', () => {
  assert.equal(
    filenameFromContentDisposition("attachment; filename*=UTF-8''report%20with%20spaces.pdf"),
    'report with spaces.pdf'
  )
  assert.equal(filenameFromContentDisposition('attachment; filename="report.md"'), 'report.md')
  // A traversal attempt in the header cannot escape the chosen directory.
  assert.equal(filenameFromContentDisposition('attachment; filename="../../etc/passwd"'), 'passwd')
  assert.equal(filenameFromContentDisposition(''), '')
  assert.equal(filenameFromContentDisposition(undefined), '')
})

test('gatewayFilePath normalizes bare paths and file:// URLs', () => {
  assert.equal(gatewayFilePath('/Users/me/report.md'), '/Users/me/report.md')
  assert.equal(gatewayFilePath('file:///Users/me/a%20b.md'), '/Users/me/a b.md')
  assert.equal(gatewayFilePath(''), '')
  assert.equal(gatewayFilePath(null), '')
})

test('isNotFoundError matches only HTTP 404', () => {
  const notFound: any = new Error('404: missing')

  notFound.statusCode = 404
  assert.equal(isNotFoundError(notFound), true)

  const forbidden: any = new Error('403: nope')

  forbidden.statusCode = 403
  assert.equal(isNotFoundError(forbidden), false)
  assert.equal(isNotFoundError(new Error('plain')), false)
  assert.equal(isNotFoundError(null), false)
})
