import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import path from 'node:path'

import { test } from 'vitest'

import { pathForRegistryBackendRequest } from './connection-config'
import type { PumpDeps } from './gateway-file-download'
import {
  downloadTempPath,
  filenameFromContentDisposition,
  gatewayFilePath,
  gatewayFileRequestPaths,
  isNotFoundError,
  parseDataUrlToBuffer,
  pumpStreamToFile,
  resolveGatewayFileBackend,
  writeBufferToFile
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

  constructor(writeReturns: boolean[] = [], { opens = true }: { opens?: boolean } = {}) {
    super()
    this.writeReturns = writeReturns

    // Like fs.WriteStream: 'open' fires once the exclusive create succeeded.
    // `opens: false` models a create that fails before any file exists.
    if (opens) {
      queueMicrotask(() => this.emit('open'))
    }
  }

  write(chunk: Buffer): boolean {
    this.chunks.push(chunk)

    return this.writeReturns.length ? this.writeReturns.shift()! : true
  }

  end(cb: () => void) {
    this.ended = true
    cb()
  }

  // Like fs.WriteStream: the descriptor is released asynchronously and 'close'
  // fires afterwards.
  destroy() {
    this.destroyed = true
    queueMicrotask(() => this.emit('close'))
  }
}

// Deps recorder shared by the pumpStreamToFile tests: captures every path the
// pump opens, renames, or unlinks so each test can assert the destination itself
// was never touched before the body finished.
function recordingDeps(ws: FakeWriteStream, { renameError }: { renameError?: Error } = {}) {
  const opened: string[] = []
  const renamed: Array<[string, string]> = []
  const unlinked: string[] = []

  const deps: PumpDeps = {
    createWriteStream: (p: string) => {
      opened.push(p)

      return ws as never
    },
    rename: async (from: string, to: string) => {
      if (renameError) {
        throw renameError
      }

      renamed.push([from, to])
    },
    unlink: async (p: string) => {
      unlinked.push(p)
    }
  }

  return { deps, opened, renamed, unlinked }
}

// Separator-agnostic: path.join emits backslashes on Windows, so the expectation
// is "short hidden .part name, same directory as the destination", not a
// literal POSIX string.
const TEMP_BASENAME = /^\.hermes-download-[0-9a-f]{8}\.part$/

// path.join normalizes separators (``/tmp`` -> ``\\tmp`` on Windows) while the
// literal destination strings in these tests do not, so compare normalized forms.
function assertTempPathBeside(tempPath: string, destPath: string) {
  assert.equal(
    path.normalize(path.dirname(tempPath)),
    path.normalize(path.dirname(destPath)),
    'temp file must sit beside the destination'
  )
  assert.match(path.basename(tempPath), TEMP_BASENAME)
}

test('downloadTempPath stays beside the destination with a short, random per-call name', () => {
  const a = downloadTempPath('/tmp/out.bin')
  const b = downloadTempPath('/tmp/out.bin')

  assertTempPathBeside(a, '/tmp/out.bin')
  assertTempPathBeside(b, '/tmp/out.bin')
  assert.notEqual(a, b, 'two concurrent saves into the same directory must not share a temp file')

  // The temp name must not grow with the user's filename: a destination near the
  // filesystem's name limit still gets a temp file that fits beside it.
  const longName = `/downloads/${'x'.repeat(250)}.bin`

  assert.equal(path.normalize(path.dirname(downloadTempPath(longName))), path.normalize(path.dirname(longName)))
  assert.ok(path.basename(downloadTempPath(longName)).length < 40)
})

test('pumpStreamToFile waits for the descriptor to close before renaming when the stream supports close()', async () => {
  const res = new FakeResponse()
  const order: string[] = []

  class ClosingWriteStream extends FakeWriteStream {
    close(cb: (err?: Error | null) => void) {
      order.push('close')
      // Like fs.WriteStream: end the stream, release the fd, then call back.
      this.ended = true
      setTimeout(() => cb(), 0)
    }
  }

  const ws = new ClosingWriteStream()
  const { deps, renamed } = recordingDeps(ws)

  deps.rename = async (from, to) => {
    order.push('rename')
    renamed.push([from, to])
  }

  const promise = pumpStreamToFile(res as never, '/tmp/out.bin', deps)

  res.emit('data', Buffer.from('abc'))
  res.emit('end')

  await promise

  assert.deepEqual(order, ['close', 'rename'])
  assert.equal(renamed.length, 1)
  assert.equal(renamed[0][1], '/tmp/out.bin')
})

test('pumpStreamToFile applies backpressure: pauses on a full buffer and resumes on drain', async () => {
  const res = new FakeResponse()
  const ws = new FakeWriteStream([false]) // first write signals "buffer full"
  const { deps } = recordingDeps(ws)

  const promise = pumpStreamToFile(res as never, '/tmp/out.bin', deps)

  res.emit('data', Buffer.from('big-chunk'))
  assert.equal(res.paused, true, 'source should be paused when write() returns false')
  assert.equal(res.resumed, false)

  ws.emit('drain')
  assert.equal(res.resumed, true, 'source should resume after the write stream drains')

  res.emit('end')
  await promise
})

test('pumpStreamToFile removes only the temp file and rejects on a write error', async () => {
  const res = new FakeResponse()
  const ws = new FakeWriteStream()
  const { deps, opened, renamed, unlinked } = recordingDeps(ws)

  const promise = pumpStreamToFile(res as never, '/tmp/out.bin', deps)

  res.emit('data', Buffer.from('abc'))
  ws.emit('error', new Error('ENOSPC: disk full'))

  await assert.rejects(promise, /disk full/)
  assert.deepEqual(unlinked, [opened[0]])
  assertTempPathBeside(unlinked[0], '/tmp/out.bin')
  assert.deepEqual(renamed, [], 'a failed body must never be moved onto the destination')
  assert.equal(res.destroyed, true, 'source should be torn down on write failure')
})

test('pumpStreamToFile waits for the write stream to close before unlinking the temp file', async () => {
  const res = new FakeResponse()
  const order: string[] = []

  class SlowCloseWriteStream extends FakeWriteStream {
    destroy() {
      this.destroyed = true
      order.push('destroy')
      // Release the fd later than a microtask: cleanup must still wait for it.
      setTimeout(() => {
        order.push('close')
        this.emit('close')
      }, 5)
    }
  }

  const ws = new SlowCloseWriteStream()
  const { deps, opened, unlinked } = recordingDeps(ws)

  deps.unlink = async (p: string) => {
    order.push('unlink')
    unlinked.push(p)
  }

  const promise = pumpStreamToFile(res as never, '/tmp/out.bin', deps)

  res.emit('data', Buffer.from('abc'))
  res.emit('error', new Error('socket hang up'))

  await assert.rejects(promise, /socket hang up/)
  assert.deepEqual(order, ['destroy', 'close', 'unlink'])
  assert.deepEqual(unlinked, [opened[0]])
})

test('writeBufferToFile leaves the destination untouched when the write fails after open', async () => {
  // fs.WriteStream surfaces a write failure before 'finish', never after, so
  // the fake errors from write() itself.
  class FailingWriteStream extends FakeWriteStream {
    write(chunk: Buffer): boolean {
      super.write(chunk)
      this.emit('error', new Error('ENOSPC: disk full'))

      return true
    }
  }

  const ws = new FailingWriteStream()
  const { deps, opened, renamed, unlinked } = recordingDeps(ws)

  await assert.rejects(writeBufferToFile(Buffer.from('whole body'), '/tmp/out.bin', deps), /disk full/)
  assert.ok(!opened.includes('/tmp/out.bin'))
  assert.deepEqual(unlinked, [opened[0]], 'only the owned temp file is removed')
  assert.deepEqual(renamed, [])
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

test('gatewayFilePath preserves bare paths and file:// URLs for gateway-native conversion', () => {
  assert.equal(gatewayFilePath('/Users/me/report.md'), '/Users/me/report.md')
  assert.equal(gatewayFilePath('file:///Users/me/a%20b.md'), 'file:///Users/me/a%20b.md')
  assert.equal(gatewayFilePath(''), '')
  assert.equal(gatewayFilePath(null), '')
})

test('gatewayFileRequestPaths keeps streaming and fallback requests on the same registered backend', () => {
  const paths = gatewayFileRequestPaths('/srv/output/image one.png', requestPath =>
    pathForRegistryBackendRequest(requestPath, 'research', { sharedRemote: true })
  )

  assert.deepEqual(paths, {
    dataUrl: '/api/fs/read-data-url?path=%2Fsrv%2Foutput%2Fimage+one.png&profile=research',
    download: '/api/fs/download?path=%2Fsrv%2Foutput%2Fimage+one.png&profile=research'
  })
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

test('resolveGatewayFileBackend pins registered files to their owning connection', async () => {
  const calls: string[] = []

  const route = await resolveGatewayFileBackend(
    { connectionId: '  work-ssh  ', profile: ' default ' },
    {
      ensureLegacy: async profile => {
        calls.push(`legacy:${profile}`)

        return { baseUrl: 'http://local.invalid' }
      },
      ensureRegistry: async (connectionId, profile) => {
        calls.push(`registry:${connectionId}:${profile}`)

        return { baseUrl: 'http://ssh.invalid' }
      }
    }
  )

  assert.deepEqual(calls, ['registry:work-ssh:default'])
  assert.deepEqual(route, {
    connection: { baseUrl: 'http://ssh.invalid' },
    connectionId: 'work-ssh',
    profile: 'default'
  })
})

test('resolveGatewayFileBackend preserves the legacy route when no connection owns the file', async () => {
  const calls: string[] = []

  const route = await resolveGatewayFileBackend(
    { profile: 'coder' },
    {
      ensureLegacy: async profile => {
        calls.push(`legacy:${profile}`)

        return { baseUrl: 'http://local.invalid' }
      },
      ensureRegistry: async connectionId => {
        calls.push(`registry:${connectionId}`)

        return { baseUrl: 'http://remote.invalid' }
      }
    }
  )

  assert.deepEqual(calls, ['legacy:coder'])
  assert.equal(route.connectionId, null)
  assert.equal(route.profile, 'coder')
  assert.deepEqual(route.connection, { baseUrl: 'http://local.invalid' })
})
