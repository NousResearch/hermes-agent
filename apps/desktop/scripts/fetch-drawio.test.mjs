import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, beforeEach, describe, test, vi } from 'vitest'

import {
  buildHtml,
  checkExisting,
  EXPECTED_SHA384_BASE64,
  fetchAndWrite,
  missingSymbols,
  sha384Base64,
} from './fetch-drawio.mjs'

const here = dirname(fileURLToPath(import.meta.url))
const committedViewerPath = join(here, '..', 'public', 'drawio', 'viewer.min.js')

function createStreamingResponse(text) {
  const encoder = new TextEncoder()
  const bytes = encoder.encode(text)
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    body: {
      getReader() {
        let done = false
        return {
          async read() {
            if (done) return { done: true }
            done = true
            return { done: false, value: bytes }
          },
        }
      },
    },
  }
}

async function withTempDir(fn) {
  const dir = await mkdtemp(join(tmpdir(), 'drawio-test-'))
  try {
    return await fn(dir)
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
}

describe('fetch-drawio integrity contract', () => {
  test('committed viewer.min.js passes hash and symbol checks', async () => {
    const result = await checkExisting({ viewerPath: committedViewerPath })
    assert.equal(result.ok, true)
    assert.equal(result.hash, EXPECTED_SHA384_BASE64)
  })

  test('checkExisting fails when viewer.min.js is missing', async () => {
    await withTempDir(async (dir) => {
      const result = await checkExisting({
        viewerPath: join(dir, 'viewer.min.js'),
      })
      assert.equal(result.ok, false)
      assert.match(result.error, /Missing/)
    })
  })

  test('checkExisting fails on a SHA-384 mismatch', async () => {
    await withTempDir(async (dir) => {
      const badPath = join(dir, 'viewer.min.js')
      await writeFile(badPath, 'not the real viewer')
      const result = await checkExisting({ viewerPath: badPath })
      assert.equal(result.ok, false)
      assert.match(result.error, /Integrity check failed/)
    })
  })

  test('checkExisting fails when required symbols are missing', async () => {
    await withTempDir(async (dir) => {
      const badPath = join(dir, 'viewer.min.js')
      // A small string that does not contain GraphViewer.
      const content = 'window.mxUtils = {};'
      await writeFile(badPath, content)
      const result = await checkExisting({ viewerPath: badPath })
      assert.equal(result.ok, false)
      assert.match(result.error, /SHA-384|Integrity check/)
      assert.deepEqual(missingSymbols(content), ['GraphViewer'])
    })
  })
})

describe('fetchAndWrite', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  test('writes a valid viewer and deterministic render.html', async () => {
    const committedJs = await readFile(committedViewerPath, 'utf8')
    globalThis.fetch.mockResolvedValue(createStreamingResponse(committedJs))

    await withTempDir(async (dir) => {
      const viewerPath = join(dir, 'viewer.min.js')
      const tmpViewerPath = join(dir, 'viewer.min.js.tmp')
      const renderPath = join(dir, 'render.html')

      const fetchResult = await fetchAndWrite({
        outDir: dir,
        viewerPath,
        tmpViewerPath,
        renderPath,
        viewerUrl: 'https://viewer.diagrams.net/js/viewer.min.js',
      })

      assert.equal(fetchResult.ok, true)
      assert.equal(fetchResult.hash, EXPECTED_SHA384_BASE64)

      const checkResult = await checkExisting({ viewerPath })
      assert.equal(checkResult.ok, true)

      const html = await readFile(renderPath, 'utf8')
      assert.match(html, /GraphViewer/)
      assert.equal(html, buildHtml(committedJs))
    })
  })

  test('rejects a viewer whose hash does not match the pinned value', async () => {
    const badContent = 'window.GraphViewer=1; window.mxUtils=1;'
    globalThis.fetch.mockResolvedValue(createStreamingResponse(badContent))
    assert.notEqual(sha384Base64(badContent), EXPECTED_SHA384_BASE64)

    await withTempDir(async (dir) => {
      const badViewerPath = join(dir, 'viewer.min.js')
      const tmpViewerPath = join(dir, 'viewer.min.js.tmp')
      await writeFile(badViewerPath, badContent)

      const fetchResult = await fetchAndWrite({
        outDir: dir,
        viewerPath: badViewerPath,
        tmpViewerPath,
        renderPath: join(dir, 'render.html'),
        viewerUrl: 'https://viewer.diagrams.net/js/viewer.min.js',
      })

      assert.equal(fetchResult.ok, false)
      assert.match(fetchResult.error, /Integrity check failed/)
      assert.equal(
        await readFile(badViewerPath, 'utf8').then(
          () => true,
          () => false
        ),
        true,
        'original bad file is untouched'
      )
    })
  })
})
