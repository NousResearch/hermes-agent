// Regression for #122478: packaging binary downloads died on the first uncoded
// transport error (TypeError: terminated / fixed-budget TimeoutError), restarted
// from byte 0 on every attempt, and re-downloaded an Electron zip the npm
// `electron` package had already fetched.
import assert from 'node:assert/strict'
import http from 'node:http'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'
import {
  ResumableDownloader, electronCacheKey, findInSharedElectronCache, isRetryableDownloadError,
  patchResolverSource, resumableFetchToFile,
} from './patch-electron-get-downloads.mjs'

test('uncoded undici transport errors are retryable; semantic errors are not', () => {
  // The two fatal shapes from the issue log, neither carries .code.
  assert.equal(isRetryableDownloadError(new TypeError('terminated')), true)
  assert.equal(isRetryableDownloadError(Object.assign(new Error('The operation was aborted due to timeout'), { name: 'TimeoutError' })), true)
  assert.equal(isRetryableDownloadError(Object.assign(new Error('fetch failed'), { name: 'TypeError' })), true)
  assert.equal(isRetryableDownloadError(Object.assign(new Error('getaddrinfo EAI_AGAIN something'), { code: 'EAI_AGAIN' })), true)
  // Not transport-shaped: must stay fatal so real failures surface.
  assert.equal(isRetryableDownloadError(new Error('Response code 404 (Not Found)')), false)
  assert.equal(isRetryableDownloadError(new Error('checksum mismatch')), false)
  assert.equal(isRetryableDownloadError(null), false)
})

test('cache key matches the shared cache layout the npm electron package writes (cross-version contract)', () => {
  // Worked example from #122478: %LOCALAPPDATA%\electron\Cache\3978a3c4…\electron-v40.10.2-win32-x64.zip
  const url = 'https://github.com/electron/electron/releases/download/v40.10.2/electron-v40.10.2-win32-x64.zip'
  assert.equal(electronCacheKey(url), '3978a3c4a2965533dc07f99112894e7e7f80c9ea0f13e2a48cd5a29593568fb2')
})

test('shared-cache lookup finds the npm-fetched zip and never resolves the checksum file', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'shared-electron-'))
  try {
    const url = 'https://github.com/electron/electron/releases/download/v40.10.2/electron-v40.10.2-win32-x64.zip'
    const cached = path.join(root, electronCacheKey(url), 'electron-v40.10.2-win32-x64.zip')
    fs.mkdirSync(path.dirname(cached), { recursive: true })
    fs.writeFileSync(cached, 'zip bytes')
    assert.equal(findInSharedElectronCache(url, root), cached)
    assert.equal(findInSharedElectronCache('https://github.com/electron/electron/releases/download/v40.10.2/SHASUMS256.txt', root), null)
    assert.equal(findInSharedElectronCache('https://example.com/missing.zip', root), null)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('downloader reuses the shared-cache artifact instead of the network', async () => {
  const shared = fs.mkdtempSync(path.join(os.tmpdir(), 'shared-electron-'))
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'eg-target-'))
  try {
    const url = 'https://github.com/electron/electron/releases/download/v40.10.2/electron-v40.10.2-win32-x64.zip'
    const cached = path.join(shared, electronCacheKey(url), 'electron-v40.10.2-win32-x64.zip')
    fs.mkdirSync(path.dirname(cached), { recursive: true })
    fs.writeFileSync(cached, 'already on disk')
    let requests = 0
    const server = http.createServer(() => { requests++ })
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    try {
      const downloader = new ResumableDownloader({ sharedRoot: shared })
      await downloader.download(url, path.join(work, 'artifact.zip'), {})
      assert.equal(fs.readFileSync(path.join(work, 'artifact.zip'), 'utf8'), 'already on disk')
      assert.equal(requests, 0) // never touched the network
    } finally {
      server.close()
    }
  } finally {
    fs.rmSync(shared, { recursive: true, force: true })
    fs.rmSync(work, { recursive: true, force: true })
  }
})

test('a mid-transfer connection reset is retried and completes (the #122478 fatality)', async () => {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'eg-retry-'))
  try {
    const body = 'x'.repeat(64 * 1024)
    let requests = 0
    const server = http.createServer((request, response) => {
      requests++
      if (requests === 1) {
        response.writeHead(200, { 'content-length': String(body.length) })
        response.write(body.slice(0, 1024))
        response.destroy() // connection reset mid-transfer, no .code on the fetch error
        return
      }
      const range = request.headers.range
      if (range) {
        const start = Number(range.match(/bytes=(\d+)-/)?.[1] ?? 0)
        response.writeHead(206, { 'content-length': String(body.length - start), 'content-range': `bytes ${start}-${body.length - 1}/${body.length}` })
        response.end(start > 0 ? body.slice(start) : body)
      } else {
        response.writeHead(200, { 'content-length': String(body.length) })
        response.end(body)
      }
    })
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    try {
      const target = path.join(work, 'toolset.zip')
      const reports = []
      await new ResumableDownloader({ sharedRoot: path.join(work, 'no-shared-cache') })
        .download(`http://127.0.0.1:${server.address().port}/toolset.zip`, target,
          { getProgressCallback: info => reports.push(info) })
      assert.equal(requests, 2, 'first attempt reset, second attempt succeeded')
      assert.equal(fs.readFileSync(target, 'utf8'), body)
      assert.equal(reports.at(-1).percent, 1)
    } finally {
      server.close()
    }
  } finally {
    fs.rmSync(work, { recursive: true, force: true })
  }
})

test('resume appends via HTTP Range and reports progress from the existing offset', async () => {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'eg-resume-'))
  try {
    const body = '0123456789abcdef'
    const server = http.createServer((request, response) => {
      const start = Number(request.headers.range?.match(/bytes=(\d+)-/)?.[1] ?? -1)
      assert.ok(start >= 0, 'resume attempt must send a Range header')
      response.writeHead(206, { 'content-length': String(body.length - start), 'content-range': `bytes ${start}-${body.length - 1}/${body.length}` })
      response.end(body.slice(start))
    })
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    try {
      const target = path.join(work, 'partial.zip')
      fs.writeFileSync(target, body.slice(0, 8)) // half a file left by a dead attempt
      const reports = []
      await resumableFetchToFile(`http://127.0.0.1:${server.address().port}/electron.zip`, target,
        { getProgressCallback: info => reports.push(info) }, 8)
      assert.equal(fs.readFileSync(target, 'utf8'), body)
      assert.equal(reports[0].transferred, 8, 'progress starts from the resumed offset')
    } finally {
      server.close()
    }
  } finally {
    fs.rmSync(work, { recursive: true, force: true })
  }
})

test('a server that ignores Range triggers a clean restart, not a corrupt append', async () => {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'eg-refused-'))
  try {
    const body = '0123456789abcdef'
    let requests = 0
    const server = http.createServer((request, response) => {
      requests++
      if (request.headers.range) {
        // Range ignored: full 200 body would be wrong to append to the partial file.
        response.writeHead(200, { 'content-length': String(body.length) })
        response.end(body)
        return
      }
      response.writeHead(200, { 'content-length': String(body.length) })
      response.end(body)
    })
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    try {
      const target = path.join(work, 'partial.zip')
      fs.writeFileSync(target, '01234567')
      await new ResumableDownloader({ sharedRoot: path.join(work, 'no-shared-cache') })
        .download(`http://127.0.0.1:${server.address().port}/electron.zip`, target, {})
      assert.equal(fs.readFileSync(target, 'utf8'), body, 'restart from zero after refused resume')
      assert.equal(requests, 2)
    } finally {
      server.close()
    }
  } finally {
    fs.rmSync(work, { recursive: true, force: true })
  }
})

test('a stalled transfer aborts on the stall watchdog, not a fixed minute budget', async () => {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'eg-stall-'))
  try {
    const server = http.createServer((request, response) => {
      response.writeHead(200, { 'content-length': '1000' })
      response.write('partial')
      // then never another byte
    })
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    try {
      const target = path.join(work, 'stalled.zip')
      await assert.rejects(
        resumableFetchToFile(`http://127.0.0.1:${server.address().port}/stalled.zip`, target, {}, 0, 150),
        /stalled/)
    } finally {
      server.close()
    }
  } finally {
    fs.rmSync(work, { recursive: true, force: true })
  }
})

test('resolver transform targets the published @electron/get 5.1.0 shape and fails closed on drift', () => {
  const published = `import { FetchDownloader } from './FetchDownloader.js';\nexport async function getDownloaderForSystem() {\n    return new FetchDownloader();\n}\n//# sourceMappingURL=downloader-resolver.js.map`
  const patched = patchResolverSource(published, 'file:///hermes/patch.mjs')
  assert.match(patched, /import\("file:\/\/\/hermes\/patch\.mjs"\)/)
  assert.match(patched, /return new ResumableDownloader\(\)/)
  assert.doesNotMatch(patched, /return new FetchDownloader\(\)/)
  assert.throws(() => patchResolverSource('export async function getDownloaderForSystem() { /* refactored */ }', 'file:///x.mjs'), /shape changed/)
})
