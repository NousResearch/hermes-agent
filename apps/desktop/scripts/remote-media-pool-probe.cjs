// Run from apps/desktop: npm exec -- electron scripts/remote-media-pool-probe.cjs
// Linux CI: env -u ELECTRON_RUN_AS_NODE xvfb-run -a npm exec -- electron --no-sandbox --ozone-platform=x11 scripts/remote-media-pool-probe.cjs
const { app, net, session, protocol, BrowserWindow } = require('electron')
protocol.registerSchemesAsPrivileged([
  { scheme: 'hermes-media', privileges: { standard: true, secure: true, stream: true, supportFetchAPI: true } }
])
const { createServer } = require('node:http')
const { mkdtempSync, rmSync, writeFileSync } = require('node:fs')
const { tmpdir } = require('node:os')
const { join } = require('node:path')
const { buildSync } = require('esbuild')
const assert = require('node:assert/strict')
const home = mkdtempSync(join(tmpdir(), 'hermes-media-pool-'))
app.setPath('userData', home)
const watchdog = setTimeout(() => {
  console.error('FAIL: media probe exceeded 45 seconds')
  app.exit(1)
}, 45_000)
buildSync({
  entryPoints: [join(__dirname, '../electron/media-protocol.ts')],
  bundle: true,
  platform: 'node',
  format: 'cjs',
  outfile: join(home, 'transport.cjs')
})
const { createMediaProtocolHandler } = require(join(home, 'transport.cjs'))
buildSync({
  entryPoints: [join(__dirname, '../electron/remote-media-fetch.ts')],
  bundle: true,
  platform: 'node',
  format: 'cjs',
  outfile: join(home, 'fetch.cjs')
})
const { fetchRemoteMedia } = require(join(home, 'fetch.cjs'))
// --baseline replays the old direct-fetch adapter to prove the cancellation
// assertion fails in Electron, independently of the mocked unit tests.
const fetchMedia = process.argv.includes('--baseline')
  ? (fetcher, headers) => fetcher(headers, undefined)
  : fetchRemoteMedia
app.whenReady().then(async () => {
  const size = 16 * 1024 * 1024
  const payload = Buffer.alloc(size)
  for (let i = 0; i < size; i++) payload[i] = i % 251
  const wav = Buffer.alloc(44 + 48000 * 2 * 6)
  wav.write('RIFF')
  wav.writeUInt32LE(wav.length - 8, 4)
  wav.write('WAVEfmt ', 8)
  wav.writeUInt32LE(16, 16)
  wav.writeUInt16LE(1, 20)
  wav.writeUInt16LE(1, 22)
  wav.writeUInt32LE(48000, 24)
  wav.writeUInt32LE(96000, 28)
  wav.writeUInt16LE(2, 32)
  wav.writeUInt16LE(16, 34)
  wav.write('data', 36)
  wav.writeUInt32LE(wav.length - 44, 40)
  const largeWav = Buffer.alloc(16 * 1024 * 1024)
  wav.copy(largeWav, 0, 0, 44)
  largeWav.writeUInt32LE(largeWav.length - 8, 4)
  largeWav.writeUInt32LE(largeWav.length - 44, 40)
  let mediaRequests = 0
  const server = createServer((req, res) => {
    if (req.url !== '/health') mediaRequests++
    if (req.url === '/health') {
      res.end('{}')
      return
    }
    const match = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || '')
    if ((req.url === '/media' || req.url === '/audio' || req.url === '/large-audio') && match) {
      const data = req.url === '/audio' ? wav : req.url === '/large-audio' ? largeWav : payload
      const start = Number(match[1]),
        end = match[2] ? Math.min(Number(match[2]), data.length - 1) : data.length - 1
      res.writeHead(206, {
        'Content-Range': `bytes ${start}-${end}/${data.length}`,
        'Content-Length': end - start + 1,
        'Content-Type': req.url.includes('audio') ? 'audio/wav' : 'application/octet-stream',
        'Accept-Ranges': 'bytes'
      })
      res.end(data.subarray(start, end + 1))
    } else {
      const data = req.url === '/audio' ? wav : req.url === '/large-audio' ? largeWav : payload
      res.writeHead(200, {
        'Content-Length': data.length,
        'Content-Type': req.url.includes('audio') ? 'audio/wav' : 'application/octet-stream',
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'no-store'
      })
      res.end(data)
    }
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const base = `http://127.0.0.1:${server.address().port}`
  const sess = session.fromPartition('media-pool-probe')
  const health = () =>
    new Promise(resolve => {
      const req = net.request({ url: base + '/health', session: sess, useSessionCookies: true })
      const timer = setTimeout(() => {
        req.abort()
        resolve('TIMEOUT')
      }, 2000)
      req.on('error', () => {})
      req.on('response', res => {
        res.on('data', () => {})
        res.on('end', () => {
          clearTimeout(timer)
          resolve('OK')
        })
      })
      req.end()
    })
  let code = 1
  try {
    assert.equal(await health(), 'OK')
    const controllers = Array.from({ length: 6 }, () => new AbortController())
    const old = await Promise.all(controllers.map(c => sess.fetch(base + '/media', { signal: c.signal })))
    assert.equal(await health(), 'TIMEOUT')
    console.log('BASELINE: six paused media responses starve health (reproduced)')
    controllers.forEach(c => c.abort())
    await Promise.all(old.map(r => r.body.cancel()))
    assert.equal(await health(), 'OK')
    const handler = createMediaProtocolHandler({
      ensureRemoteBearer: async () => null,
      resolveRemoteConnection: async () => ({ mode: 'remote', baseUrl: base, token: 'probe' }),
      resolveLocalFile: async p => p,
      fetchLocal: async () => {
        throw new Error('unexpected local fetch')
      },
      fetchRemote: (url, headers, method, signal) =>
        fetchMedia(
          (h, s) =>
            sess.fetch(
              base +
                (new URL(url).searchParams.get('path').startsWith('audio-')
                  ? '/large-audio'
                  : new URL(url).searchParams.get('path') === 'seek.wav'
                    ? '/audio'
                    : new URL(url).searchParams.get('path') === 'ignore.wav'
                      ? '/ignores-range'
                      : '/media'),
              { headers: h, method, signal: s }
            ),
          headers,
          signal
        ),
      fetchRemoteWithCookies: async () => {
        throw new Error('unexpected cookie fetch')
      }
    })
    const held = await Promise.all(
      Array.from({ length: 6 }, () =>
        handler(new Request('hermes-media://remote/ignore.wav', { headers: { range: 'bytes=0-1023' } }))
      )
    )
    assert.ok(held.every(response => response.status === 200))
    assert.equal(await health(), 'TIMEOUT')
    await Promise.all(held.map(r => r.body.cancel()))
    assert.equal(await health(), 'OK', 'cancelled protocol responses must release HTTP sockets')
    console.log('CANCEL: protocol response cancellation releases sockets even when Range is ignored')
    const complete = await handler(new Request('hermes-media://remote/clip.wav'))
    assert.equal(complete.status, 200)
    assert.deepEqual(Buffer.from(await complete.arrayBuffer()), payload)
    for (const range of ['bytes=0-262143', 'bytes=1048576-2097151', `bytes=${size - 1024}-`]) {
      const response = await handler(new Request('hermes-media://remote/clip.wav', { headers: { range } }))
      const [, start, end] = /^bytes=(\d+)-(\d*)$/.exec(range)
      assert.equal(response.status, 206)
      assert.equal(response.headers.get('content-range'), `bytes ${start}-${end || size - 1}/${size}`)
      assert.deepEqual(
        Buffer.from(await response.arrayBuffer()),
        payload.subarray(Number(start), end ? Number(end) + 1 : size)
      )
    }
    console.log('PAYLOAD: full 16 MiB and exact first/middle/suffix ranges preserved')
    const aborted = new AbortController()
    const active = await Promise.all(
      Array.from({ length: 6 }, () =>
        handler(new Request('hermes-media://remote/clip.wav', { signal: aborted.signal }))
      )
    )
    assert.equal(await health(), 'TIMEOUT')
    aborted.abort()
    await Promise.all(active.map(r => r.arrayBuffer().catch(() => {})))
    assert.equal(await health(), 'OK')
    console.log('SIGNAL: request abort releases six active upstream responses')

    protocol.handle('hermes-media', handler)
    buildSync({
      entryPoints: [join(__dirname, 'remote-media-pool-renderer.tsx')],
      bundle: true,
      platform: 'browser',
      format: 'iife',
      outfile: join(home, 'renderer.js'),
      define: { 'process.env.NODE_ENV': '"production"' }
    })
    writeFileSync(
      join(home, 'index.html'),
      '<html><body><div id="root"></div><script src="renderer.js"></script></body></html>'
    )
    const window = new BrowserWindow({ show: false, webPreferences: { autoplayPolicy: 'no-user-gesture-required' } })
    await window.loadFile(join(home, 'index.html'))
    for (let round = 0; round < 2; round++) {
      const before = mediaRequests
      await window.webContents.executeJavaScript('mountPlayers()')
      assert.equal(await health(), 'OK')
      assert.equal(mediaRequests, before, 'preload=none must not request media')
      console.log(`PLAYERS: starting round ${round}`)
      await window.webContents.executeJavaScript('startPlayers()')
      console.log('PLAYERS: replacing sources')
      await window.webContents.executeJavaScript('mountPlayers("-replacement")')
      await window.webContents.executeJavaScript('startPlayers()')
      await window.webContents.executeJavaScript('removePlayers()')
      assert.equal(await health(), 'OK', 'unmounted started players must release gateway connections')
    }
    console.log(
      'PLAYERS: two rounds of eight mounted elements, four started/paused; replace sources, replay, remove; connections released'
    )
    const playback = await window.webContents.executeJavaScript(`new Promise((resolve, reject) => {
      const audio = document.createElement('audio'); audio.muted = true;
      const timer = setTimeout(() => reject(new Error('playback timeout')), 10000);
      audio.onerror = () => reject(new Error('audio error ' + audio.error?.code));
      audio.onloadedmetadata = () => { audio.currentTime = 5.5; audio.play().catch(reject); };
      audio.onended = () => { clearTimeout(timer); resolve({ duration: audio.duration, ended: audio.ended }); };
      audio.src = 'hermes-media://remote/seek.wav'; document.body.appendChild(audio);
    })`)
    assert.deepEqual(playback, { duration: 6, ended: true })
    window.destroy()
    console.log('PLAYBACK: WAV decoding, seek to 5.5s, playback to end')
    code = 0
  } catch (error) {
    console.error(error)
  }
  server.closeAllConnections()
  server.close()
  app.exit(code)
})
app.on('quit', () => {
  clearTimeout(watchdog)
  try {
    rmSync(home, { recursive: true, force: true })
  } catch {}
})
