import { readFileSync } from 'node:fs'
import http from 'node:http'
import https from 'node:https'
import type { AddressInfo } from 'node:net'

import ts from 'typescript'
import { afterEach, expect, it } from 'vitest'

import { destroyKeepaliveAgents, htmlResponseError, httpStatusError, jsonAgentFor, withRetry } from './api-transport'

// Exercise the actual main-process functions without booting Electron.
const source = readFileSync(new URL('./main.ts', import.meta.url), 'utf8')
const ast = ts.createSourceFile('main.ts', source, ts.ScriptTarget.Latest, true)

function load(name: string, client = http) {
  const node = ast.statements.find(s => ts.isFunctionDeclaration(s) && s.name?.text === name)!
  const code = ts.transpileModule(node.getText(ast), { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText

  return new Function(
    'http',
    'https',
    'jsonAgentFor',
    'withRetry',
    'resolveTimeoutMs',
    'DEFAULT_FETCH_TIMEOUT_MS',
    'headersForRemoteRequest',
    'httpStatusError',
    'htmlResponseError',
    `${code}; return ${name}`
  )(
    client,
    https,
    jsonAgentFor,
    withRetry,
    (value: number, fallback: number) => value ?? fallback,
    1000,
    () => ({ 'X-Proxy': 'fixture' }),
    httpStatusError,
    htmlResponseError
  )
}

const fetchJson = load('fetchJson')
const fetchPublicJson = load('fetchPublicJson')
let server: http.Server
const requests: http.ClientRequest[] = []
afterEach(async () => {
  requests.splice(0).forEach(r => r.destroy())
  destroyKeepaliveAgents()
  server?.closeAllConnections()

  if (server?.listening) {await new Promise<void>(resolve => server.close(() => resolve()))}
})

async function listen(handler: http.RequestListener) {
  server = http.createServer(handler)
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`
}

it('normal JSON reuses sockets while health bypasses a stalled reused socket', async () => {
  const seen = new Set<unknown>()

  const base = await listen((req, res) => {
    if (req.url === '/api/health' && seen.has(req.socket)) {return}
    seen.add(req.socket)
    res.end(JSON.stringify({ sockets: seen.size }))
  })

  await expect(fetchJson(`${base}/warm`, 'fixture')).resolves.toEqual({ sockets: 1 })
  await expect(fetchJson(`${base}/warm`, 'fixture')).resolves.toEqual({ sockets: 1 })
  await expect(fetchJson(`${base}/api/health`, 'fixture', { freshConnection: true, timeoutMs: 100 })).resolves.toEqual({
    sockets: 2
  })
})

it('bounds a pending DNS lookup before a socket connects', async () => {
  const fetch = load('fetchJson', {
    ...http,
    request: (url: URL, options: object, callback: any) =>
      http.request(url, { ...options, lookup: (() => {}) as any }, callback)
  } as typeof http)

  const pending = fetch('http://pending.invalid/api/health', 'fixture', { freshConnection: true, timeoutMs: 60 })

  const result = await Promise.race([
    pending.then(
      () => 'success',
      () => 'deadline'
    ),
    new Promise(resolve => setTimeout(() => resolve('unbounded'), 250))
  ])

  expect(result).toBe('deadline')
})

it.each(['token', 'public'])('%s probe deadline bounds a trickling response', async mode => {
  let drip: ReturnType<typeof setInterval>

  const base = await listen((_req, res) => {
    res.write('{')
    drip = setInterval(() => res.write(' '), 10)
    res.on('close', () => clearInterval(drip))
  })

  const options = { freshConnection: true, timeoutMs: 60 }

  const pending =
    mode === 'token'
      ? fetchJson(`${base}/api/health`, 'fixture', options)
      : fetchPublicJson(`${base}/api/health`, options)

  const result = await Promise.race([
    pending.then(
      () => 'success',
      () => 'deadline'
    ),
    new Promise(resolve => setTimeout(() => resolve('unbounded'), 250))
  ])

  expect(result).toBe('deadline')
})

it('a failed fresh probe does not multiply its budget through transport retries', async () => {
  let hits = 0

  const base = await listen(req => {
    hits++
    req.socket.destroy()
  })

  const started = Date.now()
  await expect(fetchJson(`${base}/api/health`, 'fixture', { freshConnection: true, timeoutMs: 60 })).rejects.toThrow()
  expect(hits).toBe(1)
  expect(Date.now() - started).toBeLessThan(250)
})

it.each(['token', 'public'])('%s health bypasses a saturated JSON socket pool', async mode => {
  let busy = 0
  let allBusy!: () => void

  const saturated = new Promise<void>(resolve => {
    allBusy = resolve
  })

  const base = await listen((req, res) => {
    if (req.url === '/busy') {
      if (++busy === 50) {allBusy()}

      return
    }

    res.end(
      JSON.stringify({
        ok: true,
        token: req.headers['x-hermes-session-token'],
        proxy: req.headers['x-proxy'],
        auth: req.headers.authorization
      })
    )
  })

  for (let i = 0; i < 50; i++) {
    const req = http.get(`${base}/busy`, { agent: jsonAgentFor('http:') })
    req.on('error', () => {})
    requests.push(req)
  }

  await saturated
  const options = { freshConnection: true, timeoutMs: 100, bearer: 'fixture-bearer' }

  const pending =
    mode === 'token'
      ? fetchJson(`${base}/api/health`, 'fixture-token', options)
      : fetchPublicJson(`${base}/api/health`, options)

  const result = await Promise.race([
    pending.catch((e: Error) => e.message),
    new Promise(resolve => setTimeout(() => resolve('blocked by JSON pool'), 300))
  ])

  expect(result).toEqual(
    mode === 'token'
      ? { ok: true, token: 'fixture-token', proxy: 'fixture', auth: 'Bearer fixture-bearer' }
      : { ok: true, proxy: 'fixture' }
  )
})
