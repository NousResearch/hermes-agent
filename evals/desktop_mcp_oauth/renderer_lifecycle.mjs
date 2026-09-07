import { createRequire } from 'node:module'
import fs from 'node:fs/promises'
import assert from 'node:assert/strict'
import http from 'node:http'
import path from 'node:path'
const repo = process.argv[2],
  out = process.argv[3],
  scenario = process.argv[4] || 'approved'
const require = createRequire(path.join(repo, 'package.json'))
const { build } = require('esbuild')
const { chromium } = require('playwright')
const requests = []
const rpcCalls = []
const nativeCalls = []
const handlers = new Map()
const nativeBuild = await build({
  entryPoints: [repo + '/apps/desktop/electron/mcp-oauth-callback-ipc.ts'],
  bundle: true,
  write: false,
  platform: 'node',
  format: 'cjs',
  external: ['electron']
})
const nativeModule = { exports: {} }
new Function('require', 'module', 'exports', nativeBuild.outputFiles[0].text)(
  id => (id === 'electron' ? { ipcMain: { handle: (name, fn) => handlers.set(name, fn) } } : require(id)),
  nativeModule,
  nativeModule.exports
)
nativeModule.exports.registerMcpOauthCallbackIpc()
const { WebSocketServer } = require('ws')
const wss = new WebSocketServer({ port: 0, host: '127.0.0.1' })
await new Promise(r => wss.once('listening', r))
let approved = false
let redirectUri
wss.on('connection', ws =>
  ws.on('message', raw => {
    const m = JSON.parse(raw)
    rpcCalls.push(m)
    let result = { ok: true }
    if (m.method === 'mcp.servers.oauth.start') {
      redirectUri = m.params.client_redirect_uri
      result = {
        ok: true,
        session_id: 'fixture-session',
        auth_url:
          'http://127.0.0.1/authorize?state=fixture-state&redirect_uri=' +
          encodeURIComponent(scenario === 'legacy' ? 'http://remote.invalid/callback' : redirectUri)
      }
    }
    if (m.method === 'mcp.servers.oauth.callback') {
      approved = m.params.state === 'fixture-state' && m.params.code === 'fixture-code'
      result = { ok: approved }
    }
    if (m.method === 'mcp.servers.oauth.poll')
      result = { ok: true, status: approved ? 'approved' : 'pending', tools: [] }
    ws.send(JSON.stringify({ jsonrpc: '2.0', id: m.id, result }))
  })
)
const entry = `import React from 'react'; import {createRoot} from 'react-dom/client';
import {QueryClient, QueryClientProvider} from '@tanstack/react-query';
import {McpTab} from '${repo}/apps/desktop/src/app/skills/mcp-tab.tsx';
import {setApiRequestProfile,setApiRequestConnection} from '${repo}/apps/desktop/src/api/client.ts';
setApiRequestProfile('profile-b');setApiRequestConnection('fixture-remote');
const queryClient=new QueryClient({defaultOptions:{queries:{retry:false}}});function Probe(){const [profile,setProfile]=React.useState('profile-b');const [shown,setShown]=React.useState(true);window.cancelFlow=()=>setProfile('profile-c');window.unmountFlow=()=>setShown(false);return <QueryClientProvider client={queryClient}>{shown && <McpTab gateway={null} profile={{connectionId:'fixture-remote',profile}} />}</QueryClientProvider>};createRoot(document.getElementById('root')).render(<Probe/>);`
const bundle = await build({
  stdin: { contents: entry, resolveDir: repo, loader: 'tsx' },
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'browser',
  jsx: 'automatic',
  alias: { '@': repo + '/apps/desktop/src', '@hermes/shared': repo + '/apps/shared/src' },
  define: {
    'process.env.NODE_ENV': '"production"',
    'import.meta.env': '{}',
    'import.meta.env.DEV': 'false',
    'import.meta.env.PROD': 'true'
  },
  loader: { '.woff2': 'dataurl', '.woff': 'dataurl', '.svg': 'dataurl' },
  logLevel: 'warning'
})
const server = http.createServer(async (req, res) => {
  res.setHeader('Content-Type', req.url === '/app.js' ? 'text/javascript' : 'text/html')
  res.end(
    req.url === '/app.js'
      ? bundle.outputFiles[0].text
      : '<div id="root"></div><script type="module" src="/app.js"></script>'
  )
})
await new Promise(r => server.listen(0, '127.0.0.1', r))
const browser = await chromium.launch({
  headless: true,
  args: ['--no-sandbox'],
  ...(process.env.CHROMIUM_EXECUTABLE ? { executablePath: process.env.CHROMIUM_EXECUTABLE } : {})
})
try {
  const page = await browser.newPage()
  const errors = []
  page.on('pageerror', e => {
    errors.push(e.message)
    console.error(e.stack)
  })
  page.on('console', m => console.log(m.type(), m.text()))
  await page.exposeFunction('recordRequest', r => {
    requests.push(r)
    console.log('request', JSON.stringify(r))
    if (r.path === '/api/config')
      return { mcp_servers: { reports: { url: 'https://fixture.invalid/mcp', auth: 'oauth' } } }
    if (r.path.endsWith('/test')) return { ok: false, tools: [], auth_required: true, error: 'OAuth required' }
    if (r.path.includes('/logs')) return { lines: [] }
    if (r.path.includes('/catalog')) return { entries: [] }
    if (r.path.endsWith('/auth'))
      return { flow_id: 'f', status: 'error', error: 'fixture remote HTTP redirect rejected' }
    return { ok: true, providers: [] }
  })
  await page.exposeFunction('nativeOAuth', async (action, id) => {
    nativeCalls.push({ action, id })
    return handlers.get('hermes:mcp-oauth:' + action)({}, id)
  })
  await page.exposeFunction('fixtureConnection', async scope => {
    nativeCalls.push({ action: 'connection', scope })
    return { wsUrl: 'ws://127.0.0.1:' + wss.address().port, connectionId: scope.connectionId, authMode: 'token' }
  })
  await page.exposeFunction('openAuthorization', async url => {
    nativeCalls.push({ action: 'open', url })
    if (scenario === 'cancel' || scenario === 'unmount') {
      await page.evaluate(s => (s === 'unmount' ? window.unmountFlow() : window.cancelFlow()), scenario)
      return
    }
    const u = new URL(url)
    await fetch(u.searchParams.get('redirect_uri') + '?code=fixture-code&state=' + u.searchParams.get('state'))
  })
  await page.addInitScript(() => {
    window.hermesDesktop = {
      api: r => window.recordRequest(r),
      getConnectionFor: r => window.fixtureConnection(r),
      mcpOauth: {
        listen: () => {
          window.nativeListen = true
          return window.nativeOAuth('listen')
        },
        wait: id => window.nativeOAuth('wait', id),
        cancel: id => window.nativeOAuth('cancel', id)
      },
      openExternal: url => window.openAuthorization(url)
    }
  })
  await page.goto('http://127.0.0.1:' + server.address().port)
  await page.waitForTimeout(1500)
  console.log('BODY', await page.locator('body').innerText())
  await page.getByText('Reports', { exact: true }).first().click()
  await page.waitForTimeout(300)
  console.log('SELECTED', await page.locator('body').innerText())
  await page
    .getByRole('button', { name: /Authenticate|Sign in/ })
    .first()
    .click()
  await page.waitForTimeout(2500)
  const hasScope = await page.evaluate(() => !!window.nativeListen)
  const result = {
    repo,
    scenario,
    requests,
    rpcCalls,
    nativeCalls,
    approved,
    hasScope,
    errors,
    text: await page.locator('body').innerText(),
    fidelity: 'production McpTab + all real renderer imports in headless Chromium; fixture native API transport'
  }
  await fs.writeFile(out, JSON.stringify(result, null, 2))
  console.log(JSON.stringify(result))
  if (scenario === 'cancel' || scenario === 'unmount') {
    assert(
      rpcCalls.some(m => m.method === 'mcp.servers.oauth.cancel'),
      'abandoned component must cancel backend'
    )
    assert(
      nativeCalls.some(m => m.action === 'cancel'),
      'abandoned component must close listener'
    )
    assert(
      nativeCalls.filter(m => m.action === 'connection').every(m => m.scope.profile === 'profile-b'),
      'cleanup must retain original owner'
    )
    assert(!rpcCalls.some(m => m.method === 'mcp.servers.oauth.callback'), 'no callback after abandonment')
  }
  if (scenario === 'approved') {
    assert(approved && hasScope, 'Skills must complete OAuth through its native listener')
    assert(!requests.some(r => r.path.endsWith('/auth')), 'native flow must not use REST auth')
  }
  assert.deepEqual(errors, [])
} finally {
  await browser.close()
  server.close()
  for (const c of wss.clients) c.terminate()
  wss.close()
}
