// Regression based on andrexibiza’s PR #93508 review 5405545132.
// Real independent Chromium pages, ThemeProvider and native storage events against one shared config stub
// that numbers its revisions like hermes serve (GET ?with_revision, PUT { ok, revision }). Each case picks in
// one or two pages, holds both saves and answers them in a scripted order and outcome; every page and the
// shared cache must end on the config. A save lands when it is answered successfully.
// Run from the repository root: CHROMIUM_PATH=/path/to/chrome node apps/desktop/scripts/smoke-appearance-pages.mjs
import fs from 'node:fs/promises'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import {createRequire} from 'node:module'
import {pathToFileURL} from 'node:url'

const repo = path.resolve(process.argv[2] || '.')
const require = createRequire(path.join(repo, 'apps/desktop/package.json'))
const {createServer} = await import(pathToFileURL(require.resolve('vite')).href)
const {chromium} = require('@playwright/test')
const root = await fs.mkdtemp(path.join(os.tmpdir(), 'hermes-appearance-'))
await fs.writeFile(path.join(root, 'index.html'), '<div id="root"></div><script type="module" src="/entry.tsx"></script>')
await fs.writeFile(path.join(root, 'entry.tsx'), `
import React from 'react'
import {createRoot} from 'react-dom/client'
document.documentElement.dataset.hermesDesktopHost = 'browser'
const params = new URLSearchParams(location.search)
const profile = params.get('profile') || 'default', client = params.get('client')
Object.assign(window, {inflight: 0, hermesDesktop: {async api(request) {
  window.inflight++
  try {
    const response = await fetch('/__config', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({client, request})})
    const result = await response.json()
    if (result.failed) throw Error('save failed')
    return result
  } finally { window.inflight-- }
}}})
const {setApiRequestConnection, setApiRequestLocalMode, setApiRequestProfile} = await import('@/api/client')
const {$activeGatewayProfile} = await import('@/store/profile')
const {ThemeProvider, useTheme, skinPref, modePref} = await import('@/themes/context')
setApiRequestLocalMode(false); setApiRequestConnection('A'); setApiRequestProfile(profile)
$activeGatewayProfile.set(profile)
function Probe() {
  const theme = useTheme()
  Object.assign(window, {
    pick(field, value) { field === 'theme' ? theme.setTheme(value) : theme.setMode(value) },
    state(field) { return {view: field === 'theme' ? theme.themeName : theme.mode,
      cache: (field === 'theme' ? skinPref : modePref).own(profile),
      painted: document.documentElement.dataset.hermesTheme} }
  })
  return <pre>{JSON.stringify({theme: theme.themeName, mode: theme.mode})}</pre>
}
createRoot(document.getElementById('root')).render(<ThemeProvider><Probe/></ThemeProvider>)
`)

let config, held
// Saves wait for the case to answer them; reads answer with the config as it is when asked.
const transport = (req, res) => {
  let body = ''
  req.on('data', chunk => body += chunk)
  req.on('end', () => {
    const {client, request} = JSON.parse(body)
    const reply = value => { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify(value)) }
    if (request.method === 'PUT') {
      held[client].push({request, answer(ok) {
        if (!ok) return reply({failed: true})
        Object.assign(config.desktop, request.body.config.desktop)
        reply({ok: true, revision: ++config.revision})
      }})
    } else if (request.path.split('?')[0] !== '/api/config') reply({})
    else {
      const desktop = {...config.desktop}
      reply(request.path.includes('with_revision=true') ? {config: {desktop}, revision: config.revision} : {desktop})
    }
  })
}
// Vite reads port 0 as its default 5173, so concurrent runs would collide: take a free port instead.
const port = await new Promise((resolve, reject) => {
  const probe = net.createServer().once('error', reject).listen(0, '127.0.0.1', () => {
    const {port} = probe.address()
    probe.close(() => resolve(port))
  })
})
const server = await createServer({
  plugins: [{name: 'config-transport', configureServer(server) { server.middlewares.use('/__config', transport) }}],
  configFile: path.join(repo, 'apps/desktop/vite.config.ts'), root,
  server: {host: '127.0.0.1', port, strictPort: true, fs: {allow: [repo, root]}}
})
// Resolves once check passes; 5 s bound, polled because the transport answers asynchronously.
const until = async (check, what) => {
  for (const end = Date.now() + 5000; Date.now() < end; await new Promise(resolve => setTimeout(resolve, 10))) {
    const value = check()
    if (value) return value
  }
  throw Error('timeout: ' + what)
}
let browser
const results = [], errors = []
try {
  await server.listen()
  browser = await chromium.launch({headless: true,
    ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}),
    args: ['--disable-background-networking']})
  for (const profile of ['default', 'alpha']) for (const field of ['theme', 'theme_mode']) {
    const [, first, second] = field === 'theme' ? ['ember', 'mono', 'everforest'] : ['light', 'dark', 'system']
    for (const scenario of ['peer-fail-A-B', 'peer-fail-B-A', 'single-both-fail', 'single-first-ok', 'peer-last-ok', 'peer-first-ok', 'peer-first-ok-late']) {
      config = {desktop: {theme: 'ember', theme_mode: 'light'}, revision: 1}
      held = {A: [], B: []}
      const context = await browser.newContext()
      context.setDefaultTimeout(5000)
      await context.addInitScript(profile => {
        localStorage.setItem('hermes-desktop-theme-v2', 'ember')
        localStorage.setItem('hermes-desktop-mode-v1', 'light')
        if (profile !== 'default') {
          localStorage.setItem('hermes-desktop-profile-themes-v1', JSON.stringify({[profile]: 'ember'}))
          localStorage.setItem('hermes-desktop-profile-modes-v1', JSON.stringify({[profile]: 'light'}))
        }
      }, profile)
      const open = async client => {
        const page = await context.newPage()
        page.on('pageerror', e => errors.push(e.message))
        await page.goto(`http://127.0.0.1:${port}/?profile=${profile}&client=${client}`, {timeout: 60000})
        await page.waitForFunction(() => typeof window.pick === 'function', null, {timeout: 60000})
        return page
      }
      const state = page => page.evaluate(field => window.state(field), field)
      const pick = (page, value) => page.evaluate(({field, value}) => window.pick(field, value), {field, value})
      const settle = async (client, ok) => {
        const save = await until(() => held[client][0], `${client}'s save`)
        held[client].shift()
        const {request} = save
        if (request.connectionId !== 'A' || request.profile !== profile || request.path !== '/api/config') throw Error('wrong owner/endpoint')
        save.answer(ok)
      }
      let error
      try {
        const a = await open('A')
        const peer = scenario.startsWith('peer')
        const b = peer ? await open('B') : a
        await pick(a, first)
        await until(() => held.A.length === 1, "A's first save")
        await pick(b, second)
        // A single page queues its second save behind the first; a peer's goes out at once.
        if (peer) await until(() => held.B.length === 1, "B's save")
        const [one, two] = peer ? ['A', 'B'] : ['A', 'A']
        if (scenario === 'peer-fail-B-A' || scenario === 'peer-first-ok-late') { await settle(two, false); await settle(one, scenario === 'peer-first-ok-late') }
        else { await settle(one, scenario === 'single-first-ok' || scenario === 'peer-first-ok'); await settle(two, scenario === 'peer-last-ok') }
        const durable = config.desktop[field]
        for (const page of new Set([a, b])) {
          await page.waitForFunction(({field, durable}) => window.inflight === 0 &&
            window.state(field).view === durable && window.state(field).cache === durable, {field, durable})
        }
      } catch (e) { error = e.message.split('\n')[0] }
      const durable = config.desktop[field]
      const pages = context.pages()
      const states = await Promise.all(pages.map(state))
      const pass = !error && states.every(s => s.view === durable && s.cache === durable && (field !== 'theme' || s.painted === durable))
      results.push({profile, field, scenario, durable, states, error, pass})
      console.log(JSON.stringify(results.at(-1)))
      await context.close()
    }
  }
  const receipt = {results, errors, failed: results.filter(r => !r.pass).length, passed: results.filter(r => r.pass).length}
  await fs.writeFile(process.env.HERMES_APPEARANCE_RECEIPT || path.join(root, 'receipt.json'), JSON.stringify(receipt, null, 2))
  console.log('SUMMARY', JSON.stringify({failed: receipt.failed, passed: receipt.passed, errors}))
  if (receipt.failed || errors.length) process.exitCode = 1
} finally {
  await browser?.close()
  await server.close()
  await fs.rm(root, {recursive: true, force: true})
}
