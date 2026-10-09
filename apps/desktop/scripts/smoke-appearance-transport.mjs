// Regression based on andrexibiza’s PR #93508 review 5408957939 fixture: racing appearance saves across windows.
// Two real Chromium pages per case run ThemeProvider, useHermesConfig and native storage events against a
// stateful config stub that numbers its revisions like hermes serve (GET ?with_revision, PUT { ok, revision })
// and orders commits, responses and config reads independently. Schedules: commit x response order (F7); a peer
// read before a save lands (F8); stale peer reads answered after that save settles or around the re-read its
// cache write starts; a later pick while that re-read is held; a picker that leaves the profile before its save
// lands. For theme and mode on the default and a named profile, both pages, the cache and the config must agree,
// every save is one the schedule picked, and only a failed save's own window may say it failed. Every step waits
// on transport, storage or page state (5 s bounds), never on elapsed time.
// Run from the repository root: CHROMIUM_PATH=/path/to/chrome node apps/desktop/scripts/smoke-appearance-transport.mjs
import {EventEmitter} from 'node:events'
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
const root = await fs.mkdtemp(path.join(os.tmpdir(), 'hermes-appearance-transport-'))
await fs.writeFile(path.join(root, 'index.html'), '<div id="root"></div><script type="module" src="/entry.tsx"></script>')
await fs.writeFile(path.join(root, 'entry.tsx'), `
import React from 'react'
import {createRoot} from 'react-dom/client'
document.documentElement.dataset.hermesDesktopHost = 'browser'
const params = new URLSearchParams(location.search)
const profile = params.get('profile') || 'default', client = params.get('client')
// Storage events from one window arrive in write order: once its marker arrives, its earlier writes have too.
const seen = {}
addEventListener('storage', event => { if (event.key) seen[event.key] = event.newValue })
Object.assign(window, {seen, inflight: 0, saveAnswers: 0,
  mark() { const id = crypto.randomUUID(); localStorage.setItem('smoke-barrier', id); return id },
  hermesDesktop: {async api(request) {
    window.inflight++
    try {
      const response = await fetch('/__config', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({client, request})})
      const result = await response.json()
      if (result.failed) throw Error('save failed')
      return result
    } finally {
      window.inflight--
      if (request.method === 'PUT') window.saveAnswers++
    }
  }}})
const {setApiRequestConnection, setApiRequestLocalMode, setApiRequestProfile} = await import('@/api/client')
const {$activeGatewayProfile} = await import('@/store/profile')
const {useHermesConfig} = await import('@/app/session/hooks/use-hermes-config')
const {$connection} = await import('@/store/session')
$connection.set({connectionId: 'A', profile, mode: 'remote', baseUrl: 'http://test.invalid', logs: [], token: '', wsUrl: '',
  isFullscreen: false, nativeOverlayWidth: 0, windowButtonPosition: null})
const {ThemeProvider, useTheme, skinPref, modePref} = await import('@/themes/context')
const {$notifications} = await import('@/store/notifications')
setApiRequestLocalMode(false); setApiRequestConnection('A'); setApiRequestProfile(profile)
$activeGatewayProfile.set(profile)
Object.assign(window, {
  errors: () => $notifications.get().filter(n => n.kind === 'error').length,
  // A profile switch as the gateway applies one: request scope, active profile and connection descriptor.
  route(next) { setApiRequestProfile(next); $activeGatewayProfile.set(next); $connection.set({...$connection.get(), profile: next}) }
})
function Probe() {
  const theme = useTheme()
  const {refreshHermesConfig} = useHermesConfig({activeSessionIdRef: {current: null}})
  Object.assign(window, {
    refresh: refreshHermesConfig,
    pick(field, value) { field === 'theme' ? theme.setTheme(value) : theme.setMode(value) },
    state(field) { return {view: field === 'theme' ? theme.themeName : theme.mode,
      cache: (field === 'theme' ? skinPref : modePref).own(profile),
      painted: document.documentElement.dataset.hermesTheme} }
  })
  return <pre>{JSON.stringify({theme: theme.themeName, mode: theme.mode})}</pre>
}
createRoot(document.getElementById('root')).render(<ThemeProvider><Probe/></ThemeProvider>)
`)

let owner, durable, elsewhere, saves, heldReads, holding
const changed = new EventEmitter()
// The profile under test has the config every case checks; a page that switched away reads and writes another.
const configOf = request => !request.profile || request.profile === owner ? durable : elsewhere
const transport = (req, res) => {
  let body = ''
  req.on('data', chunk => body += chunk)
  req.on('end', () => {
    const {client, request} = JSON.parse(body)
    const reply = value => { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify(value)) }
    const hold = () => { holding[client]++; return value => { holding[client]--; reply(value) } }
    const config = configOf(request), desktop = {...config.desktop}
    // A read answers with the config, and the revision, as they were when it was served.
    const served = request.path.includes('with_revision=true') ? {config: {desktop}, revision: config.revision} : {desktop}
    // The owner's saves wait for the schedule to commit and answer them; its reads are answered unless held.
    if (request.method === 'PUT' && config === durable) {
      const item = {answer: hold(), commit() {
        if (!item.committed) { Object.assign(durable.desktop, request.body.config.desktop); item.revision = ++durable.revision }
        item.committed = true
      }}
      saves[client].push(item)
    } else if (request.method === 'PUT') { Object.assign(config.desktop, request.body.config.desktop); reply({ok: true, revision: ++config.revision}) }
    else if (request.path.split('?')[0] !== '/api/config') reply({})
    else if (config === durable && heldReads[client]) { const answer = hold(); heldReads[client].push(() => answer(served)) }
    else reply(served)
    changed.emit('change')
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

// Resolves once the transport passes check; every request it records re-checks.
const until = (check, what) => new Promise((resolve, reject) => {
  const test = () => { const value = check(); if (value) { stop(); resolve(value) } }
  const stop = () => { clearTimeout(timer); changed.off('change', test) }
  const timer = setTimeout(() => { stop(); reject(Error('timeout: ' + what)) }, 5000)
  changed.on('change', test)
  test()
})
const heldSave = client => until(() => saves[client].find(item => !item.answered), `${client}'s save`)
const commit = async client => (await heldSave(client)).commit()
const respond = async (client, ok) => {
  const item = await heldSave(client)
  item.answered = true
  if (ok) item.commit()
  item.answer(ok ? {ok: true, revision: item.revision} : {failed: true})
}
// Holds the client's next read of the owner's config; the result answers it with the config as it was when asked.
const holdRead = client => {
  heldReads[client] = []
  const held = until(() => heldReads[client]?.[0], `${client}'s config read`).then(deliver => { heldReads[client] = null; return deliver })
  held.catch(() => {})
  return held
}
const named = (what, wait) => wait.catch(e => { throw Error(`${what}: ${e.message.split('\n')[0]}`) })
const state = (page, field) => page.evaluate(field => window.state(field), field)
const shows = (page, field, value) => page.waitForFunction(({field, value}) => window.state(field).view === value, {field, value})
const pick = (page, field, value) => page.evaluate(({field, value}) => window.pick(field, value), {field, value})
// page.evaluate has no timeout of its own; a refresh that never settles fails the case instead of hanging the run.
const refresh = page => {
  let timer
  const done = Promise.race([page.evaluate(() => window.refresh()),
    new Promise((_, reject) => { timer = setTimeout(() => reject(Error('timeout: config refresh')), 5000) })])
    .finally(() => clearTimeout(timer))
  done.catch(() => {})
  return done
}
const barrier = async (from, to) => {
  const id = await from.evaluate(() => window.mark())
  await to.waitForFunction(id => window.seen['smoke-barrier'] === id, id)
}
// Answers a held save; once its window has the answer, everything it wrote reaches the other window.
const settle = async (pages, client, ok) => {
  const [page, peer] = client === 'A' ? [pages.A, pages.B] : [pages.B, pages.A]
  const before = await page.evaluate(() => window.saveAnswers)
  await respond(client, ok)
  await page.waitForFunction(before => window.saveAnswers > before, before)
  await barrier(page, peer)
}
// Nothing in flight but what the transport holds, and every storage write delivered both ways, until a round
// writes nothing: the schedule, not CPU load, orders what happens next.
const quiet = async pages => {
  let last
  for (let round = 0; round < 10; round++) {
    await barrier(pages.A, pages.B); await barrier(pages.B, pages.A)
    await Promise.all(Object.entries(pages).map(([client, page]) => page.waitForFunction(held => window.inflight === held, holding[client])))
    const now = JSON.stringify(await Promise.all(Object.values(pages).map(page =>
      page.evaluate(() => Object.entries(localStorage).filter(([key]) => key !== 'smoke-barrier').sort()))))
    if (now === last) return
    last = now
  }
  throw Error('storage never settled')
}
// Both fields: the view, the shared cache and, for the theme, the painted skin all match the config.
const agrees = config => ['theme', 'theme_mode'].every(field => {
  const s = window.state(field)
  return s.view === config[field] && s.cache === config[field] && (field !== 'theme' || s.painted === config.theme)
})

// Each schedule starts with A's pick painted in A and its save held at the transport. A peer shows a pick once
// its save lands: the picker caches it then, and the peer re-reads.
const scenarios = {
  // F7: the transport commits both saves in one order and answers them in either.
  ...Object.fromEntries(['AB', 'BA'].flatMap(commits => ['BA', 'AB'].map(answers => [`commit${commits}-resp${answers}`,
    async ({b, pages, field, second}) => {
      await pick(b, field, second); await shows(b, field, second); await heldSave('B')
      for (const client of commits) await commit(client)
      for (const client of answers) await settle(pages, client, true)
    }]))),
  // F8: B reads the config before A's save lands.
  'peer-read-then-success': async ({b, pages}) => { await refresh(b); await settle(pages, 'A', true) },
  'peer-read-then-failure': async ({b, pages}) => { await refresh(b); await settle(pages, 'A', false) },
  // B's read was served before A's save landed and answers after B learned of it.
  'stale-peer-read-after-settle': async ({b, pages}) => {
    const read = holdRead('B'), load = refresh(b), deliver = await read
    await settle(pages, 'A', true)
    deliver(); await load
  },
  // B adopts one stale read (F8); a second, served before A's save landed, answers before or after the re-read
  // A's cache write starts in B.
  ...Object.fromEntries(['before', 'after'].map(order => [`second-stale-peer-read-${order}-reread`, async ({b, pages}) => {
    await refresh(b)
    const read = holdRead('B'), load = refresh(b), deliverStale = await read
    const reread = holdRead('B')
    await settle(pages, 'A', true)
    const deliverReread = await reread
    if (order === 'before') { deliverStale(); await load; deliverReread() } else { deliverReread(); deliverStale(); await load }
  }])),
  // B adopts a stale read (F8) and A's save lands. B's re-read is held while a later pick (either window, either
  // field) goes out; that pick's save settles last.
  ...Object.fromEntries([['own', 'other'], ['peer', 'other'], ['own', 'same'], ['peer', 'same']].map(([who, what]) =>
    [`reread-then-${who}-${what}-pick${what === 'same' ? '-fails' : ''}`, async ({b, pages, field, other, otherValue, second}) => {
      await refresh(b)
      const reread = holdRead('B')
      await settle(pages, 'A', true)
      const deliver = await reread, client = who === 'own' ? 'A' : 'B'
      await quiet(pages)
      await (what === 'other' ? pick(pages[client], other, otherValue) : pick(pages[client], field, second))
      await heldSave(client); await quiet(pages)
      deliver()
      await quiet(pages)
      await settle(pages, client, what === 'other')
    }])),
  // B adopts a stale read; A leaves for another profile and loads it before its save lands.
  'picker-leaves-owner': async t => {
    const {a, b, pages, profile, field, base, first} = t
    await refresh(b); await shows(b, field, base)
    await a.evaluate(() => window.route('beta')); await refresh(a)
    await settle(pages, 'A', true)
    await named('B re-reads when the window that left caches its save', b.waitForFunction(({field, first}) => {
      const s = window.state(field)
      return s.view === first && s.cache === first
    }, {field, first}))
    // Returning, A paints what it last knew of the owner's config, which the shared cache now holds too.
    await a.evaluate(profile => window.route(profile), profile)
    await named('A paints the owner on return', a.waitForFunction(field => window.state(field).view === window.state(field).cache, field))
    t.returned = (await state(a, field)).view
    await refresh(a)
  }
}
// Only a failed save's own window says so; a peer's read must not swallow it.
const failedSaves = {'peer-read-then-failure': '1/0', 'reread-then-own-same-pick-fails': '1/0', 'reread-then-peer-same-pick-fails': '0/1'}

let browser
const results = [], errors = []
try {
  await server.listen()
  browser = await chromium.launch({headless: true,
    ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}),
    args: ['--disable-background-networking']})
  for (const profile of ['default', 'alpha']) for (const field of ['theme', 'theme_mode']) {
    const [base, first, second] = field === 'theme' ? ['ember', 'mono', 'everforest'] : ['light', 'dark', 'system']
    const [other, otherValue] = field === 'theme' ? ['theme_mode', 'dark'] : ['theme', 'mono']
    for (const [scenario, run] of Object.entries(scenarios)) {
      owner = profile; saves = {A: [], B: []}; heldReads = {A: null, B: null}; holding = {A: 0, B: 0}; changed.removeAllListeners()
      durable = {desktop: {theme: 'ember', theme_mode: 'light'}, revision: 1}
      elsewhere = {desktop: {theme: 'everforest', theme_mode: 'system'}, revision: 1}
      const context = await browser.newContext()
      context.setDefaultTimeout(5000)
      const a = await context.newPage(), b = await context.newPage(), pages = {A: a, B: b}
      const t = {a, b, pages, profile, field, other, otherValue, base, first, second}
      let error
      try {
        for (const [client, page] of Object.entries(pages)) {
          page.on('pageerror', e => errors.push(e.message))
          await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/?profile=${profile}&client=${client}`, {timeout: 60000})
          await page.waitForFunction(() => typeof window.pick === 'function', null, {timeout: 60000})
          await refresh(page)
        }
        await pick(a, field, first); await shows(a, field, first); await heldSave('A')
        await run(t)
        await named('pages go quiet', quiet(pages))
        await named('pages agree with the config', Promise.all([a, b].map(page => page.waitForFunction(agrees, durable.desktop))))
        // The schedule answers every save it picked; one still held at the transport is a write-back.
        if (Object.values(saves).flat().some(item => !item.answered)) throw Error('a save nobody picked is held')
      } catch (e) { error = e.message.split('\n')[0] }
      const config = {...durable.desktop}
      const [stateA, stateB] = await Promise.all([a, b].map(page =>
        page.evaluate(() => ({theme: window.state('theme'), theme_mode: window.state('theme_mode'), notices: window.errors()}))))
      const notices = `${stateA.notices}/${stateB.notices}`
      const pass = !error && notices === (failedSaves[scenario] ?? '0/0') && (t.returned ?? config[field]) === config[field]
      results.push({profile, field, scenario, config, pages: {A: stateA, B: stateB}, notices, returned: t.returned, error, pass})
      const shown = s => `${s.theme.view}/${s.theme.cache}/${s.theme.painted}+${s.theme_mode.view}/${s.theme_mode.cache}`
      console.log(`${pass ? 'PASS' : 'FAIL'} ${profile} ${field} ${scenario} config=${config.theme}+${config.theme_mode}`,
        `A=${shown(stateA)} B=${shown(stateB)} notices=${notices}${t.returned ? ' returned=' + t.returned : ''}${error ? ' error=' + error : ''}`)
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
