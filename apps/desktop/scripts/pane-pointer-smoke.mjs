/**
 * Packaged-Electron pointer smoke for pane-scoped titlebar controls.
 *
 * Launches the PACKAGED app (release/mac-arm64/Hermes.app — run `npm run pack`
 * first) with an ISOLATED Electron userData sandbox. The app derives a fresh
 * sandbox HERMES_HOME under that userData dir (electron/main.ts
 * resolveHermesHome), so the harness pre-seeds a COPY of the installed
 * visual-workbench plugin into `<sandbox home>/desktop-plugins/` — nothing in
 * the user's real ~/.hermes is read from or written to by the sandboxed app's
 * state, and the real plugin directory is never modified.
 *
 * All interactions are REAL CDP pointer/keyboard events
 * (Input.dispatchMouseEvent / dispatchKeyEvent), never element.click() or
 * store mutation. Rendered zone IDs + dimensions are captured before/after
 * every sequence.
 *
 * Scenarios:
 *   1. fresh-profile hydration (plugin panes arrive visible AND report open);
 *   2. left sidebar / Browser / QC single-toggle independence (+ zone
 *      calibration from the observed flip);
 *   3. right sidebar (files) — including the workspace-gated branch;
 *   4. Browser × QC all pairwise ON/OFF combos;
 *   5. ⌘W on the main zone (never collapses a side / closes the window),
 *      qualified by a ⌘B keyboard-dispatch probe;
 *   6. layout reset (⌘-click the Layout editor button) — reopens chrome
 *      panes, preserves plugin pane preference;
 *   7. narrow viewport (700px, ≤768px breakpoint) — toggle reachability via
 *      real pointer + overlay reveal + round-trip restore;
 *   8. full main-process restart — persisted-false pane stays hidden; atom /
 *      persisted snapshot / button label / rendered visibility agree.
 *
 * Usage:
 *   node scripts/pane-pointer-smoke.mjs [--port 9228] [--keep-open]
 *
 * The CDP port is advisory — the harness picks a free port and re-probes the
 * live endpoint. Evidence: scripts/.smoke/pane-pointer-smoke-<ts>.jsonl.
 */

import { spawn } from 'node:child_process'
import fs from 'node:fs'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const DESKTOP_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const ARCH = process.arch === 'arm64' ? 'arm64' : 'x64'
const APP_BINARY =
  process.platform === 'darwin'
    ? path.join(DESKTOP_ROOT, 'release', `mac-${ARCH}`, 'Hermes.app', 'Contents', 'MacOS', 'Hermes')
    : path.join(DESKTOP_ROOT, 'release', `${process.platform === 'win32' ? 'win' : 'linux'}-unpacked`, 'Hermes')
const PLUGIN_SOURCE_DIR = path.join(os.homedir(), '.hermes', 'desktop-plugins', 'visual-workbench')

const ARGS = process.argv.slice(2)
const argValue = flag => {
  const i = ARGS.indexOf(flag)
  return i >= 0 ? ARGS[i + 1] : undefined
}
const REQUESTED_PORT = Number(argValue('--port') ?? 9228)
const KEEP_OPEN = ARGS.includes('--keep-open')

const PANE_STATE_IDS = {
  browser: 'visual-workbench:browser',
  files: 'file-browser',
  qc: 'visual-workbench:qc',
  sessions: 'chat-sidebar'
}

const BUTTONS = {
  browser: ['Hide Browser pane', 'Show Browser pane'],
  files: ['Hide right sidebar', 'Show right sidebar'],
  layout: ['Layout editor'],
  qc: ['Hide Quality Control pane', 'Show Quality Control pane'],
  sessions: ['Hide sidebar', 'Show sidebar']
}

const LOG_DIR = path.join(DESKTOP_ROOT, 'scripts', '.smoke')
fs.mkdirSync(LOG_DIR, { recursive: true })
const LOG_PATH = path.join(LOG_DIR, `pane-pointer-smoke-${new Date().toISOString().replace(/[:.]/g, '-')}.jsonl`)
const logLine = entry => fs.appendFileSync(LOG_PATH, `${JSON.stringify({ ts: Date.now(), ...entry })}\n`)

const failures = []
let checkCount = 0
function check(ok, label, detail) {
  checkCount += 1
  const status = ok ? 'PASS' : 'FAIL'
  console.log(`  [${status}] ${label}`)
  logLine({ detail, kind: 'check', label, ok })
  if (!ok) failures.push({ detail, label })
}

const note = text => {
  console.log(`  [note] ${text}`)
  logLine({ kind: 'note', note: text })
}

const sleep = ms => new Promise(r => setTimeout(r, ms))

async function freePort(preferred) {
  const usable = p =>
    new Promise(resolve => {
      const srv = net.createServer()
      srv.once('error', () => resolve(false))
      srv.listen(p, '127.0.0.1', () => srv.close(() => resolve(true)))
    })
  if (await usable(preferred)) return preferred
  for (let p = preferred + 1; p < preferred + 50; p++) {
    if (await usable(p)) return p
  }
  throw new Error('no free CDP port found')
}

// ---------------------------------------------------------------------------
// CDP plumbing
// ---------------------------------------------------------------------------

class Cdp {
  constructor(ws) {
    this.ws = ws
    this.id = 0
    this.pending = new Map()
    ws.addEventListener('message', ev => {
      const m = JSON.parse(ev.data)
      if (m.id != null && this.pending.has(m.id)) {
        const { reject, resolve } = this.pending.get(m.id)
        this.pending.delete(m.id)
        if (m.error) reject(new Error(m.error.message))
        else resolve(m.result)
      }
    })
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const i = ++this.id
      this.pending.set(i, { reject, resolve })
      this.ws.send(JSON.stringify({ id: i, method, params }))
    })
  }

  async eval(expression) {
    const r = await this.send('Runtime.evaluate', { awaitPromise: true, expression, returnByValue: true })
    if (r.exceptionDetails) {
      throw new Error(`page eval failed: ${r.exceptionDetails.text} ${r.exceptionDetails.exception?.description ?? ''}`)
    }
    return r.result.value
  }
}

async function fetchJson(url) {
  const res = await fetch(url)
  return res.json()
}

async function connectPage(port, timeoutMs = 120_000) {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    try {
      const list = await fetchJson(`http://127.0.0.1:${port}/json/list`)
      const tgt = list.find(t => t.type === 'page' && !/devtools/.test(t.url))
      if (tgt) {
        const ws = new WebSocket(tgt.webSocketDebuggerUrl)
        await new Promise((resolve, reject) => {
          ws.addEventListener('open', resolve)
          ws.addEventListener('error', reject)
        })
        return new Cdp(ws)
      }
    } catch {
      // endpoint not up yet
    }
    if (Date.now() > deadline) throw new Error(`CDP page target not reachable on port ${port}`)
    await sleep(500)
  }
}

async function browserClose(port) {
  try {
    const version = await fetchJson(`http://127.0.0.1:${port}/json/version`)
    const ws = new WebSocket(version.webSocketDebuggerUrl)
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', resolve)
      ws.addEventListener('error', reject)
    })
    ws.send(JSON.stringify({ id: 1, method: 'Browser.close' }))
    await sleep(2000)
    ws.close()
    return true
  } catch {
    return false
  }
}

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

function launchApp(port, userDataDir) {
  return spawn(APP_BINARY, [`--remote-debugging-port=${port}`], {
    detached: false,
    env: { ...process.env, HERMES_DESKTOP_USER_DATA_DIR: userDataDir },
    stdio: 'ignore'
  })
}

// ---------------------------------------------------------------------------
// Page-side capture + real pointer/keyboard input
// ---------------------------------------------------------------------------

const CAPTURE_EXPR = `(() => {
  const zones = [...document.querySelectorAll('[data-tree-group]')].map(el => {
    const r = el.getBoundingClientRect()
    return {
      id: el.dataset.treeGroup,
      displayed: r.width > 0 && r.height > 0,
      rect: { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) },
      tabs: [...el.querySelectorAll('[data-tree-tab]')].map(t => t.dataset.treeTab)
    }
  })
  const button = labels => {
    for (const label of labels) {
      const el = document.querySelector('button[aria-label="' + label + '"]')
      if (el) {
        const r = el.getBoundingClientRect()
        return { label, pressed: el.getAttribute('aria-pressed'), x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2) }
      }
    }
    return null
  }
  const overlayEl = [...document.querySelectorAll('div')].find(
    d => typeof d.className === 'string' && d.className.includes('z-40') && d.className.includes('shadow-2xl') && d.className.includes('inset-y-0')
  )
  const overlayRect = overlayEl ? overlayEl.getBoundingClientRect() : null
  const sdk = window.__HERMES_PLUGIN_SDK__
  const paneIds = ${JSON.stringify(PANE_STATE_IDS)}
  const sdkOpen = sdk
    ? Object.fromEntries(Object.entries(paneIds).map(([k, id]) => [k, sdk.host.panes.open(id).get()]))
    : null
  let persisted = {}
  try { persisted = JSON.parse(window.localStorage.getItem('hermes.desktop.paneStates.v1') || '{}') } catch {}
  return {
    buttons: {
      browser: button(${JSON.stringify(BUTTONS.browser)}),
      files: button(${JSON.stringify(BUTTONS.files)}),
      layout: button(${JSON.stringify(BUTTONS.layout)}),
      qc: button(${JSON.stringify(BUTTONS.qc)}),
      sessions: button(${JSON.stringify(BUTTONS.sessions)})
    },
    narrow: window.matchMedia('(max-width: 768px)').matches,
    overlay: overlayRect
      ? { x: Math.round(overlayRect.x), y: Math.round(overlayRect.y), w: Math.round(overlayRect.width), h: Math.round(overlayRect.height) }
      : null,
    persistedOpen: Object.fromEntries(Object.entries(paneIds).map(([k, id]) => [k, persisted[id]?.open])),
    sdkOpen,
    viewport: { h: window.innerHeight, w: window.innerWidth },
    zones
  }
})()`

async function capture(cdp, tag) {
  const snap = await cdp.eval(CAPTURE_EXPR)
  logLine({ kind: 'capture', snap, tag })
  return snap
}

async function pointerClick(cdp, x, y, modifiers = 0) {
  await cdp.send('Input.dispatchMouseEvent', { modifiers, type: 'mouseMoved', x, y })
  await cdp.send('Input.dispatchMouseEvent', { button: 'left', clickCount: 1, modifiers, type: 'mousePressed', x, y })
  await cdp.send('Input.dispatchMouseEvent', { button: 'left', clickCount: 1, modifiers, type: 'mouseReleased', x, y })
  logLine({ kind: 'pointer', modifiers, x, y })
}

async function keyTap(cdp, key, code, modifiers, vk) {
  await cdp.send('Input.dispatchKeyEvent', { code, key, modifiers, type: 'keyDown', windowsVirtualKeyCode: vk })
  await cdp.send('Input.dispatchKeyEvent', { code, key, modifiers, type: 'keyUp', windowsVirtualKeyCode: vk })
  logLine({ code, kind: 'key', modifiers })
}

/** Hit-test (x, y): what would a real pointer land on? Reports the covering
 *  onboarding overlay (it can mount seconds after boot, once the runtime
 *  check resolves unconfigured) and the resolved button aria-label. */
async function hitProbe(cdp, x, y) {
  return cdp.eval(`(() => {
    const el = document.elementFromPoint(${x}, ${y})
    if (!el) return { hit: null, overlay: false }
    const overlay = Boolean(el.closest('.z-1300'))
    const label = el.closest('button')?.getAttribute('aria-label') ?? null
    return { hit: label, overlay, tag: el.tagName }
  })()`)
}

/** Clear any pointer-blocking onboarding overlay via its real skip button. */
async function clearPointerPath(cdp, x, y) {
  for (let attempt = 0; attempt < 5; attempt++) {
    const probe = await hitProbe(cdp, x, y)
    if (!probe.overlay) return probe
    note(`onboarding overlay intercepts (${x}, ${y}) — dismissing via its skip button`)
    await dismissOnboarding(cdp)
    await sleep(800)
  }
  return hitProbe(cdp, x, y)
}

/** Click a titlebar toggle by its live button coordinates (real pointer),
 *  verifying the pointer would actually land on that button first. */
async function clickToggle(cdp, key, modifiers = 0) {
  let before = await capture(cdp, `before:${key}`)
  let btn = before.buttons[key]
  if (!btn) throw new Error(`toggle button for ${key} not found`)

  for (let attempt = 0; attempt < 5; attempt++) {
    const probe = await clearPointerPath(cdp, btn.x, btn.y)
    if (probe.hit === btn.label) break
    logLine({ attempt, btn, kind: 'hit-retry', probe })
    await sleep(700)
    before = await capture(cdp, `before:${key}`)
    btn = before.buttons[key]
    if (!btn) throw new Error(`toggle button for ${key} vanished`)
    if (attempt === 4) throw new Error(`pointer path to ${key} blocked (hits ${JSON.stringify(probe)})`)
  }

  await pointerClick(cdp, btn.x, btn.y, modifiers)
  await sleep(600)
  const after = await capture(cdp, `after:${key}`)
  return { after, before }
}

const displayedSet = snap => new Set(snap.zones.filter(z => z.displayed).map(z => z.id))

/** A minimized tool rail (e.g. the terminal's collapsed 28px strip) flips its
 *  display purely as a side effect of grid redistribution when a sibling zone
 *  hides/shows. It hosts no pane surface, so it never counts as a substantive
 *  visibility change — but every raw flip still lands in the evidence log. */
const RAIL_MAX_PX = 40

/** TOOL-PANEL zones (terminal/logs collapse-panes). Their zone visibility is
 *  owned by the collapse system (⌃` / rail), NOT by the four toggles under
 *  test, and it legitimately redistributes on sibling changes AND transiently
 *  expands on a layout reset (observed: reset shows grp-terminal 277px wide
 *  until the next tree commit re-collapses it — logged for the audit lanes).
 *  Excluded from substantive flips; raw flips keep them for evidence. */
const TOOL_ZONE_IDS = new Set(['grp-terminal', 'grp-logs'])

function isRailZone(before, after, id) {
  const rect = zoneRect(after, id) ?? zoneRect(before, id)
  return rect != null && rect.w > 0 && rect.h > 0 && Math.min(rect.w, rect.h) <= RAIL_MAX_PX
}

function flippedZones(before, after) {
  const b = displayedSet(before)
  const a = displayedSet(after)
  const flips = []
  for (const id of new Set([...b, ...a])) {
    if (b.has(id) !== a.has(id)) {
      flips.push({
        id,
        now: a.has(id) ? 'displayed' : 'hidden',
        rail: isRailZone(before, after, id) || TOOL_ZONE_IDS.has(id)
      })
    }
  }
  logLine({ flips, kind: 'flips' })
  return flips.filter(f => !f.rail)
}

const zoneRect = (snap, id) => snap.zones.find(z => z.id === id)?.rect

/** Assert a toggle changed at most its own zone and no other pane's state.
 *  `paneScopedOnly` (narrow mode): the grid legitimately redistributes
 *  leftover space when a zone collapses (absorber tracks like the terminal
 *  rail flip in/out) — only zones OWNED by other tracked panes count as
 *  foreign there; wide mode keeps the strict any-zone contract. */
function assertTargetOnly(key, { after, before }, paneZones, expectFlip, paneScopedOnly = false) {
  const flips = flippedZones(before, after)
  const ownZone = paneZones[key]
  const foreign = flips.filter(
    f => f.id !== ownZone && (!paneScopedOnly || Object.values(paneZones).includes(f.id))
  )
  check(foreign.length === 0, `${key}: no foreign ${paneScopedOnly ? 'pane-owned ' : ''}zone flipped (flips: ${JSON.stringify(flips)})`, { flips })
  if (expectFlip !== null) {
    check(
      flips.some(f => f.id === ownZone) === expectFlip,
      `${key}: own zone ${ownZone ?? '?'} ${expectFlip ? 'flipped' : 'stayed'}`,
      { flips, ownZone }
    )
  }
  for (const other of Object.keys(PANE_STATE_IDS).filter(k => k !== key)) {
    if (before.sdkOpen && after.sdkOpen) {
      check(
        before.sdkOpen[other] === after.sdkOpen[other],
        `${key}: ${other} open-atom unchanged (${before.sdkOpen[other]} -> ${after.sdkOpen[other]})`
      )
    }
  }
}

/** Atom ↔ persisted ↔ button ↔ rendered agreement for one pane. */
function assertAgreement(snap, key, paneZones, notes = '') {
  const open = snap.sdkOpen?.[key]
  const zone = paneZones[key]
  const rendered = zone ? displayedSet(snap).has(zone) : null
  const btn = snap.buttons[key]
  const pressedOk =
    btn == null ||
    btn.pressed == null ||
    (btn.pressed === 'true') === open ||
    btn.label === (open ? BUTTONS[key][0] : BUTTONS[key][1])
  check(
    open != null && (rendered === null || rendered === open) && pressedOk,
    `${key}: atom(${open}) / rendered(${rendered ?? 'n.a.'}) / button("${btn?.label}",pressed=${btn?.pressed}) agree ${notes}`,
    { btn, open, rendered, zone }
  )
  if (snap.persistedOpen[key] !== undefined) {
    check(snap.persistedOpen[key] === open, `${key}: persisted snapshot (${snap.persistedOpen[key]}) matches atom ${notes}`)
  }
}

// ---------------------------------------------------------------------------
// Main flow
// ---------------------------------------------------------------------------

async function waitReady(cdp) {
  const deadline = Date.now() + 150_000
  for (;;) {
    const snap = await capture(cdp, 'ready-poll').catch(() => null)
    if (
      snap &&
      snap.zones.length >= 2 &&
      snap.sdkOpen &&
      snap.buttons.sessions &&
      snap.buttons.files &&
      snap.buttons.browser &&
      snap.buttons.qc
    ) {
      return snap
    }
    if (Date.now() > deadline) {
      const body = await cdp.eval('document.body ? document.body.innerText.slice(0, 400) : "(no body)"').catch(() => 'n/a')
      throw new Error(`app never became ready (zones/buttons/SDK missing). body: ${body}`)
    }
    await sleep(1000)
  }
}

/**
 * A fresh sandbox HERMES_HOME has no provider, so the first-run onboarding
 * overlay (z-1300) covers the shell and swallows every pointer. Dismiss it the
 * way a user would: a REAL pointer click on "I'll choose a provider later"
 * (the skip persists to localStorage, so the restart leg boots clear).
 */
async function dismissOnboarding(cdp) {
  const deadline = Date.now() + 90_000
  for (;;) {
    const probe = await cdp
      .eval(`(() => {
        const overlay = [...document.querySelectorAll('div')].find(
          d => typeof d.className === 'string' && d.className.includes('z-1300') && !d.className.includes('pointer-events-none')
        )
        if (!overlay) return { state: 'clear' }
        const btn = [...overlay.querySelectorAll('button')].find(b => b.textContent.trim() === "I'll choose a provider later")
        if (btn) {
          const r = btn.getBoundingClientRect()
          return { state: 'skip', x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2) }
        }
        return { state: 'blocked' }
      })()`)
      .catch(() => null)

    if (probe?.state === 'clear') return
    if (probe?.state === 'skip') {
      note(`onboarding overlay: pointer-clicking "I'll choose a provider later" at (${probe.x}, ${probe.y})`)
      await pointerClick(cdp, probe.x, probe.y)
      await sleep(1200)
      continue
    }
    if (Date.now() > deadline) throw new Error('onboarding overlay never cleared')
    await sleep(1000)
  }
}

async function main() {
  if (!fs.existsSync(APP_BINARY)) {
    console.error(`Missing packaged app: ${APP_BINARY}\nRun: npm run pack`)
    process.exit(1)
  }
  if (!fs.existsSync(path.join(PLUGIN_SOURCE_DIR, 'plugin.js'))) {
    console.error(`Missing visual-workbench plugin to seed from: ${PLUGIN_SOURCE_DIR}`)
    process.exit(1)
  }

  const port = await freePort(REQUESTED_PORT)
  const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-pane-smoke-'))
  const userDataDir = path.join(sandbox, 'electron-user-data')
  fs.mkdirSync(userDataDir, { recursive: true })

  // resolveHermesHome() derives `<userData>/hermes-home` when
  // HERMES_DESKTOP_USER_DATA_DIR is set — seed a COPY of the plugin there so
  // the sandboxed backend serves it without touching the real ~/.hermes.
  const sandboxHome = path.join(userDataDir, 'hermes-home')
  fs.cpSync(PLUGIN_SOURCE_DIR, path.join(sandboxHome, 'desktop-plugins', 'visual-workbench'), { recursive: true })

  console.log(`packaged binary : ${APP_BINARY}`)
  console.log(`userData sandbox: ${userDataDir}`)
  console.log(`sandbox home    : ${sandboxHome} (plugin seeded)`)
  console.log(`CDP port        : ${port} (requested ${REQUESTED_PORT})`)
  console.log(`evidence log    : ${LOG_PATH}\n`)
  logLine({ binary: APP_BINARY, kind: 'meta', port, userDataDir })

  let child = launchApp(port, userDataDir)
  let cdp = await connectPage(port)

  console.log('— boot: waiting for shell + plugin toggles + SDK')
  await dismissOnboarding(cdp)
  const initial = await waitReady(cdp)

  console.log('\n— 1. fresh profile hydration')
  check(initial.sdkOpen.browser === true, 'fresh profile: Browser pane reports open', initial.sdkOpen)
  check(initial.sdkOpen.qc === true, 'fresh profile: QC pane reports open', initial.sdkOpen)
  check(initial.sdkOpen.sessions === true, 'fresh profile: sessions sidebar reports open', initial.sdkOpen)
  check(initial.persistedOpen.browser === true, 'fresh profile: Browser visibility seeded into persisted store')
  check(initial.persistedOpen.qc === true, 'fresh profile: QC visibility seeded into persisted store')

  console.log('\n— 1b. titlebar drag-strip geometry (OS drag regions vs control clusters)')
  {
    // An OS drag region wins hit-testing at the compositor level: DOM z-index,
    // pointer-events and even a no-drag class on a FIXED cluster do not
    // reliably carve it out. Synthetic CDP clicks bypass native drag
    // hit-testing entirely, so a strip that overlaps a titlebar button passes
    // every pointer scenario here while real mouse clicks drag the window.
    // The only trustworthy contract is geometric: no visible titlebar button
    // may intersect any [-webkit-app-region:drag] rect.
    const geo = await cdp.eval(`(() => {
      const rectOf = el => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height } }
      const cls = el => el.getAttribute('class') || ''
      const all = [...document.querySelectorAll('*')]
      const drags = all
        .filter(el => cls(el).includes('app-region:drag'))
        .map(el => ({ cls: cls(el).slice(0, 90), rect: rectOf(el) }))
        .filter(d => d.rect.w > 0 && d.rect.h > 0)
      const buttons = all
        .filter(el => cls(el).includes('app-region:no-drag'))
        .filter(el => getComputedStyle(el).position === 'fixed')
        .flatMap(el => [...el.querySelectorAll('button')])
        .map(b => ({ label: b.getAttribute('aria-label') || b.textContent.trim().slice(0, 24), rect: rectOf(b) }))
        .filter(b => b.rect.w > 0 && b.rect.h > 0)
      return { buttons, drags }
    })()`)
    logLine({ kind: 'drag-geometry', ...geo })
    check(geo.drags.length > 0, `drag-geometry: window drag strips present (${geo.drags.length})`)
    check(geo.buttons.length > 0, `drag-geometry: fixed titlebar buttons present (${geo.buttons.length})`)
    const overlaps = (a, b) =>
      Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x) > 1 && Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y) > 1
    for (const btn of geo.buttons) {
      const hits = geo.drags.filter(d => overlaps(btn.rect, d.rect))
      check(
        hits.length === 0,
        `drag-geometry: '${btn.label}' clear of every OS drag region`,
        hits.length ? { btn, hits } : undefined
      )
    }
  }

  const paneZones = {}
  console.log('\n— 2. calibration + single-toggle independence (pointer)')
  for (const key of ['sessions', 'browser', 'qc']) {
    const off = await clickToggle(cdp, key) // visible -> hidden
    const offFlips = flippedZones(off.before, off.after)
    check(offFlips.length === 1 && offFlips[0].now === 'hidden', `${key}: OFF flips exactly one zone`, { offFlips })
    paneZones[key] = offFlips[0]?.id
    check(off.after.sdkOpen[key] === false, `${key}: OFF drives open-atom false`)
    assertTargetOnly(key, off, paneZones, true)
    logLine({ key, kind: 'zone-owner', rect: zoneRect(off.before, paneZones[key]), zone: paneZones[key] })
    console.log(`      ${key} -> zone ${paneZones[key]} ${JSON.stringify(zoneRect(off.before, paneZones[key]))}`)

    const on = await clickToggle(cdp, key) // hidden -> visible
    assertTargetOnly(key, on, paneZones, true)
    check(on.after.sdkOpen[key] === true, `${key}: ON drives open-atom true`)
    assertAgreement(on.after, key, paneZones, '(after ON)')
  }

  console.log('\n— 3. right sidebar (files)')
  {
    const on = await clickToggle(cdp, 'files')
    const onFlips = flippedZones(on.before, on.after)
    check(on.after.sdkOpen.files === true, 'files: Show drives remembered intent true')
    if (onFlips.length === 1) {
      paneZones.files = onFlips[0].id
      console.log(`      files -> zone ${paneZones.files} ${JSON.stringify(zoneRect(on.after, paneZones.files))}`)
      assertTargetOnly('files', on, paneZones, true)
      assertAgreement(on.after, 'files', paneZones, '(after Show)')
      const off = await clickToggle(cdp, 'files')
      assertTargetOnly('files', off, paneZones, true)
      check(off.after.sdkOpen.files === false, 'files: Hide drives open-atom false')
      const on2 = await clickToggle(cdp, 'files')
      assertTargetOnly('files', on2, paneZones, true)
      const off2 = await clickToggle(cdp, 'files') // leave at default (hidden)
      assertTargetOnly('files', off2, paneZones, true)
    } else {
      check(onFlips.length === 0, 'files (workspace-gated): no zone flips without a workspace', { onFlips })
      assertTargetOnly('files', on, paneZones, null)
      note('files: workspace-gated (no cwd in fresh sandbox) — intent recorded, no zone moved; rendered flip covered when a workspace exists')
      await clickToggle(cdp, 'files') // restore intent
    }
  }

  console.log('\n— 4. Browser × QC pairwise combos (pointer)')
  const walk = [
    ['browser', false, [false, true]],
    ['qc', false, [false, false]],
    ['browser', true, [true, false]],
    ['qc', true, [true, true]]
  ]
  for (const [key, target, [b, q]] of walk) {
    const step = await clickToggle(cdp, key)
    assertTargetOnly(key, step, paneZones, true)
    check(step.after.sdkOpen[key] === target, `${key} -> ${target} (combo browser=${b}, qc=${q})`)
    check(step.after.sdkOpen.browser === b && step.after.sdkOpen.qc === q, `combo state is exactly (browser=${b}, qc=${q})`)
    assertAgreement(step.after, 'browser', paneZones, `(combo ${b},${q})`)
    assertAgreement(step.after, 'qc', paneZones, `(combo ${b},${q})`)
  }

  console.log('\n— 5. keyboard: ⌘B probe + ⌘W on the main zone')
  let keyboardWorks = false
  {
    const before = await capture(cdp, 'before:cmdb')
    await keyTap(cdp, 'b', 'KeyB', 4, 66)
    await sleep(600)
    const mid = await capture(cdp, 'after:cmdb')
    keyboardWorks = mid.sdkOpen.sessions !== before.sdkOpen.sessions
    note(`⌘B keyboard dispatch ${keyboardWorks ? 'reaches the app (sessions toggled)' : 'did NOT change sessions — ⌘W check is alive-only'}`)
    if (keyboardWorks) {
      const flips = flippedZones(before, mid)
      check(
        flips.every(f => f.id === paneZones.sessions),
        `⌘B: only the sessions zone flipped (flips: ${JSON.stringify(flips)})`
      )
      await keyTap(cdp, 'b', 'KeyB', 4, 66) // restore
      await sleep(600)
    }

    const preW = await capture(cdp, 'before:cmdw')
    const main = preW.zones.filter(z => z.displayed).sort((a, b) => b.rect.w * b.rect.h - a.rect.w * a.rect.h)[0]
    const mx = main.rect.x + Math.floor(main.rect.w / 2)
    const my = main.rect.y + Math.floor(main.rect.h / 2)
    await clearPointerPath(cdp, mx, my)
    await pointerClick(cdp, mx, my)
    await sleep(300)
    await keyTap(cdp, 'w', 'KeyW', 4, 87)
    await sleep(600)
    const after = await capture(cdp, 'after:cmdw').catch(() => null)
    check(after !== null, '⌘W: window still alive and inspectable')
    if (after) {
      const flips = flippedZones(preW, after)
      check(flips.length === 0, `⌘W: no zone visibility changed (flips: ${JSON.stringify(flips)})`)
      check(
        Object.keys(PANE_STATE_IDS).every(k => preW.sdkOpen[k] === after.sdkOpen[k]),
        '⌘W: no pane open-atom changed'
      )
    }
  }

  console.log('\n— 6. layout reset (⌘-click Layout editor) restores chrome panes, keeps plugin preference')
  {
    const hideSessions = await clickToggle(cdp, 'sessions')
    check(hideSessions.after.sdkOpen.sessions === false, 'reset prep: sessions hidden by pointer')
    const hideBrowser = await clickToggle(cdp, 'browser')
    check(hideBrowser.after.sdkOpen.browser === false, 'reset prep: Browser hidden by pointer')

    const reset = await clickToggle(cdp, 'layout', 4) // meta-click = reset
    await sleep(600)
    const after = await capture(cdp, 'after:reset')
    const terminalZone = after.zones.find(z => TOOL_ZONE_IDS.has(z.id) && z.displayed && z.rect.w > RAIL_MAX_PX)
    if (terminalZone) {
      note(
        `OBSERVATION for audit lanes: layout reset transiently expands the collapsed tool zone ${terminalZone.id} (rect ${JSON.stringify(terminalZone.rect)}) until the next tree commit re-collapses it — collapse-pane state does not survive the default-tree swap`
      )
    }
    check(after.sdkOpen.sessions === true, 'reset: sessions reopened through its owning store')
    check(
      displayedSet(after).has(paneZones.sessions) || after.zones.some(z => z.displayed && z.id !== paneZones.sessions && z.tabs.includes('sessions')),
      `reset: sessions rendered again (zone ${paneZones.sessions})`,
      { zones: after.zones }
    )
    check(after.sdkOpen.browser === false, 'reset: hidden Browser keeps its own preference (stays closed)')
    check(after.sdkOpen.qc === true, 'reset: QC untouched')
    check(after.persistedOpen.browser === false, 'reset: Browser persisted snapshot still false')
    logLine({ kind: 'reset', zones: reset.after.zones })

    // Re-open Browser by pointer; the reset may have re-homed it to a new
    // zone id — recalibrate from the observed flip for later scenarios.
    const on = await clickToggle(cdp, 'browser')
    const onFlips = flippedZones(on.before, on.after)
    check(onFlips.length === 1 && onFlips[0].now === 'displayed', 'post-reset: Show Browser reveals exactly one zone', { onFlips })
    paneZones.browser = onFlips[0]?.id ?? paneZones.browser
    check(on.after.sdkOpen.browser === true, 'post-reset: Browser open-atom true')
    assertAgreement(on.after, 'browser', paneZones, '(post-reset)')
  }

  console.log('\n— 7. narrow viewport (700px ≤ 768px breakpoint)')
  {
    const before = await capture(cdp, 'before:narrow')
    await cdp.send('Emulation.setDeviceMetricsOverride', { deviceScaleFactor: 0, height: 800, mobile: false, width: 700 })
    await sleep(800)
    const narrow = await capture(cdp, 'narrow')
    check(narrow.narrow === true, 'narrow: breakpoint reported by matchMedia')
    check(
      ['browser', 'qc'].every(k => narrow.sdkOpen[k] === before.sdkOpen[k]),
      'narrow: plugin pane open-atoms untouched by the breakpoint'
    )
    logLine({ kind: 'narrow-zones', zones: narrow.zones })

    // Toggle reachability with a REAL pointer while narrow: the left sidebar
    // button must still be hittable and reveal the sessions overlay.
    const btn = narrow.buttons.sessions
    check(btn !== null, 'narrow: sessions toggle button still present/hittable', { btn })
    if (btn) {
      await clearPointerPath(cdp, btn.x, btn.y)
      await pointerClick(cdp, btn.x, btn.y)
      await sleep(700)
      const revealed = await capture(cdp, 'narrow:sessions-toggle-1')
      const overlayShown = revealed.overlay !== null && revealed.overlay.w > 0
      check(overlayShown, `narrow: pointer toggle reveals the sessions overlay (rect: ${JSON.stringify(revealed.overlay)})`)
      check(
        ['browser', 'qc', 'files'].every(k => revealed.sdkOpen[k] === narrow.sdkOpen[k]),
        'narrow: overlay reveal left Browser/QC/files atoms untouched'
      )
      await clearPointerPath(cdp, btn.x, btn.y)
      await pointerClick(cdp, btn.x, btn.y) // second real click: overlay closes
      await sleep(700)
      const closed = await capture(cdp, 'narrow:sessions-toggle-2')
      check(closed.overlay === null, 'narrow: second pointer toggle closes the overlay', { overlay: closed.overlay })
    }

    // Browser/QC are not collapsible — their toggles still act pane-scoped.
    const step = await clickToggle(cdp, 'browser')
    assertTargetOnly('browser', step, paneZones, null, true)
    check(step.after.sdkOpen.browser === false, 'narrow: Browser toggle still drives its atom')
    const back = await clickToggle(cdp, 'browser')
    check(back.after.sdkOpen.browser === true, 'narrow: Browser toggle round-trips')

    await cdp.send('Emulation.clearDeviceMetricsOverride')
    await sleep(800)
    const restored = await capture(cdp, 'after:narrow')
    check(
      JSON.stringify([...displayedSet(restored)].sort()) === JSON.stringify([...displayedSet(before)].sort()),
      'narrow: restoring the width restores the exact displayed-zone set',
      { after: [...displayedSet(restored)], before: [...displayedSet(before)] }
    )
  }

  console.log('\n— 8. restart persistence (pointer-hide Browser, full main-process restart)')
  {
    const off = await clickToggle(cdp, 'browser')
    assertTargetOnly('browser', off, paneZones, true)
    check(off.after.persistedOpen.browser === false, 'pre-restart: Browser persisted open:false')
    check(off.after.sdkOpen.qc === true, 'pre-restart: QC still open')
    const sessionsPreRestart = off.after.sdkOpen.sessions
    logLine({ kind: 'pre-restart', zones: off.after.zones })

    const closed = await browserClose(port)
    if (!closed) child.kill('SIGTERM')
    await sleep(4000)

    child = launchApp(port, userDataDir)
    cdp = await connectPage(port)
    await dismissOnboarding(cdp) // skip persisted — returns immediately when clear
    const rebooted = await waitReady(cdp)
    logLine({ kind: 'post-restart', zones: rebooted.zones })
    check(rebooted.sdkOpen.browser === false, 'post-restart: Browser open-atom stays false')
    check(rebooted.persistedOpen.browser === false, 'post-restart: Browser persisted snapshot stays false')
    check(rebooted.sdkOpen.qc === true, 'post-restart: QC stays open')
    check(
      rebooted.sdkOpen.sessions === sessionsPreRestart,
      `post-restart: sessions sidebar keeps its pre-restart state (${sessionsPreRestart})`
    )
    const browserZoneNow = rebooted.zones.find(z => z.id === paneZones.browser)
    check(
      browserZoneNow == null || !browserZoneNow.displayed,
      `post-restart: Browser zone ${paneZones.browser} not rendered (rect: ${JSON.stringify(browserZoneNow?.rect)})`
    )
    check(
      rebooted.buttons.browser?.label === 'Show Browser pane' &&
        (rebooted.buttons.browser?.pressed === 'false' || rebooted.buttons.browser?.pressed == null),
      `post-restart: Browser button label/pressed agree ("${rebooted.buttons.browser?.label}", pressed=${rebooted.buttons.browser?.pressed})`
    )

    const on = await clickToggle(cdp, 'browser') // pointer round-trip after restart
    const onFlips = flippedZones(on.before, on.after)
    check(onFlips.length === 1 && onFlips[0].now === 'displayed', 'post-restart: Show Browser reveals exactly one zone', { onFlips })
    paneZones.browser = onFlips[0]?.id ?? paneZones.browser
    assertAgreement(on.after, 'browser', paneZones, '(post-restart re-open)')
  }

  if (!KEEP_OPEN) {
    const closed = await browserClose(port)
    if (!closed) child.kill('SIGTERM')
  }

  console.log(`\n${checkCount} checks, ${failures.length} failed.`)
  console.log(`evidence: ${LOG_PATH}`)
  if (failures.length > 0) {
    console.log('\nFailures:')
    for (const f of failures) console.log(`  - ${f.label}`)
    process.exit(1)
  }
}

main().catch(err => {
  console.error(`\nSMOKE ERROR: ${err.message}`)
  logLine({ error: String(err), kind: 'error' })
  process.exit(1)
})
