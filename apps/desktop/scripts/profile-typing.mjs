#!/usr/bin/env node
// Profile typing lag in the Electron renderer by:
//  1. Connecting to a running renderer via CDP (--remote-debugging-port=9222)
//  2. Focusing the composer contentEditable
//  3. Starting CPU profile + heap snapshot
//  4. Synthesizing keystrokes via Input.dispatchKeyEvent (so the run is
//     reproducible, no human-typing variance)
//  5. Stopping the profile + capturing a second heap snapshot
//  6. Saving .cpuprofile + .heapsnapshot
//
// Usage:
//   node apps/desktop/scripts/profile-typing.mjs
//     [--port=9222] [--out=/tmp/hermes-typing]
//     [--chars=400]              # how many characters to type
//     [--cps=30]                 # keystrokes per second
//     [--text="..."]             # override generated text
//     [--no-heap]                # skip heap snapshots
//     [--seconds=N]              # idle-record for N seconds instead of typing
//
// Zero deps — uses Node 24's global WebSocket + fetch.

import { writeFileSync } from 'node:fs'

const args = Object.fromEntries(
  process.argv.slice(2).flatMap(s => {
    const m = s.match(/^--([^=]+)(?:=(.*))?$/)
    return m ? [[m[1], m[2] ?? true]] : []
  })
)

const PORT = Number(args.port ?? 9222)
const OUT = String(args.out ?? `/tmp/hermes-typing-${Date.now()}`)
const CHARS = Number(args.chars ?? 400)
const CPS = Number(args.cps ?? 30)
const HEAP = args['no-heap'] ? false : true
const IDLE_SECONDS = args.seconds ? Number(args.seconds) : null
const CUSTOM_TEXT = args.text === undefined || args.text === true ? null : String(args.text)

const log = (...m) => console.log('[profile]', ...m)
const banner = m => console.log(`\n========== ${m} ==========`)

async function pickRenderer() {
  const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()
  const pages = list.filter(t => t.type === 'page' && t.url.startsWith('http'))
  if (!pages.length) {
    console.error('No renderer page. Targets:')
    list.forEach(t => console.error(' ', t.type, t.url))
    process.exit(2)
  }
  return pages[0]
}

function connect(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url)
    let id = 0
    const pending = new Map()
    const events = new Map()
    ws.addEventListener('open', () =>
      resolve({
        send(method, params = {}) {
          const myId = ++id
          ws.send(JSON.stringify({ id: myId, method, params }))
          return new Promise((res, rej) => pending.set(myId, { res, rej }))
        },
        on(method, h) {
          if (!events.has(method)) events.set(method, [])
          events.get(method).push(h)
        },
        close: () => ws.close()
      })
    )
    ws.addEventListener('error', reject)
    ws.addEventListener('message', ev => {
      const txt = typeof ev.data === 'string' ? ev.data : ev.data.toString('utf8')
      const m = JSON.parse(txt)
      if (m.id != null) {
        const p = pending.get(m.id)
        if (!p) return
        pending.delete(m.id)
        m.error ? p.rej(new Error(m.error.message)) : p.res(m.result)
      } else if (m.method) {
        ;(events.get(m.method) ?? []).forEach(h => h(m.params))
      }
    })
  })
}

async function captureHeap(cdp, path) {
  log(`heap snapshot → ${path}`)
  const chunks = []
  cdp.on('HeapProfiler.addHeapSnapshotChunk', ({ chunk }) => chunks.push(chunk))
  await cdp.send('HeapProfiler.enable')
  await cdp.send('HeapProfiler.takeHeapSnapshot', { reportProgress: false, captureNumericValue: true })
  writeFileSync(path, chunks.join(''))
  log(`  ${(Buffer.byteLength(chunks.join(''), 'utf8') / 1024 / 1024).toFixed(1)} MB`)
}

async function focusComposer(cdp) {
  // Focus the rich-input contentEditable. RICH_INPUT_SLOT is the data-slot
  // value used by the composer's editable div. If focus fails (no composer
  // mounted yet — disabled state, etc.) the script logs and continues; the
  // profile will still show idle behavior.
  const result = await cdp.send('Runtime.evaluate', {
    expression: `
      (() => {
        const el = document.querySelector('[data-slot="composer-rich-input"]')
        if (!el) return { ok: false, reason: 'composer-rich-input not found' }
        el.focus()
        // place caret at end
        const range = document.createRange()
        range.selectNodeContents(el)
        range.collapse(false)
        const sel = window.getSelection()
        sel.removeAllRanges()
        sel.addRange(range)
        return { ok: true, text: el.innerText.length }
      })()
    `,
    returnByValue: true
  })
  if (!result.result.value?.ok) {
    log(`focus failed: ${result.result.value?.reason ?? 'unknown'}`)
    return false
  }
  log(`composer focused (existing text length: ${result.result.value.text})`)
  return true
}

function genText(n) {
  const lorem =
    'the quick brown fox jumps over the lazy dog while the agent thinks really hard about why typing into this composer feels like wading through molasses on a hot afternoon '
  let s = ''
  while (s.length < n) s += lorem
  return s.slice(0, n)
}

async function dispatchChar(cdp, ch) {
  // For printable chars, char + keypress is enough — Electron treats it as text input
  // and the contentEditable input event fires. For Enter / Space we could add
  // specials; this run is one long line.
  await cdp.send('Input.dispatchKeyEvent', {
    type: 'char',
    text: ch,
    unmodifiedText: ch
  })
}

async function typeText(cdp, text, cps) {
  const intervalMs = Math.max(1, Math.round(1000 / cps))
  const start = Date.now()
  for (let i = 0; i < text.length; i++) {
    await dispatchChar(cdp, text[i])
    // Pace evenly; account for dispatch latency so we don't drift much.
    const expected = start + (i + 1) * intervalMs
    const wait = expected - Date.now()
    if (wait > 0) await new Promise(r => setTimeout(r, wait))
  }
}

async function main() {
  log(`CDP port ${PORT}, out ${OUT}`)
  const target = await pickRenderer()
  log(`target ${target.url}`)
  const cdp = await connect(target.webSocketDebuggerUrl)
  await cdp.send('Runtime.enable')
  await cdp.send('Page.enable')
  await cdp.send('Profiler.enable')

  // Pre-GC so the cpu profile + heap delta are clean.
  try {
    await cdp.send('HeapProfiler.collectGarbage')
  } catch (e) {
    log('GC skipped:', e.message)
  }

  if (HEAP) await captureHeap(cdp, `${OUT}.before.heapsnapshot`)

  // 1ms sampling — fine enough for per-frame React work.
  await cdp.send('Profiler.setSamplingInterval', { interval: 1000 })

  let typedText = ''
  if (!IDLE_SECONDS) {
    const focused = await focusComposer(cdp)
    if (!focused) {
      log('aborting — composer not focusable. Make sure the app is past the boot screen.')
      cdp.close()
      process.exit(3)
    }
    typedText = CUSTOM_TEXT ?? genText(CHARS)
  }

  await cdp.send('Profiler.start')

  if (IDLE_SECONDS) {
    banner(`IDLE recording for ${IDLE_SECONDS}s — DO NOT TOUCH`)
    await new Promise(r => setTimeout(r, IDLE_SECONDS * 1000))
  } else {
    banner(`TYPING ${typedText.length} chars @ ${CPS} cps (≈${(typedText.length / CPS).toFixed(1)}s)`)
    const t0 = Date.now()
    await typeText(cdp, typedText, CPS)
    log(`typing wall time: ${((Date.now() - t0) / 1000).toFixed(2)}s`)
    // Settle frame for trailing React work.
    await new Promise(r => setTimeout(r, 500))
  }

  banner('STOP — saving profile')
  const { profile } = await cdp.send('Profiler.stop')
  writeFileSync(`${OUT}.cpuprofile`, JSON.stringify(profile))
  log(`cpu profile → ${OUT}.cpuprofile (${(JSON.stringify(profile).length / 1024 / 1024).toFixed(1)} MB)`)

  if (HEAP) {
    try {
      await cdp.send('HeapProfiler.collectGarbage')
    } catch {}
    await captureHeap(cdp, `${OUT}.after.heapsnapshot`)
  }

  // Quick triage: top-self-time frames from the profile.
  const top = summarizeProfile(profile)
  banner('TOP SELF-TIME FRAMES')
  for (const row of top.slice(0, 20)) {
    console.log(
      `  ${row.selfMs.toFixed(1).padStart(7)}ms  ${row.functionName || '(anonymous)'}` +
        `  ${row.url ? '· ' + row.url.replace(/^.*\/src\//, 'src/').slice(0, 80) : ''}`
    )
  }
  console.log()
  log(`total samples: ${top.totalSamples}, total time: ${(top.totalMs / 1000).toFixed(2)}s`)

  cdp.close()
}

function summarizeProfile(profile) {
  // Cumulative samples = how many sampling ticks landed on each node.
  // selfMs = own time only, using sampling interval.
  const intervalMs = (profile.endTime - profile.startTime) / 1000 / Math.max(1, profile.samples?.length ?? 1)
  const counts = new Map()
  for (const s of profile.samples ?? []) counts.set(s, (counts.get(s) ?? 0) + 1)
  const rows = profile.nodes.map(n => {
    const self = counts.get(n.id) ?? 0
    return {
      id: n.id,
      functionName: n.callFrame.functionName,
      url: n.callFrame.url,
      lineNumber: n.callFrame.lineNumber,
      selfSamples: self,
      selfMs: self * intervalMs
    }
  })
  rows.sort((a, b) => b.selfSamples - a.selfSamples)
  rows.totalSamples = (profile.samples ?? []).length
  rows.totalMs = ((profile.endTime - profile.startTime) / 1000)
  return rows
}

main().catch(e => {
  console.error('[profile] fatal:', e.stack ?? e.message)
  process.exit(1)
})
