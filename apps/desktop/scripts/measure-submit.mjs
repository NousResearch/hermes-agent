#!/usr/bin/env node
// Measure submit (Enter) latency in the composer.
//
// For each round:
//   1. Focus composer, type N chars of stub text
//   2. Mark a timestamp, fire Enter via Input.dispatchKeyEvent
//   3. Observe: time until the composer becomes empty (submit accepted),
//      time until the user message renders in the thread viewport,
//      time until the optional "running…" indicator appears,
//      time until the next frame is painted after the message renders.
//
// Pre-condition: a session is loaded (load via click-session.mjs first).
// Note: this DOES talk to the real gateway/agent, so each round triggers
//       a real prompt submission. Don't run this on a live conversation
//       you care about — use a throwaway session.

import { writeFileSync } from 'node:fs'

const args = Object.fromEntries(
  process.argv.slice(2).flatMap(s => {
    const m = s.match(/^--([^=]+)(?:=(.*))?$/)
    return m ? [[m[1], m[2] ?? true]] : []
  })
)
const PORT = Number(args.port ?? 9222)
const ROUNDS = Number(args.rounds ?? 3)

async function pickRenderer() {
  const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()
  return list.find(t => t.type === 'page' && t.url.startsWith('http'))
}

function connect(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url)
    let id = 0
    const pending = new Map()
    ws.addEventListener('open', () =>
      resolve({
        send(method, params = {}) {
          const myId = ++id
          ws.send(JSON.stringify({ id: myId, method, params }))
          return new Promise((res, rej) => pending.set(myId, { res, rej }))
        },
        close: () => ws.close()
      })
    )
    ws.addEventListener('error', reject)
    ws.addEventListener('message', ev => {
      const m = JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString('utf8'))
      if (m.id != null) {
        const p = pending.get(m.id)
        if (!p) return
        pending.delete(m.id)
        m.error ? p.rej(new Error(m.error.message)) : p.res(m.result)
      }
    })
  })
}

async function evalP(cdp, expr) {
  const r = await cdp.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text)
  return r.result.value
}

async function focusAndType(cdp, text) {
  await evalP(cdp, `
    (() => {
      const el = document.querySelector('[data-slot="composer-rich-input"]')
      if (!el) return
      el.focus()
      const range = document.createRange()
      range.selectNodeContents(el)
      range.collapse(false)
      const sel = window.getSelection()
      sel.removeAllRanges()
      sel.addRange(range)
    })()
  `)
  for (const c of text) {
    await cdp.send('Input.dispatchKeyEvent', { type: 'char', text: c, unmodifiedText: c })
    await new Promise(r => setTimeout(r, 8))
  }
}

async function submitAndMeasure(cdp, timeoutMs = 5000) {
  // Install observers, record submit time as performance.now() inside the page,
  // and wait for all milestones.
  return await evalP(cdp, `
    new Promise((resolve) => {
      const composer = document.querySelector('[data-slot="composer-rich-input"]')
      const threadRoot = document.querySelector('[data-slot="aui_thread-content"]') ||
                         document.querySelector('[data-slot="aui_thread-viewport"]')
      const startMessageCount = threadRoot ? threadRoot.querySelectorAll('[data-slot="aui_turn-pair"], [data-slot="aui_message"]').length : 0
      const startComposerText = composer ? composer.innerText : ''

      const milestones = { start: performance.now() }
      let done = false
      const finish = (reason) => {
        if (done) return
        done = true
        clearInterval(poll); clearTimeout(timer)
        composerObs.disconnect()
        threadObs?.disconnect()
        milestones.reason = reason
        milestones.end = performance.now()
        milestones.totalMs = milestones.end - milestones.start
        resolve(milestones)
      }

      const composerObs = new MutationObserver(() => {
        if (!milestones.composerClearedMs && composer && composer.innerText.length === 0) {
          milestones.composerClearedMs = performance.now() - milestones.start
        }
      })
      composer && composerObs.observe(composer, { childList: true, subtree: true, characterData: true })

      let threadObs = null
      if (threadRoot) {
        threadObs = new MutationObserver(() => {
          const c = threadRoot.querySelectorAll('[data-slot="aui_turn-pair"], [data-slot="aui_message"]').length
          if (!milestones.userMessageRenderedMs && c > startMessageCount) {
            milestones.userMessageRenderedMs = performance.now() - milestones.start
            requestAnimationFrame(() => {
              milestones.userMessagePaintMs = performance.now() - milestones.start
              finish('paint')
            })
          }
        })
        threadObs.observe(threadRoot, { childList: true, subtree: true })
      }

      const poll = setInterval(() => {
        if (milestones.composerClearedMs && !milestones.userMessageRenderedMs &&
            performance.now() - milestones.start > 2000) {
          finish('timeout-after-clear')
        }
      }, 100)
      const timer = setTimeout(() => finish('timeout-overall'), ${timeoutMs})

      // Send Enter immediately
      window.dispatchEvent(new KeyboardEvent('keydown'))  // no-op marker
      const enterEv = new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true, cancelable: true })
      composer?.dispatchEvent(enterEv)
    })
  `)
}

async function main() {
  const tgt = await pickRenderer()
  console.log('target', tgt.url)
  const cdp = await connect(tgt.webSocketDebuggerUrl)
  await cdp.send('Runtime.enable')

  const samples = []
  for (let i = 1; i <= ROUNDS; i++) {
    await focusAndType(cdp, `latency test ${i} ${'x'.repeat(40)}`)
    await new Promise(r => setTimeout(r, 300))
    const result = await submitAndMeasure(cdp, 4000)
    samples.push({ round: i, ...result })
    console.log(
      `r${i}: clear=${(result.composerClearedMs ?? -1).toFixed?.(0) ?? '?'}ms ` +
        `userMsg=${(result.userMessageRenderedMs ?? -1).toFixed?.(0) ?? '?'}ms ` +
        `paint=${(result.userMessagePaintMs ?? -1).toFixed?.(0) ?? '?'}ms ` +
        `reason=${result.reason}`
    )
    // wait for any agent activity to finish before next round so we're not piling up
    await new Promise(r => setTimeout(r, 4000))
  }
  writeFileSync('/tmp/hermes-submit-latency.json', JSON.stringify(samples, null, 2))
  console.log('\nwrote /tmp/hermes-submit-latency.json')
  cdp.close()
}

main().catch(e => {
  console.error('fatal:', e.stack ?? e.message)
  process.exit(1)
})
