#!/usr/bin/env node
/**
 * AC1 feasibility spike (Phase 0, desktop startup-latency).
 *
 * The spec's AC1 target is: warm-cache cold launch paints full UI at <1.5s.
 * §0 measured window-visible at ~1.41s on the Studio, leaving ~90ms for the
 * cache path: read 3 JSON files off local disk -> IPC to renderer -> hydrate
 * stores -> React paint. This spike measures the DISK+PARSE+SERIALIZE half of
 * that budget with realistic payloads, so we know before building Phases 1-2
 * whether <1.5s is achievable or AC1 must be softened to a measured floor.
 *
 * What it does NOT measure: the IPC hop and React render (those need the live
 * app). It bounds the FILE half; if even that blows the ~90ms budget, <1.5s is
 * already unreachable and we stop. If it's a few ms (expected for local SSD),
 * the budget survives to the IPC+render measurement in Phase 2.
 *
 * Run: node electron/spike-cache-read-cost.mjs
 */

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// Realistic payloads matching the spec's cache shape (§D2, §7):
//  - sessions.json: a 48-row session-list page (the list RPC page size) + total
//  - status.json: a small status snapshot
//  - transcript-<id>.json: last N=200 rows of the active session's transcript
function makeSessions(n = 48) {
  return {
    schema: 1,
    appVersion: '0.17.0',
    gatewayUrl: 'http://mac-studio-m3u:9119',
    total: 2576,
    sessions: Array.from({ length: n }, (_, i) => ({
      id: `20260710_${String(180000 + i).padStart(6, '0')}_${'a'.repeat(8)}`,
      title: `Session number ${i} with a reasonably long human title that people actually use`,
      pinned: i < 5,
      archived: false,
      source: 'cli',
      updatedAt: `2026-07-10T${String(10 + (i % 12)).padStart(2, '0')}:30:00Z`,
      messageCount: 20 + i * 3
    }))
  }
}

function makeStatus() {
  return {
    schema: 1,
    appVersion: '0.17.0',
    gatewayUrl: 'http://mac-studio-m3u:9119',
    authRequired: true,
    gatewayRunning: true,
    model: 'claude-apr/claude-opus-4-8',
    provider: 'claude-apr'
  }
}

function makeTranscript(rows = 200) {
  // Rows carry real chat CONTENT (the honest I4 framing) — size them like real
  // messages (a few hundred chars each, some longer) so the parse cost is real.
  return {
    schema: 1,
    appVersion: '0.17.0',
    gatewayUrl: 'http://mac-studio-m3u:9119',
    storedSessionId: '20260710_182847_84cb906d',
    rows: Array.from({ length: rows }, (_, i) => ({
      id: i,
      role: i % 2 === 0 ? 'user' : 'assistant',
      text:
        `Message ${i}. ` +
        'This is a realistic chat turn body with enough text to represent a genuine message. '.repeat(
          3 + (i % 5)
        ),
      ts: `2026-07-10T18:${String(28 + (i % 30)).padStart(2, '0')}:00Z`
    }))
  }
}

function fmt(ms) {
  return `${ms.toFixed(3)}ms`
}

function main() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'startup-cache-spike-'))
  const files = {
    'sessions.json': makeSessions(),
    'status.json': makeStatus(),
    'transcript-active.json': makeTranscript()
  }

  // Write the cache files once (this is the runtime write path, not boot).
  let totalBytes = 0
  for (const [name, obj] of Object.entries(files)) {
    const json = JSON.stringify(obj)
    totalBytes += Buffer.byteLength(json)
    fs.writeFileSync(path.join(dir, name), json)
  }

  const ITERS = 200
  // Warm the FS cache once (cold-launch on macOS: the files were just written,
  // so they're in the page cache — this matches a real relaunch where the
  // previous run wrote them seconds ago).
  for (const name of Object.keys(files)) fs.readFileSync(path.join(dir, name), 'utf8')

  const readParse = []
  for (let it = 0; it < ITERS; it++) {
    const t0 = performance.now()
    let hydrated = 0
    for (const name of Object.keys(files)) {
      const raw = fs.readFileSync(path.join(dir, name), 'utf8')
      const obj = JSON.parse(raw)
      // Touch the parsed data the way hydration would (count rows/sessions).
      hydrated += (obj.sessions?.length || 0) + (obj.rows?.length || 0) + 1
    }
    const t1 = performance.now()
    readParse.push(t1 - t0)
    if (hydrated < 1) throw new Error('sanity: nothing hydrated')
  }

  readParse.sort((a, b) => a - b)
  const median = readParse[Math.floor(readParse.length / 2)]
  const p95 = readParse[Math.floor(readParse.length * 0.95)]
  const max = readParse[readParse.length - 1]

  fs.rmSync(dir, { recursive: true, force: true })

  const BUDGET_MS = 90 // ~90ms left under the 1.5s target after 1.41s window-visible
  console.log('=== AC1 feasibility spike — cache read+parse cost (main-process half) ===')
  console.log(`payload: 3 files, ${(totalBytes / 1024).toFixed(1)} KB total (48-row list + 200-row transcript)`)
  console.log(`iterations: ${ITERS} (FS-cache warm, mirrors a relaunch)`)
  console.log(`read+parse+touch: median=${fmt(median)}  p95=${fmt(p95)}  max=${fmt(max)}`)
  console.log(`budget (post-1.41s window): ~${BUDGET_MS}ms`)
  const verdict = p95 < BUDGET_MS * 0.5 // leave >=half the budget for IPC+render
  console.log(
    verdict
      ? `VERDICT: PASS — file half is p95 ${fmt(p95)} << ${BUDGET_MS}ms; ample budget remains for IPC+React render (measured live in Phase 2).`
      : `VERDICT: TIGHT — file half p95 ${fmt(p95)} consumes >=50% of the ${BUDGET_MS}ms budget; AC1 <1.5s at risk, measure IPC+render before committing.`
  )
  console.log(`SPIKE-EXIT=${verdict ? 0 : 2}`)
  process.exit(verdict ? 0 : 2)
}

main()
