#!/usr/bin/env node
'use strict'
// mac-bundled-relaunch-watch.cjs — the EXTERNAL relaunch witness for the
// macOS packaged-app -> open-app-update E2E arm.
//
// Runs as a detached process owned by the DRIVER'S SHELL (never a child of
// Playwright or of the app), so its proof survives Playwright teardown and
// the app's quit-and-install. It launches nothing: it only observes.
//
// Proof this process owns (everything else in the arm defers to it):
//   1. the OLD app process appeared under <app-bin> (pid + birth recorded),
//   2. that exact pid EXITED (Squirrel.Mac replaced it),
//   3. a NEW process appeared at the same executable path with a
//      DIFFERENT pid and its own birth time — the automatic relaunch,
//      observed without any driver action starting it.
//
// Output: --out JSON file with the observation record (rewritten as state
// advances); exit 0 on a full proof, 2 on timeout, 3 on usage error.

const { execFileSync } = require('node:child_process')
const fs = require('node:fs')

function usage(message) {
  console.error(message)
  console.error('usage: node mac-bundled-relaunch-watch.cjs --app-bin <path> --out <proof.json> ' +
    '[--timeout-ms 600000] [--poll-ms 1000] [--old-pid-file <path>] [--new-pid-file <path>]')
  process.exit(3)
}

const args = {}
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i]
  if (!key.startsWith('--')) usage(`unexpected argument: ${key}`)
  args[key.slice(2)] = process.argv[i + 1]
}
if (!args['app-bin'] || !args.out) usage('--app-bin and --out are required')
const timeoutMs = Number(args['timeout-ms'] || 600_000)
const pollMs = Number(args['poll-ms'] || 1_000)
if (!Number.isFinite(timeoutMs) || !Number.isFinite(pollMs)) usage('timeouts must be numbers')

const appBin = fs.realpathSync(args['app-bin'])
const record = { appBin, startedAt: new Date().toISOString(), oldPid: null, oldBirth: null, oldSeenAt: null, oldExitedAt: null, newPid: null, newBirth: null, newSeenAt: null }

function writeRecord() {
  fs.writeFileSync(args.out, JSON.stringify(record, null, 2) + '\n')
  if (args['old-pid-file'] && record.oldPid) {
    fs.writeFileSync(args['old-pid-file'], `${record.oldPid}\n`)
  }
  if (args['new-pid-file'] && record.newPid) {
    fs.writeFileSync(args['new-pid-file'], `${record.newPid}\n`)
  }
}

// One ps pass: pid, birth time, and the executable path. On macOS `comm`
// is the real executable path; Electron helpers carry their own paths and
// never match <app-bin>, so this sees only the main app process.
function psSnapshot() {
  const out = execFileSync('ps', ['-axo', 'pid=,lstart=,comm='], { encoding: 'utf8' })
  const rows = []
  for (const line of out.split('\n')) {
    const match = /^\s*(\d+)\s+(.+?)\s+((?:\/[^/]+)+)$/.exec(line)
    if (match) rows.push({ pid: Number(match[1]), birth: match[2].trim(), comm: match[3] })
  }
  return rows
}

let oldGoneSince = 0
const startedAt = Date.now()
const timer = setInterval(() => {
  if (Date.now() - startedAt > timeoutMs) {
    clearInterval(timer)
    writeRecord()
    console.error(JSON.stringify(record))
    process.exit(2)
  }
  let rows
  try {
    rows = psSnapshot()
  } catch {
    return // transient ps failure; keep polling
  }
  if (!record.oldPid) {
    const old = rows.find(row => row.comm === appBin)
    if (old) {
      record.oldPid = old.pid
      record.oldBirth = old.birth
      record.oldSeenAt = new Date().toISOString()
      writeRecord()
    }
    return
  }
  if (!record.oldExitedAt) {
    if (rows.some(row => row.pid === record.oldPid)) {
      oldGoneSince = 0
    } else if (!oldGoneSince) {
      oldGoneSince = Date.now() // one miss may be a ps hiccup; need two
    } else if (Date.now() - oldGoneSince > pollMs) {
      record.oldExitedAt = new Date().toISOString()
      writeRecord()
    }
    return
  }
  const next = rows.find(row => row.comm === appBin && row.pid !== record.oldPid)
  if (next) {
    record.newPid = next.pid
    record.newBirth = next.birth
    record.newSeenAt = new Date().toISOString()
    writeRecord()
    console.log(JSON.stringify(record))
    process.exit(0)
  }
}, pollMs)
writeRecord()
