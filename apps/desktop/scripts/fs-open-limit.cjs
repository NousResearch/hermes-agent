// Bound how many fs.open calls are in flight at once, process-wide.
//
// WHY THIS EXISTS
//
// @electron/osx-sign decides what to sign by walking the whole .app
// (dist/util.js::walk). The walk recurses with
// `Promise.all(children.map(...))` and no concurrency bound, and every
// regular file it reaches is probed by isbinaryfile, which opens a
// descriptor. Peak descriptors therefore track the size of the tree, not
// any fixed budget.
//
// The bundled payload makes that fatal. site-packages/lark_oapi alone is
// 11,112 files, so signing died with:
//
//   EMFILE: too many open files, open
//   '.../agent-payload/site-packages/lark_oapi/api/im/v1/model/reaction.py'
//
// Raising `ulimit -n` also clears it, but only moves the ceiling: the
// walk still asks for as many descriptors as there are files, so the
// next payload growth hits the new limit. This caps the demand instead.
//
// WHAT IT PATCHES
//
// `fs.open` (callback form) and `fs.promises.open`. That covers the
// paths that actually consume descriptors here:
//   - isbinaryfile captures `promisify(fs.open)` at module load, so the
//     patch must be installed BEFORE it is required — hence --require.
//   - fs.createReadStream routes through the public fs.open (verified).
//
// Deliberately NOT patched: openSync. A synchronous open cannot be
// queued without blocking the loop that would drain the queue, and it is
// self-limiting anyway — one at a time, by definition.
//
// WHY IT CANNOT DEADLOCK
//
// The slot is released when the open() call SETTLES, not when the
// descriptor is closed. A task holding ten descriptors occupies zero
// slots while it works, so "hold one, open another" always makes
// progress. Gating the descriptor's lifetime instead does deadlock —
// measured: 4 tasks each holding 1 fd and awaiting a second, under
// cap=2, never completes.
//
// Failures propagate unchanged: a rejected open releases its slot in
// `finally` and the original error reaches the caller. This changes
// timing only, never behavior.
//
// HERMES_FS_OPEN_LIMIT overrides the computed cap; 0 disables the patch.

'use strict'

const fs = require('fs')

// The cap must sit well BELOW the process's fd limit, not near it. Each
// queued open is held across the caller's read+close (osx-sign's walk
// does exactly that), so `LIMIT` descriptors can be live at once, on top
// of the fds node already holds for stdio, the module loader, sockets
// and spawned children. Measured against the real walk: cap 100 fails at
// a 64 limit and passes at 256; cap 16 passes at 64.
//
// So: a quarter of the soft limit, clamped to a sane band. On the macOS
// default of 256 that is 64; on a raised limit it grows, keeping the
// walk fast; on a tiny limit it shrinks instead of guaranteeing EMFILE.
function defaultLimit() {
  const override = Number(process.env.HERMES_FS_OPEN_LIMIT)
  if (Number.isFinite(override) && override >= 0) return override
  let soft = 0
  try {
    soft = process.report.getReport().userLimits.open_files.soft
  } catch {
    soft = 0
  }
  // `unlimited` reports as -1 (or a huge number); either way the clamp holds.
  if (!Number.isFinite(soft) || soft <= 0) soft = 256
  return Math.max(8, Math.min(512, Math.floor(soft / 4)))
}

const LIMIT = defaultLimit()

// A guard against double-install (nested spawns inheriting NODE_OPTIONS).
if (!globalThis.__hermesFsOpenLimited && LIMIT > 0) {
  globalThis.__hermesFsOpenLimited = true

  let active = 0
  /** @type {Array<() => void>} */
  const queue = []

  const pump = () => {
    while (active < LIMIT && queue.length > 0) {
      active += 1
      queue.shift()()
    }
  }

  const release = () => {
    active -= 1
    pump()
  }

  // ── fs.open (callback form) ───────────────────────────────────────
  const realOpen = fs.open
  fs.open = function open(...args) {
    const cb = args[args.length - 1]
    if (typeof cb !== 'function') {
      // No callback: not the async form we can queue (and node throws on
      // it anyway). Hand it straight to the real implementation.
      return realOpen.apply(this, args)
    }
    const rest = args.slice(0, -1)
    queue.push(() => {
      realOpen.call(fs, ...rest, function (...cbArgs) {
        // Release BEFORE the callback runs: the descriptor is open, this
        // call is done, and the caller may synchronously open another.
        release()
        cb.apply(this, cbArgs)
      })
    })
    pump()
    return undefined
  }
  // promisify(fs.open) reads these off the function it is handed.
  Object.defineProperty(fs.open, 'name', { value: 'open' })
  if (realOpen[Symbol.for('nodejs.util.promisify.custom')]) {
    fs.open[Symbol.for('nodejs.util.promisify.custom')] =
      realOpen[Symbol.for('nodejs.util.promisify.custom')]
  }

  // ── fs.promises.open ──────────────────────────────────────────────
  const realPromisesOpen = fs.promises.open
  fs.promises.open = function open(...args) {
    return new Promise((resolve, reject) => {
      queue.push(() => {
        realPromisesOpen
          .apply(fs.promises, args)
          .then(resolve, reject)
          .finally(release)
      })
      pump()
    })
  }

  if (process.env.HERMES_FS_OPEN_LIMIT_DEBUG) {
    console.log(`[fs-open-limit] fs.open concurrency capped at ${LIMIT}`)
  }
}
