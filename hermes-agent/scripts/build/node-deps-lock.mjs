// The checkout lock of scripts/build/node-deps.mjs and the supervision of the children that write
// the trees: one run at a time per checkout, and a child that stops writing once its run has lost
// the lock. node-deps.mjs owns what happens to the trees; this module owns who may touch them.
import { execFileSync, spawnSync } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { closeSync, mkdirSync, openSync, readFileSync, renameSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs'
import { hostname } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Worker } from 'node:worker_threads'

// One run at a time per checkout. The callers hold different locks or none (`hermes update`,
// `hermes --tui`, the dashboard, Desktop, another profile), and a run moves trees in and out of
// place, so even a reuse check must not run beside an install. The owner refreshes the lock's mtime
// from a worker thread (node-deps-heartbeat.mjs; the main thread blocks in pnpm), so liveness never
// depends on pids, hosts or PID namespaces: a lock a waiter has seen unchanged (content and mtime)
// for `stale` ms of its own monotonic clock is dead and broken; no clock is compared across
// machines. A live owner is waited for at most `wait` ms; then the run fails, and a launch can fall
// back to its previous build. The children that write the trees (pnpm, Electron's installer) run
// under a watcher that re-reads the lock every `watch` ms and kills them once it is no longer this
// run's; a run that broke a lock waits `grace` ms (longer than `watch`) before it touches the
// trees, so the watcher of the run it broke kills that run's pnpm first. Exported for tests only.
export const lockTiming = { beat: 1_000, stale: 30_000, wait: 10 * 60_000, watch: 250, grace: 2_000 }
const sleep = ms => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms)

function readText(path) {
  try {
    return readFileSync(path, 'utf8')
  } catch (error) {
    if (error.code === 'ENOENT') return undefined
    throw error
  }
}

function mtimeOf(path) {
  try {
    return statSync(path).mtimeMs
  } catch (error) {
    if (error.code === 'ENOENT') return undefined
    throw error
  }
}

// Every removal of the lock (a release, or breaking a dead one) runs under a guard held for a
// few syscalls and checks the content it removes under it, so it can never remove a lock another
// run has just created; creating needs no guard (an exclusive create cannot replace a lock). A
// guard left by a killed remover is judged like the lock, never by its mtime against this clock (a
// network filesystem's server clock, or a clock stepped back, can date it in the future): one this
// process has seen unchanged for twice the stale limit is removed. The wait is bounded too.
function underGuard(lock, fn) {
  const guard = `${lock}.guard`
  const started = performance.now()
  let seen
  for (;;) {
    let fd
    try {
      fd = openSync(guard, 'wx')
    } catch (error) {
      if (error.code !== 'EEXIST') throw error
      const now = performance.now()
      const id = identityOf(guard)
      if (id === undefined || seen?.id !== id) {
        seen = { id, since: now }
      } else if (now - seen.since > 2 * lockTiming.stale) {
        if (identityOf(guard) === id) rmSync(guard, { force: true })
        seen = undefined
      }
      if (now - started > lockTiming.wait) throw new Error(`node-deps: ${guard} has been held for too long; delete it if no other install is running`)
      sleep(20)
      continue
    }
    try {
      return fn()
    } finally {
      closeSync(fd)
      rmSync(guard, { force: true })
    }
  }
}

function identityOf(path) {
  try {
    const { ino, mtimeMs } = statSync(path)
    return `${ino}:${mtimeMs}`
  } catch (error) {
    if (error.code === 'ENOENT') return undefined
    throw error
  }
}

const removeIf = (lock, holds) => underGuard(lock, () => {
  const text = readText(lock)
  if (text !== undefined && holds(text)) rmSync(lock, { force: true })
})

function describeOwner(text) {
  try {
    const { pid, host } = JSON.parse(text)
    return `pid ${pid} on ${host}`
  } catch {
    return 'another run'
  }
}

// A run that lost its lock (stopped, e.g. by Ctrl-Z, past lockTiming.stale, so another run broke
// it) must not touch the trees: the run that holds it now may be moving them. `check()` reads the
// lock itself, never the heartbeat's view, which lags a resumed process by up to a beat.
class LockLost extends Error {
  constructor(lock, cause) {
    super(cause
      ? `node-deps: cannot read the checkout lock, so this run stops without touching the trees; run again (${lock}: ${cause.code ?? cause.message})`
      : `node-deps: this run lost the checkout lock (it was stopped for too long) and another run took over; run again (${lock})`)
    this.code = 'ENODEPSLOCK'
  }
}

// `replacing`: over a dead lock (breakLock), by rename.
function takeLock(lock, token, replacing = false) {
  const flags = new Int32Array(new SharedArrayBuffer(4))
  const content = JSON.stringify({ token, pid: process.pid, host: hostname() })
  if (replacing) {
    const temp = `${lock}.${token}`
    writeFileSync(temp, content)
    try {
      renameSync(temp, lock)
    } catch (error) {
      rmSync(temp, { force: true })
      throw error
    }
  } else {
    const fd = openSync(lock, 'wx')
    try {
      writeSync(fd, content)
    } catch (error) {
      rmSync(lock, { force: true })
      throw error
    } finally {
      closeSync(fd)
    }
  }
  const worker = new Worker(new URL('./node-deps-heartbeat.mjs', import.meta.url), { workerData: { lock, token, state: flags.buffer, beat: lockTiming.beat } })
  worker.unref()
  const check = () => {
    // A transient read error is retried; one that persists means ownership is unknown, so the
    // run touches nothing.
    for (let tries = 1; ; tries++) {
      try {
        if (readText(lock)?.includes(token)) return
        break
      } catch (error) {
        if (tries === 3) throw new LockLost(lock, error)
        sleep(50)
      }
    }
    throw new LockLost(lock)
  }
  return {
    check,
    // Runs a child that writes the trees, killed once this run loses the lock.
    run: (file, args, options) => {
      try {
        runWatched({ lock, token }, file, args, options)
      } catch (error) {
        check()
        throw error
      }
      check()
    },
    release: () => {
      Atomics.store(flags, 0, 1)
      void worker.terminate()
      removeIf(lock, text => text.includes(token))
    },
  }
}

export function lockCheckout(source) {
  const lock = join(source, '.cache/node-deps.lock')
  const token = randomUUID()
  const started = performance.now()
  let seen
  let announced = false
  for (;;) {
    try {
      mkdirSync(dirname(lock), { recursive: true })
      return takeLock(lock, token)
    } catch (error) {
      // No run can write a checkout this process cannot write, so a reader needs no lock.
      if (['EACCES', 'EPERM', 'EROFS'].includes(error.code)) {
        return { check: () => {}, run: (file, args, options) => execFileSync(file, args, { ...options, stdio: 'inherit' }), release: () => {} }
      }
      if (error.code !== 'EEXIST') throw error
    }
    const text = readText(lock)
    const mtime = mtimeOf(lock)
    if (text === undefined || mtime === undefined) continue
    const now = performance.now()
    if (seen?.text !== text || seen.mtime !== mtime) {
      seen = { text, mtime, since: now }
    } else if (now - seen.since > lockTiming.stale) {
      const held = breakLock(lock, token, text, mtime)
      if (held) {
        // The run that held it may have been resumed a moment ago, its pnpm with it: its watcher
        // kills that pnpm within lockTiming.watch of seeing the lock gone.
        sleep(lockTiming.grace)
        return held
      }
      // A failed rename (false) waits like a live owner, so the deadline still ends it.
      if (held === undefined) {
        seen = undefined
        continue
      }
    }
    // What matters comes first: a progress line cut to the terminal width still says it.
    const limit = lockTiming.wait >= 120_000 ? `${Math.round(lockTiming.wait / 60_000)} min` : `${Math.round(lockTiming.wait / 1000)} s`
    if (now - started > lockTiming.wait) {
      throw new Error(`node-deps: another install in this checkout is still running after ${limit}; run again once it finishes (${describeOwner(text)}, ${lock})`)
    }
    if (!announced) console.log(`node-deps: waiting up to ${limit} for another install in this checkout (${describeOwner(text)})`)
    announced = true
    sleep(500)
  }
}

// Replaces the dead lock (still `text` and `mtime`) with this run's, in one rename: the lock never
// disappears, so no third run can take it in between and skip the grace. undefined: it changed
// meanwhile; false: the rename failed (an open handle on Windows), so the next poll tries again.
function breakLock(lock, token, text, mtime) {
  try {
    return underGuard(lock, () => (readText(lock) === text && mtimeOf(lock) === mtime ? takeLock(lock, token, true) : undefined))
  } catch (error) {
    if (!['EPERM', 'EBUSY', 'EACCES'].includes(error.code)) throw error
    return false
  }
}

// A child that writes the trees (pnpm, Electron's installer) runs under a watcher
// (node-deps-watcher.mjs) that kills it, with everything it started, once the lock is no longer `token`'s: a run stopped past
// lockTiming.stale while pnpm runs (Ctrl-Z), then resumed, must not let that pnpm go on writing
// into the trees of the run that took over. The child stays in the caller's process group, so the
// terminal's Ctrl-C and Ctrl-Z reach it and its scripts as they always did. The watcher does not:
// on POSIX it runs in a session of its own, so the Ctrl-Z that stops the run leaves it watching,
// and it kills the stopped tree as soon as the lock is broken, before the job can be resumed.
// The tree is the child's descendants: stopped first, so none forks while it is read, then
// killed. A process that left the tree (daemonized, so re-parented away) is out of reach.
// Windows has no job control; the watcher ends the tree with taskkill /T /F.
// A relay (node-deps-relay.mjs) starts both, since this thread blocks: the watcher first, and the child only once the
// watcher has started (one that cannot start, at a process or descriptor limit, fails the run
// before anything writes the trees); the watcher gets the child's pid on stdin and ends with the
// relay. A signal to the caller's group (Ctrl-C, a caller's killpg) reaches the child itself, so
// the relay only waits; a signal aimed at this process alone kills it, and the relay, unsignalled
// and orphaned, passes SIGTERM on to the child, which used to be left running.
//
// Ctrl-C and a closed terminal signal the whole job, so the child gets them too and exits: this run
// outlives them long enough to put the previous trees back and release the lock, instead of leaving
// both for the next run to wait lockTiming.stale on. SIGTERM is left out: one aimed at this process
// alone still kills it, and the relay passes it on to the child.
export const JOB_SIGNALS = ['SIGINT', 'SIGHUP']

const RELAY = fileURLToPath(new URL('./node-deps-relay.mjs', import.meta.url))
const outlive = () => {}

function runWatched({ lock, token }, file, args, { cwd, env }) {
  for (const name of JOB_SIGNALS) process.on(name, outlive)
  let child
  try {
    child = spawnSync(process.execPath, [RELAY, lock, token, String(lockTiming.watch), file, ...args],
      { cwd, env, stdio: 'inherit' })
  } finally {
    for (const name of JOB_SIGNALS) process.off(name, outlive)
  }
  if (child.error) throw child.error
  if (child.status !== 0) {
    throw Object.assign(new Error(`Command failed: ${file} ${args.join(' ')}`), { status: child.status, signal: child.signal })
  }
}
