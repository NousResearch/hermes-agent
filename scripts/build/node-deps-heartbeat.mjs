// The checkout lock's heartbeat, a worker thread of the run that holds it (node-deps-lock.mjs):
// refreshes the lock until it is released, or until the lock is gone or holds another run's token
// (broken while this process was stopped). Any other error (an antivirus or indexer handle on
// Windows, ESTALE on NFS, EMFILE) is retried on the next beat: stopping would let a waiter break
// a live lock. flags[0]: released.
import { readFileSync, utimesSync } from 'node:fs'
import { workerData } from 'node:worker_threads'

const { lock, token, state, beat } = workerData
const flags = new Int32Array(state)
const timer = setInterval(() => {
  if (Atomics.load(flags, 0)) return clearInterval(timer)
  try {
    if (!readFileSync(lock, 'utf8').includes(token)) return clearInterval(timer)
    const now = new Date()
    utimesSync(lock, now, now)
  } catch (error) {
    if (error.code === 'ENOENT') clearInterval(timer)
  }
}, beat)
