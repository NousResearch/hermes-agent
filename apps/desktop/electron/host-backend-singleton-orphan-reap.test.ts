// #81275 regression coverage: the duplicate-serve/orphan bug class (one
// backend per host across reconnect storms; no PPID-1 orphans after a quit).
//
// The causal fixes are on main (dial-claim coalescing #90812, host-backend
// singleton 58eb248d080, ownership-ledger orphan reaping #89298/#87295). What
// the canonical issue still lacked was a test pinning the WHOLE contract at
// the seams the report actually exercised: N concurrent dials for the same
// scope collapse to one spawn (reconnect storm), the spawn-guard makes a
// second local backend unreachable even when a dial bypasses routing, and a
// crashed "Electron" leaves a child that a later launch's reap sweep stops
// (the PPID-1 orphan). Real child processes + the real platform start-marker
// probe; Electron itself is faked only where it cannot run under vitest.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'

import { test } from 'vitest'

import { processStartMarker } from './backend-claim'
import { BackendDialClaims } from './backend-dial-claim'
import {
  type BackendIdentity,
  type BackendOwnershipEntry,
  createBackendOwnership
} from './backend-ownership'
import { assertNoSecondLocalBackend, SecondLocalBackendError } from './host-backend-singleton'
import { isPidAlive } from './update-marker'

// A real child that outlives its "parent" — the orphan the report's PPID-1
// processes came from. `node -e` children are cheap, real, and portable.
function spawnDetachedChild(): { pid: number; stop: () => void } {
  const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 60000)'], {
    stdio: 'ignore'
  })

  if (!child.pid) {
    throw new Error('fixture child never got a pid')
  }

  const pid = child.pid

  return {
    pid,
    stop: () => {
      try {
        child.kill('SIGKILL')
      } catch {
        // Already gone.
      }
    }
  }
}

test('#81275 storm: 12 concurrent same-scope reconnect dials coalesce to ONE spawn', async () => {
  const claims = new BackendDialClaims()
  let spawns = 0

  const dial = async () => {
    spawns += 1
    await new Promise(resolve => setTimeout(resolve, 20)) // simulate a slow backend boot

    return { baseUrl: 'http://127.0.0.1:53150', pid: 4242 }
  }

  // The reconnect storm from the report: N renderer reconnect ticks racing
  // one wake. backendDialClaims is the seam main.ts wires at every dial site
  // (ensureTerminalBackend, media-stream resolveRemoteConnection, window
  // routes), so this is the production coalescing contract, not a model.
  const results = await Promise.all(
    Array.from({ length: 12 }, (_, attempt) => claims.run('default', dial))
  )

  assert.equal(spawns, 1, 'a same-scope reconnect storm must produce exactly one backend spawn')
  assert.equal(claims.inFlight('default'), false, 'the claim must release once the dial settles')

  for (const result of results) {
    assert.equal(result, results[0], 'every coalesced dial must receive the same connection')
  }
})

test('#81275 storm: distinct scopes dial independently, so coalescing never serializes profiles', async () => {
  const claims = new BackendDialClaims()
  let firstDialPending = true

  const dial = async (baseUrl: string) => {
    await new Promise(resolve => setTimeout(resolve, 30))

    return { baseUrl }
  }

  const slow = claims.run('conn:office-ssh::default', () => {
    firstDialPending = false

    return dial('http://office:1')
  })

  await new Promise(resolve => setTimeout(resolve, 10))
  assert.equal(firstDialPending, false, 'the first dial starts eagerly')

  const fast = claims.run('conn:office-ssh::work', () => dial('http://office:2'))
  const local = claims.run('default', () => dial('http://127.0.0.1:3'))

  const [a, b, c] = await Promise.all([slow, fast, local])

  assert.deepEqual(
    [a.baseUrl, b.baseUrl, c.baseUrl],
    ['http://office:1', 'http://office:2', 'http://127.0.0.1:3'],
    'each (connectionId, profile) scope keeps its own dial'
  )
})

test('#81275 singleton: a local pool spawn that routing missed still cannot reach the OS', () => {
  // The report's "respawns duplicate serve backends" happened because several
  // spawn paths each started a child. assertNoSecondLocalBackend sits at the
  // one local spawn site (runPoolBackendStart) so ANY caller that reaches it
  // through a path routing does not cover fails loudly instead of quietly
  // reintroducing a process per profile.
  assert.throws(() => assertNoSecondLocalBackend('worker'), SecondLocalBackendError)
  assert.throws(() => assertNoSecondLocalBackend('worker', { isolated: false }), SecondLocalBackendError)

  // The documented escape hatches still reach a (deliberate) private backend.
  assertNoSecondLocalBackend('worker', { isolated: true })
  assertNoSecondLocalBackend('worker', { profileRemoteOverride: true })
  assertNoSecondLocalBackend('worker', { primaryRemoteActive: true })
  assertNoSecondLocalBackend('worker', { unscopableRequest: true })
})

test('#81275 orphan reap: a crashed parent leaves a REAL child that the next launch reaps', { timeout: 60_000 }, async () => {
  // The issue's orphans were PPID-1 serve children left by a Desktop that
  // quit before its backends. Simulate exactly the recorded shape: the
  // ownership ledger knows the child (pid + start marker) and its parent;
  // the parent is dead; the next launch's reapOrphans must stop the child.
  // Real processes + the real platform start-marker probe (ps / /proc), so
  // this exercises the same identity matching the packaged reaper uses.
  if (process.platform === 'win32') {
    return // Windows uses PowerShell Get-Process; covered by the Windows lane.
  }

  const orphan = spawnDetachedChild()
  const fakeParent = spawnDetachedChild()

  try {
    const marker = await processStartMarker(orphan.pid)
    const parentMarker = await processStartMarker(fakeParent.pid)

    const entry: BackendOwnershipEntry = {
      nonce: 'reap-probe-nonce',
      pid: orphan.pid,
      profile: 'default',
      startMarker: marker,
      parentPid: fakeParent.pid,
      parentStartMarker: parentMarker
    }

    let store = JSON.stringify({ backends: [entry] })
    const stopped: BackendIdentity[] = []

    const ownership = createBackendOwnership({
      // The REAL probe: the orphan is alive and its recorded start marker
      // matches, so identity is confirmed — the reaper must stop it.
      matchesIdentity: async identity => (await processStartMarker(identity.pid)) === identity.startMarker,
      // The parent is dead (killed below), so parent liveness resolves via
      // the same marker comparison main.ts's backendParentMatches performs.
      matchesParent: async candidate => {
        if (!candidate.parentPid || !candidate.parentStartMarker) {
          return undefined
        }

        try {
          return (await processStartMarker(candidate.parentPid)) === candidate.parentStartMarker
        } catch {
          return false // ESRCH: the parent is gone.
        }
      },
      stop: async identity => {
        try {
          process.kill(identity.pid, 'SIGTERM')
        } catch {
          // Race with an external kill: still record the attempt.
        }

        stopped.push(identity)
      },
      store: {
        read: () => store,
        write: (next: string) => {
          store = next
        }
      }
    })

    // The parent dies first — the crash that orphans the backend child.
    fakeParent.stop()
    await new Promise<void>(resolve => {
      const timer = setTimeout(resolve, 500)

      void (async () => {
        while (isPidAlive(fakeParent.pid)) {
          await new Promise(wait => setTimeout(wait, 25))
        }

        clearTimeout(timer)
        resolve()
      })()
    })
    assert.equal(isPidAlive(fakeParent.pid), false, 'fixture parent must be dead before the sweep')

    const reaped = await ownership.reapOrphans()

    assert.deepEqual(reaped, [orphan.pid], 'the sweep must reap the orphaned child')
    assert.equal(stopped.length, 1, 'exactly one stop for the confirmed orphan')

    await new Promise(resolve => setTimeout(resolve, 300))
    assert.equal(isPidAlive(orphan.pid), false, 'the orphaned backend must actually be stopped')

    assert.doesNotThrow(() => {
      const survivors = JSON.parse(store).backends as BackendOwnershipEntry[]
      assert.equal(
        survivors.some(candidate => candidate.pid === orphan.pid),
        false,
        'a reaped backend must leave the ownership ledger'
      )
    })
  } finally {
    orphan.stop()
    fakeParent.stop()
  }
})

test('#81275 orphan reap: a backend whose parent Electron is STILL alive is never reaped', { timeout: 60_000 }, async () => {
  // #87295: the running instance's live backend is never an orphan, even when
  // a second launch reaches reapOrphans. Same real-process probe, parent kept
  // alive — the mirror image of the crash case and the guard that makes the
  // reaper safe to run on every boot.
  if (process.platform === 'win32') {
    return
  }

  const owned = spawnDetachedChild()
  const liveParent = spawnDetachedChild()

  try {
    const entry: BackendOwnershipEntry = {
      nonce: 'live-parent-nonce',
      pid: owned.pid,
      profile: 'default',
      startMarker: await processStartMarker(owned.pid),
      parentPid: liveParent.pid,
      parentStartMarker: await processStartMarker(liveParent.pid)
    }

    let store = JSON.stringify({ backends: [entry] })
    let stopCalls = 0

    const ownership = createBackendOwnership({
      matchesIdentity: async identity => (await processStartMarker(identity.pid)) === identity.startMarker,
      matchesParent: async candidate => {
        if (!candidate.parentPid || !candidate.parentStartMarker) {
          return undefined
        }

        return (await processStartMarker(candidate.parentPid)) === candidate.parentStartMarker
      },
      stop: async () => {
        stopCalls += 1
      },
      store: {
        read: () => store,
        write: (next: string) => {
          store = next
        }
      }
    })

    const reaped = await ownership.reapOrphans()

    assert.deepEqual(reaped, [], 'a live parent means the backend is not an orphan')
    assert.equal(stopCalls, 0, 'the sweep must never stop a live instance\'s backend')
    assert.equal(isPidAlive(owned.pid), true, 'the owned backend survives the sweep')
  } finally {
    owned.stop()
    liveParent.stop()
  }
})
