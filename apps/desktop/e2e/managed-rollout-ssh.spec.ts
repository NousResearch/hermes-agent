/**
 * Managed SSH host journeys over a real, disposable fixture transport.
 *
 * With no selected disposable fixture every journey reports a named skip. With
 * one selected (HERMES_MANAGED_ROLLOUT_FIXTURE_SET / _KEY / _USER) the
 * journeys below execute against a real SSH endpoint on loopback:
 *
 * - foreign-marker refusal before any drain,
 * - a real production launch that settles from its durable receipt after the
 *   wrapper acknowledgement is lost (exit-75 independent handoff),
 * - a terminal exit without a correlated receipt treated as unknown, with
 *   restart-equivalent recovery proving clearance without redispatch,
 * - a retained fence while the marker owner is still live, released only on
 *   positive clearance after the owner crashes,
 * - the marker-PID vs coordinator-readiness fence.
 *
 * The launcher programs are fixture-side stand-ins for `hermes update`; every
 * transport, observation, launch, and recovery step below is the production
 * Electron module, not a mock.
 */

import * as crypto from 'node:crypto'

import {
  createFixtureSshExec,
  selectManagedRolloutFixture,
  selectManagedRolloutSshFixtures
} from './managed-rollout-fixtures'
import { expect, test } from './test'

type SshExec = (command: string, options?: { timeoutMs?: number }) => Promise<string>

interface RemoteUpdateTarget {
  ssh: { exec: SshExec }
  platform: 'Darwin' | 'Linux' | 'Windows'
  hermesPath: string
  hermesHome: string
  pythonPath?: string
}

interface RemoteUpdateObservation {
  marker: string
  markerPid?: number
  launchIntent: string
  exitCode: number | null
  receipt: { correlationId: string; outcome: string } | null
}

interface RemoteUpdateProof {
  exitCode: number
  receipt: { correlationId: string; outcome: string }
}

interface ManagedSshUpdateModule {
  assertManagedUpdatePreflightClear: (target: RemoteUpdateTarget, correlationId: string) => Promise<void>
  executeManagedRemoteUpdate: (
    target: RemoteUpdateTarget,
    correlationId: string,
    options?: { timeoutMs?: number; pollMs?: number },
    beforeLaunchDispatch?: () => Promise<void>
  ) => Promise<RemoteUpdateProof>
  launchManagedRemoteUpdate: (target: RemoteUpdateTarget, correlationId: string) => Promise<void>
  markerIsClear: (observation: Pick<RemoteUpdateObservation, 'marker'>) => boolean
  observeManagedRemoteUpdate: (target: RemoteUpdateTarget, correlationId: string) => Promise<RemoteUpdateObservation>
  recoverManagedSshScopes: <TScope>(deps: {
    afterClearance?: () => Promise<void>
    awaitClearance: () => Promise<void>
    completeRecovery: () => Promise<void>
    restoreScope: (scope: TScope) => Promise<unknown>
    scopes: TScope[]
  }) => Promise<PromiseSettledResult<unknown>[]>
  waitForManagedRemoteClearance: (
    target: RemoteUpdateTarget,
    correlationId: string,
    options?: { timeoutMs?: number; pollMs?: number; requireTerminal?: boolean }
  ) => Promise<void>
  waitForManagedRemoteUpdate: (
    target: RemoteUpdateTarget,
    correlationId: string,
    options?: { timeoutMs?: number; pollMs?: number }
  ) => Promise<RemoteUpdateProof>
}

const selection = selectManagedRolloutSshFixtures(process.env)

const endpoint = selection.state === 'configured' ? selection.endpoints[0] : null
const fixtureExec: SshExec | null =
  selection.state === 'configured' && endpoint
    ? createFixtureSshExec({ ...endpoint, user: selection.user, keyPath: selection.keyPath })
    : null

const ROOT = '/tmp/hermes-managed-ssh-journeys'
const HOME = `${ROOT}/home`
const LAUNCHERS = `${ROOT}/launchers`

// Loaded dynamically so this spec stays inside the e2e project boundary; the
// production Electron module is the real implementation under test, not a mock.
let managedSsh: ManagedSshUpdateModule
let target: RemoteUpdateTarget | null = null

function requireFixture(): { exec: SshExec; target: RemoteUpdateTarget } {
  if (!fixtureExec || !target) {
    throw new Error(
      `disposable fixture unavailable: ${selection.state === 'refused' ? selection.reason : 'not configured'}`
    )
  }

  return { exec: fixtureExec, target }
}

async function shell(command: string, timeoutMs = 30_000): Promise<string> {
  const { exec } = requireFixture()

  return exec(command, { timeoutMs })
}

async function writeRemoteFile(path: string, content: string): Promise<void> {
  const encoded = Buffer.from(content, 'utf8').toString('base64')

  await shell(`printf %s '${encoded}' | base64 -d > '${path}'`)
}

async function resetRemoteState(): Promise<void> {
  await shell(
    `rm -f '${HOME}/.hermes-update-in-progress' '${HOME}/.update_exit_code.'* '${HOME}/.update_launch_intent.'* ` +
      `'${HOME}/.update_coordinator_ready.'* && rm -rf '${HOME}/logs' && mkdir -p '${HOME}/logs'`
  )
}

function spawnRemoteProcess(command: string): Promise<number> {
  return shell(`nohup ${command} >/dev/null 2>&1 & echo $!`).then(output => {
    const pid = Number.parseInt(output.trim().split('\n').at(-1) || '', 10)

    expect(Number.isSafeInteger(pid) && pid > 0, `remote process id for ${command}`).toBe(true)

    return pid
  })
}

async function waitForRemoteExit(pid: number): Promise<void> {
  await shell(
    `i=0; while kill -0 ${pid} 2>/dev/null; do i=$((i+1)); [ "$i" -ge 80 ] && exit 1; sleep 0.25; done; exit 0`
  )
}

async function writeMarker(pid: number): Promise<void> {
  await shell(`printf '%s\\n%s\\n' '${pid}' 1 > '${HOME}/.hermes-update-in-progress'`)
}

function correlationId(): string {
  return crypto.randomUUID()
}

async function readMarkerPid(): Promise<number> {
  const output = await shell(`head -n 1 '${HOME}/.hermes-update-in-progress'`)

  return Number.parseInt(output.trim(), 10)
}

test.describe.configure({ mode: 'serial' })

test.beforeAll(async () => {
  const modulePath = ['..', 'electron', 'managed-ssh-update.ts'].join('/')

  managedSsh = (await import(modulePath)) as ManagedSshUpdateModule

  if (selection.state !== 'configured' || !endpoint || !fixtureExec) {
    return
  }

  await fixtureExec(`mkdir -p '${HOME}/logs' '${LAUNCHERS}'`)
  target = {
    ssh: { exec: fixtureExec },
    platform: 'Linux',
    hermesPath: `${LAUNCHERS}/ok-handoff.sh`,
    hermesHome: HOME
  }

  // Independent-handoff launcher: publishes a correlated receipt plus the
  // coordinator readiness proof, releases the marker, and exits 75 so the
  // wrapper deliberately does not write a wrapper status file.
  await writeRemoteFile(
    `${LAUNCHERS}/ok-handoff.sh`,
    [
      '#!/bin/sh',
      'set -e',
      'corr="$HERMES_UPDATE_CORRELATION_ID"',
      'home="$HERMES_HOME"',
      'mkdir -p "$home/logs/update_receipts"',
      `printf '%s\\n%s\\n' "$$" 1 > "$home/.hermes-update-in-progress"`,
      'now=$(date -u +%Y-%m-%dT%H:%M:%SZ)',
      'printf \'{"correlation_id":"%s","outcome":"success","started_at":"%s","finished_at":"%s",' +
        '"pre_update":{"sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},' +
        '"post_update":{"sha":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}\' "$corr" "$now" "$now" ' +
        '> "$home/logs/update_receipts/update_$$.json"',
      `printf '{"correlation_id":"%s","pid":%s}' "$corr" "$$" > "$home/.update_coordinator_ready.$corr"`,
      'rm -f "$home/.hermes-update-in-progress"',
      'exit 75'
    ].join('\n') + '\n'
  )

  // Lost-acknowledgement launcher: claims the marker, writes nothing else,
  // and exits 0 — the wrapper records a bare terminal exit with no receipt.
  await writeRemoteFile(
    `${LAUNCHERS}/lost-ack.sh`,
    ['#!/bin/sh', `printf '%s\\n%s\\n' "$$" 1 > "$HERMES_HOME/.hermes-update-in-progress"`, 'exit 0'].join('\n') + '\n'
  )

  // Wedged launcher: claims the marker and stays alive until it is killed.
  // The busy loop keeps the shell itself as the marker owner — a trailing
  // `sleep` could be exec-optimized and replace the recorded PID.
  await writeRemoteFile(
    `${LAUNCHERS}/wedged.sh`,
    [
      '#!/bin/sh',
      `printf '%s\\n%s\\n' "$$" 1 > "$HERMES_HOME/.hermes-update-in-progress"`,
      'while true; do sleep 1; done'
    ].join('\n') + '\n'
  )

  await fixtureExec(`chmod +x '${LAUNCHERS}'/*.sh`)
  await resetRemoteState()
})

test.beforeEach(async () => {
  if (target) {
    await resetRemoteState()
  }
})

test('refuses non-test targets and missing credentials before any connection', () => {
  expect(selectManagedRolloutFixture({ HERMES_MANAGED_ROLLOUT_E2E_TARGET: 'ssh://production.example' }).state).toBe(
    'refused'
  )
  expect(
    selectManagedRolloutFixture({
      HERMES_MANAGED_ROLLOUT_E2E_TARGET: 'test://hermes-managed-rollout/two-hosts',
      HERMES_MANAGED_ROLLOUT_E2E_CREDENTIAL_REF: '-----BEGIN PRIVATE KEY-----'
    }).state
  ).toBe('refused')
  expect(
    selectManagedRolloutFixture({
      HERMES_MANAGED_ROLLOUT_E2E_TARGET: 'test://hermes-managed-rollout/two-hosts'
    }).state
  ).toBe('refused')

  const key = 'C:/Users/example/.hermes-fixture/fixture_key'

  expect(
    selectManagedRolloutSshFixtures({
      HERMES_MANAGED_ROLLOUT_FIXTURE_SET: 'ssh.example.com:22',
      HERMES_MANAGED_ROLLOUT_FIXTURE_KEY: key
    })
  ).toEqual({
    state: 'refused',
    reason: 'refusing non-disposable SSH fixture host in HERMES_MANAGED_ROLLOUT_FIXTURE_SET'
  })

  expect(selectManagedRolloutSshFixtures({ HERMES_MANAGED_ROLLOUT_FIXTURE_SET: '127.0.0.1:2222' })).toEqual({
    state: 'refused',
    reason: 'HERMES_MANAGED_ROLLOUT_FIXTURE_KEY is unset; no disposable fixture key is selected'
  })

  expect(
    selectManagedRolloutSshFixtures({
      HERMES_MANAGED_ROLLOUT_FIXTURE_SET: '127.0.0.1:2222,127.0.0.1:2222',
      HERMES_MANAGED_ROLLOUT_FIXTURE_KEY: key
    }).state
  ).toBe('refused')
})

test('refuses a foreign live update marker before draining anything', async () => {
  if (selection.state !== 'configured') {
    test.skip(true, selection.reason)

    return
  }

  const { target: targetValue } = requireFixture()
  const owner = await spawnRemoteProcess('sleep 300')

  try {
    await writeMarker(owner)

    const observation = await managedSsh.observeManagedRemoteUpdate(targetValue, correlationId())

    expect(observation.marker).toBe('live')
    expect(observation.markerPid).toBe(owner)
    expect(managedSsh.markerIsClear(observation)).toBe(false)

    await expect(managedSsh.assertManagedUpdatePreflightClear(targetValue, correlationId())).rejects.toThrow(
      /refusing to drain its serves/
    )
  } finally {
    await shell(`kill -TERM ${owner} 2>/dev/null; exit 0`)
    await waitForRemoteExit(owner)
  }

  // A dead owner is clear: the same journey proceeds without any drain refusal.
  await expect(managedSsh.assertManagedUpdatePreflightClear(requireFixture().target, correlationId())).resolves.toBeUndefined()
})

test('settles a lost wrapper acknowledgement from the durable receipt', async () => {
  if (selection.state !== 'configured') {
    test.skip(true, selection.reason)

    return
  }

  const { target: targetValue } = requireFixture()
  const correlation = correlationId()
  const order: string[] = []

  const proof = await managedSsh.executeManagedRemoteUpdate(
    targetValue,
    correlation,
    { timeoutMs: 60_000, pollMs: 250 },
    async () => {
      order.push('before-launch-dispatch')
    }
  )

  expect(order).toEqual(['before-launch-dispatch'])
  expect(proof.exitCode).toBe(0)
  expect(proof.receipt.correlationId).toBe(correlation)
  expect(proof.receipt.outcome).toBe('success')

  // The wrapper status was intentionally suppressed by the exit-75 handoff…
  const status = await shell(
    `if [ -e '${HOME}/.update_exit_code.${correlation}' ]; then echo present; else echo absent; fi`
  )

  expect(status.trim()).toBe('absent')

  // …and the terminal truth came from the correlated receipt plus a released marker.
  const observation = await managedSsh.observeManagedRemoteUpdate(targetValue, correlation)

  expect(observation.marker).toBe('absent')
  expect(observation.receipt?.correlationId).toBe(correlation)
})

test('treats a terminal exit with no correlated receipt as unknown, then proves clearance without redispatch', async () => {
  if (selection.state !== 'configured') {
    test.skip(true, selection.reason)

    return
  }

  const { target: targetValue } = requireFixture()
  const correlation = correlationId()

  targetValue.hermesPath = `${LAUNCHERS}/lost-ack.sh`
  await managedSsh.launchManagedRemoteUpdate(targetValue, correlation)

  await expect(
    managedSsh.waitForManagedRemoteUpdate(targetValue, correlation, { timeoutMs: 30_000, pollMs: 250 })
  ).rejects.toThrow(/without a correlated durable receipt/)

  // The unknown attempt must not be redispatched: no receipt exists and the
  // bare terminal exit is all the host recorded.
  const receipts = await shell(`ls '${HOME}/logs/update_receipts' 2>/dev/null | wc -l`)

  expect(receipts.trim()).toBe('0')

  // Restart-equivalent recovery: clearance is proven from the dead launch
  // intent, then the recorded scopes are restored exactly once.
  const restores: string[] = []
  const completions: string[] = []

  const results = await managedSsh.recoverManagedSshScopes({
    awaitClearance: () =>
      managedSsh.waitForManagedRemoteClearance(targetValue, correlation, {
        timeoutMs: 30_000,
        pollMs: 250,
        requireTerminal: true
      }),
    completeRecovery: async () => {
      completions.push(correlation)
    },
    restoreScope: async (scope: string) => {
      restores.push(scope)
    },
    scopes: [`scope-${correlation}`]
  })

  expect(results).toHaveLength(1)
  expect(restores).toEqual([`scope-${correlation}`])
  expect(completions).toEqual([correlation])
})

test('keeps the recovery fence while the marker owner lives and releases it only on proof', async () => {
  if (selection.state !== 'configured') {
    test.skip(true, selection.reason)

    return
  }

  const { target: targetValue } = requireFixture()
  const correlation = correlationId()

  targetValue.hermesPath = `${LAUNCHERS}/wedged.sh`
  await managedSsh.launchManagedRemoteUpdate(targetValue, correlation)

  const owner = await readMarkerPid()
  const restores: string[] = []
  const completions: string[] = []

  // The owner is still alive: clearance cannot be proven, so the fence holds
  // and no scope is restored while the update may still be mutating the host.
  await expect(
    managedSsh.recoverManagedSshScopes({
      awaitClearance: () =>
        managedSsh.waitForManagedRemoteClearance(targetValue, correlation, {
          timeoutMs: 2_500,
          pollMs: 250,
          requireTerminal: true
        }),
      completeRecovery: async () => {
        completions.push(correlation)
      },
      restoreScope: async (scope: string) => {
        restores.push(scope)
      },
      scopes: [`scope-${correlation}`]
    })
  ).rejects.toThrow(/Could not prove remote update clearance/)

  expect(restores).toEqual([])
  expect(completions).toEqual([])

  // Crash the wedged owner. The marker PID dies with it, and the recorded
  // launch intent follows, so positive clearance becomes provable.
  await shell(`kill -KILL ${owner} 2>/dev/null; exit 0`)
  await waitForRemoteExit(owner)

  const results = await managedSsh.recoverManagedSshScopes({
    awaitClearance: () =>
      managedSsh.waitForManagedRemoteClearance(targetValue, correlation, {
        timeoutMs: 30_000,
        pollMs: 250,
        requireTerminal: true
      }),
    completeRecovery: async () => {
      completions.push(correlation)
    },
    restoreScope: async (scope: string) => {
      restores.push(scope)
    },
    scopes: [`scope-${correlation}`]
  })

  expect(results.every(entry => entry.status === 'fulfilled')).toBe(true)
  expect(restores).toEqual([`scope-${correlation}`])
  expect(completions).toEqual([correlation])

  // No redispatch: nothing wrote a fresh receipt for this correlation.
  const receipts = await shell(`ls '${HOME}/logs/update_receipts' 2>/dev/null | wc -l`)

  expect(receipts.trim()).toBe('0')
})

test('rejects a marker whose owner PID disagrees with the coordinator readiness proof', async () => {
  if (selection.state !== 'configured') {
    test.skip(true, selection.reason)

    return
  }

  const { target: targetValue } = requireFixture()
  const correlation = correlationId()
  const coordinator = await spawnRemoteProcess('sleep 300')

  try {
    await writeMarker(1)
    await shell(
      `printf '{"correlation_id":"%s","pid":%s}' '${correlation}' '${coordinator}' > '${HOME}/.update_coordinator_ready.${correlation}'`
    )

    await expect(
      managedSsh.waitForManagedRemoteUpdate(targetValue, correlation, { timeoutMs: 5_000, pollMs: 250 })
    ).rejects.toThrow(/did not match coordinator PID/)
  } finally {
    await shell(`kill -TERM ${coordinator} 2>/dev/null; exit 0`)
    await waitForRemoteExit(coordinator)
  }
})
