import { spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from '@playwright/test'

import {
  assertGpuPackagedEligibility,
  buildPopulationContract,
  createLunarCityPopulationFixture,
  expectedStandardProjection,
  gpuPackagedLaunchOptions,
  GROUP_DISTRICTS,
  LEADER_FAMILIES,
  readStandardPopulation,
  startPopulationGateways
} from './lunar-city-fixtures'

for (const population of [25, 100, 250] as const) {
  test(`population ${population} has a deterministic immutable contract`, () => {
    const first = buildPopulationContract(population)
    const second = buildPopulationContract(population)

    expect(first).toEqual(second)
    expect(Object.isFrozen(first)).toBe(true)
    expect(first.population).toBe(population)
    expect(Object.values(first.entitiesByKind).reduce((sum, value) => sum + value, 0)).toBe(population)
    expect(first.activity.active + first.activity.idle + first.activity.unavailable).toBe(population)
    expect(first.lod.near + first.lod.mid + first.lod.far + first.lod.aggregate).toBe(population)
    expect(first.groups).toEqual(GROUP_DISTRICTS)
    expect(first.leaderFamilies).toEqual(LEADER_FAMILIES)
    expect(first.digest).toMatch(/^[a-f0-9]{64}$/u)
    expect(first.entities.some(row => row.kind === 'profile' && row.groups.length === 0)).toBe(true)
    expect(first.entities.some(row => row.kind === 'profile' && row.activity === 'unavailable')).toBe(true)
    expect(new Set(first.entities.flatMap(row => row.groups))).toEqual(new Set(GROUP_DISTRICTS.map(([name]) => name)))
    expect(sourceDistribution(first.entities)).toEqual(
      {
        25: { local: 11, 'remote-lab': 9, 'remote-archive': 5 },
        100: { local: 35, 'remote-lab': 34, 'remote-archive': 31 },
        250: { local: 85, 'remote-lab': 84, 'remote-archive': 81 }
      }[population]
    )
  })
}

test('fixture retains exact-source collisions and every required entity family', () => {
  const contract = buildPopulationContract(100)
  const duplicateRows = contract.entities.filter(row => row.displayId === 'shared-steward')

  expect(new Set(duplicateRows.map(row => row.connectionId))).toEqual(new Set(['local', 'remote-lab']))
  expect(new Set(duplicateRows.map(row => row.durableId)).size).toBe(1)
  expect(new Set(duplicateRows.map(row => row.exactKey)).size).toBe(2)
  expect(contract.entities.some(row => row.kind === 'profile' && row.groups.length === 0)).toBe(true)
  expect(contract.entities.some(row => row.kind === 'profile' && row.activity === 'unavailable')).toBe(true)
  expect(new Set(contract.entities.map(row => row.kind))).toEqual(
    new Set(['profile', 'session', 'subagent', 'task', 'worker'])
  )
})

test('fixture entity keys have byte parity with production EntityIdentity for every kind', async () => {
  const identityModulePath = '../src/app/lunar-city/identity'

  const { entityKey } = (await import(identityModulePath)) as {
    entityKey(identity: Record<string, unknown>): string
  }

  const contract = buildPopulationContract(100)

  for (const row of contract.entities) {
    const identity =
      row.kind === 'profile'
        ? { kind: 'profile', connectionId: row.connectionId, profile: row.profile }
        : row.kind === 'session'
          ? {
              kind: 'session',
              connectionId: row.connectionId,
              profile: row.profile,
              sessionId: row.sessionId
            }
          : row.kind === 'subagent'
            ? {
                kind: 'subagent',
                connectionId: row.connectionId,
                profile: row.profile,
                sessionId: row.sessionId,
                subagentId: row.subagentId
              }
            : {
                kind: 'kanban',
                connectionId: row.connectionId,
                profile: row.profile,
                board: row.board,
                taskId: row.taskId,
                runId: row.runId,
                ...(row.kind === 'worker' ? { workerId: row.workerId } : {})
              }

    expect(row.exactKey).toBe(entityKey(identity))
  }
})

test('worker keys and immutable digests are byte-identical across fresh processes', () => {
  const script = `
    const fixture = await import('./e2e/lunar-city-fixtures.ts')
    const contract = fixture.buildPopulationContract(250)
    process.stdout.write(JSON.stringify({
      contractDigest: contract.digest,
      projectionDigest: fixture.expectedStandardProjection(contract).digest,
      workerKeys: contract.entities.filter(row => row.kind === 'worker').map(row => row.exactKey)
    }))
  `

  const run = (): string => {
    const result = spawnSync(process.execPath, ['--import', 'tsx', '--input-type=module', '-e', script], {
      cwd: path.resolve(import.meta.dirname, '..'),
      encoding: 'utf8'
    })

    expect(result.status, result.stderr || result.stdout).toBe(0)

    return result.stdout
  }

  const first = run()
  const second = run()

  expect(first).toBe(second)
  expect(JSON.parse(first)).toMatchObject({
    contractDigest: expect.stringMatching(/^[a-f0-9]{64}$/u),
    projectionDigest: expect.stringMatching(/^[a-f0-9]{64}$/u),
    workerKeys: expect.arrayContaining([expect.stringContaining('kind:string:6:kanban')])
  })
})

test('seed uses isolated standard Hermes files, contains no secrets, and cleans up', () => {
  const fixture = createLunarCityPopulationFixture(25)

  expect(fixture.root.startsWith(path.resolve(process.env.TMPDIR || '/tmp'))).toBe(true)
  expect(fs.existsSync(path.join(fixture.hermesHome, 'config.yaml'))).toBe(true)
  expect(fs.existsSync(path.join(fixture.hermesHome, 'state.db'))).toBe(true)
  expect(fs.existsSync(path.join(fixture.hermesHome, 'kanban.db'))).toBe(true)
  expect(fs.existsSync(path.join(fixture.hermesHome, 'profiles'))).toBe(true)

  for (const [connectionId, home] of Object.entries(fixture.sourceHomes)) {
    const expectedProfiles = new Set(
      fixture.contract.entities
        .filter(row => row.kind === 'profile' && row.connectionId === connectionId)
        .map(row => row.profile)
    )

    expectedProfiles.delete('default')
    expect(new Set(fs.readdirSync(path.join(home, 'profiles')))).toEqual(expectedProfiles)
    expect(fs.existsSync(path.join(home, 'profile.yaml'))).toBe(true)
  }

  const files = walkFiles(fixture.root)

  const text = files
    .filter(file => !file.endsWith('.db') && !file.endsWith('.db-shm') && !file.endsWith('.db-wal'))
    .map(file => fs.readFileSync(file, 'utf8'))
    .join('\n')

  expect(text).not.toMatch(/(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s"']+/iu)
  expect(JSON.parse(fs.readFileSync(fixture.contractPath, 'utf8')).digest).toBe(fixture.contract.digest)

  fixture.cleanup()
  expect(fs.existsSync(fixture.root)).toBe(false)
})

for (const population of [25, 100, 250] as const) {
  test(`standard gateway routes expose exact persisted sources for population ${population}`, async () => {
    test.slow()
    const fixture = createLunarCityPopulationFixture(population)
    let gateways: Awaited<ReturnType<typeof startPopulationGateways>> | undefined

    try {
      gateways = await startPopulationGateways(fixture)
      expect(JSON.parse(fs.readFileSync(path.join(fixture.userDataDir, 'connections.json'), 'utf8')).version).toBe(2)

      const observed = { profile: 0, session: 0, task: 0, worker: 0 }

      for (const source of gateways.sources) {
        const headers = { 'X-Hermes-Session-Token': source.token }

        const unauthorized = await fetch(`${source.url}/api/profiles`, {
          headers: { 'X-Hermes-Session-Token': `${source.token}-wrong` }
        })

        expect(unauthorized.ok).toBe(false)

        const profiles = (await (await fetch(`${source.url}/api/profiles`, { headers })).json()) as {
          profiles: unknown[]
        }

        const sessions = (await (
          await fetch(`${source.url}/api/profiles/sessions?limit=500&profile=all`, { headers })
        ).json()) as { sessions: unknown[]; total: number }

        const board = (await (
          await fetch(`${source.url}/api/plugins/kanban/board?board=default`, { headers })
        ).json()) as { columns: Array<{ tasks: unknown[] }> }

        const workers = (await (
          await fetch(`${source.url}/api/plugins/kanban/workers/active?board=default`, { headers })
        ).json()) as { count: number; workers: Array<{ worker_pid: number; ended_at?: unknown }> }

        observed.profile += profiles.profiles.length
        observed.session += sessions.total
        observed.task += board.columns.reduce((sum, column) => sum + column.tasks.length, 0)
        observed.worker += workers.count

        const expectedWorkerIds = new Set(
          fixture.contract.entities
            .filter(row => row.kind === 'worker' && row.connectionId === source.connectionId)
            .map(row => row.workerId)
        )

        expect(new Set(workers.workers.map(row => `pid:${row.worker_pid}`))).toEqual(expectedWorkerIds)
        expect(workers.workers.every(row => row.ended_at == null)).toBe(true)
      }

      expect(observed).toEqual({
        profile: fixture.contract.entitiesByKind.profile,
        session: fixture.contract.entitiesByKind.session,
        task: fixture.contract.entitiesByKind.task,
        worker: fixture.contract.entitiesByKind.worker
      })
      expect(await readStandardPopulation(gateways)).toEqual(expectedStandardProjection(fixture.contract))

      expect(Object.values(observed).reduce((sum, value) => sum + value, 0)).toBe(
        population - fixture.contract.entitiesByKind.subagent
      )
    } finally {
      await gateways?.close()
      fixture.cleanup()
    }
  })
}

test('subagent population requires a supported real gateway emission path', () => {
  test.skip(
    true,
    'Real /api/ws supports subagent interrupt, steer, and replay but has no authenticated fixture emission method; direct store mutation is forbidden and this skip is not subagent evidence.'
  )
})

test('GPU packaged path rejects dev, absent, dirty, fallback, and mismatched packages', () => {
  const clean = {
    builtAt: '2026-08-31T12:00:00.000Z',
    commit: 'a'.repeat(40),
    dirty: false,
    schemaVersion: 1,
    source: 'local'
  }

  const binary = '/fixture/Hermes.app/Contents/MacOS/Hermes'

  expect(() =>
    assertGpuPackagedEligibility({ binaryExists: false, binaryPath: binary, headSha: clean.commit, stamp: clean })
  ).toThrow(/packaged binary is missing/iu)
  expect(() =>
    assertGpuPackagedEligibility({
      binaryExists: true,
      binaryPath: 'node_modules/.bin/electron',
      headSha: clean.commit,
      stamp: clean
    })
  ).toThrow(/dev Electron/iu)
  expect(() =>
    assertGpuPackagedEligibility({
      binaryExists: true,
      binaryPath: binary,
      headSha: clean.commit,
      stamp: { ...clean, dirty: true }
    })
  ).toThrow(/dirty/iu)
  expect(() =>
    assertGpuPackagedEligibility({
      binaryExists: true,
      binaryPath: binary,
      headSha: clean.commit,
      stamp: { ...clean, commit: '0'.repeat(40), source: 'fallback' }
    })
  ).toThrow(/fallback/iu)
  expect(() =>
    assertGpuPackagedEligibility({ binaryExists: true, binaryPath: binary, headSha: 'b'.repeat(40), stamp: clean })
  ).toThrow(/does not match/iu)

  expect(
    assertGpuPackagedEligibility({ binaryExists: true, binaryPath: binary, headSha: clean.commit, stamp: clean })
  ).toEqual({
    buildSha: clean.commit,
    executablePath: binary
  })
})

test('GPU packaged launch never disables the GPU', () => {
  const options = gpuPackagedLaunchOptions({
    executablePath: '/fixture/Hermes.app/Contents/MacOS/Hermes',
    env: {
      DATABASE_URL: 'must-not-cross',
      GH_PAT: 'must-not-cross',
      HERMES_HOME: '/tmp/fixture',
      NPM_CONFIG__AUTH: 'must-not-cross',
      NODE_OPTIONS: '--inspect',
      OPENAI_API_KEY: 'must-not-cross'
    },
    userDataDir: '/tmp/fixture-user-data'
  })

  expect(options.args).toEqual(['--no-sandbox', '--user-data-dir=/tmp/fixture-user-data'])
  expect(options.args.join(' ')).not.toContain('--disable-gpu')
  expect(options.env.HERMES_HOME).toBe('/tmp/fixture')
  expect(options.env.OPENAI_API_KEY).toBeUndefined()
  expect(options.env.GH_PAT).toBeUndefined()
  expect(options.env.DATABASE_URL).toBeUndefined()
  expect(options.env.NPM_CONFIG__AUTH).toBeUndefined()
  expect(options.env.NODE_OPTIONS).toBeUndefined()
})

function walkFiles(root: string): string[] {
  return fs.readdirSync(root, { withFileTypes: true }).flatMap(entry => {
    const target = path.join(root, entry.name)

    return entry.isDirectory() ? walkFiles(target) : [target]
  })
}

function sourceDistribution(entities: readonly { connectionId: string }[]): Record<string, number> {
  return Object.fromEntries(
    ['local', 'remote-lab', 'remote-archive'].map(connectionId => [
      connectionId,
      entities.filter(entity => entity.connectionId === connectionId).length
    ])
  )
}
