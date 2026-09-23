import { type ChildProcess, spawn, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import * as fs from 'node:fs'
import * as net from 'node:net'
import * as os from 'node:os'
import * as path from 'node:path'

const REPO_ROOT = path.resolve(import.meta.dirname, '..', '..', '..')
const CONTRACT_VERSION = 'lunar-city-population-v3' as const
const EXACT_SHA = /^[a-f0-9]{40}$/u

export const GROUP_DISTRICTS = Object.freeze([
  ['Arts Studio', 'garden'],
  ['Acceptance & Release', 'review'],
  ['Archive and Acquisition', 'library'],
  ['CI Repair Triage', 'triage'],
  ['Community Intake', 'bus'],
  ['Content Studio', 'library'],
  ['Control Plane Incidents', 'council'],
  ['Core Runtime & UX Repairs', 'project'],
  ['Data & Performance Repairs', 'lab'],
  ['Editorial Desk', 'library'],
  ['Engineering Guild', 'project'],
  ['Federation Council', 'council'],
  ['Knowledge Commons', 'library'],
  ['Memory Stewardship', 'library'],
  ['Operations and Release', 'depot'],
  ['PR Merge Train', 'depot'],
  ['Research Lab', 'lab'],
  ['Research Review Board', 'review'],
  ['Upstream Hermes Maintenance', 'project']
] as const)

export const LEADER_FAMILIES = Object.freeze([
  'lunar-fox',
  'archive-owl',
  'triage-hare',
  'council-lynx',
  'depot-bear',
  'garden-moth'
] as const)

export type LunarCityPopulation = 25 | 100 | 250
type EntityKind = 'profile' | 'session' | 'subagent' | 'task' | 'worker'
type Activity = 'active' | 'idle' | 'unavailable'
type Lod = 'near' | 'mid' | 'far' | 'aggregate'

export interface PopulationEntity {
  activity: Activity
  connectionId: string
  displayId: string
  durableId: string
  exactKey: string
  groups: readonly string[]
  kind: EntityKind
  leaderFamily?: (typeof LEADER_FAMILIES)[number]
  lod: Lod
  profile: string
  sourceLabel: string
  title: string
  board?: string
  runId?: string
  sessionId?: string
  subagentId?: string
  taskId?: string
  workerId?: string
}

export interface PopulationContract {
  activity: Readonly<Record<Activity, number>>
  digest: string
  entities: readonly PopulationEntity[]
  entitiesByKind: Readonly<Record<EntityKind, number>>
  groups: typeof GROUP_DISTRICTS
  leaderFamilies: typeof LEADER_FAMILIES
  lod: Readonly<Record<Lod, number>>
  population: LunarCityPopulation
  version: typeof CONTRACT_VERSION
}

interface ScenarioShape {
  activity: Record<Activity, number>
  kinds: Record<EntityKind, number>
  lod: Record<Lod, number>
}

interface CanonicalIdentityInput {
  board?: string
  connectionId: string
  durableId: string
  kind: EntityKind
  profile: string
  runId?: string
  sessionId?: string
  subagentId?: string
  taskId?: string
  workerId?: string
}

function identityField(name: string, value: string): string {
  return `${name}:string:${value.length}:${encodeURIComponent(value)}`
}

function optionalIdentityField(name: string, value: string | undefined): string {
  return value === undefined ? `${name}:undefined` : identityField(name, value)
}

/** Mirrors the production typed, length-prefixed Lunar City identity contract. */
export function canonicalEntityKey(identity: CanonicalIdentityInput): string {
  const identityKind = identity.kind === 'task' || identity.kind === 'worker' ? 'kanban' : identity.kind
  const common = [identityField('kind', identityKind), identityField('connection', identity.connectionId)]

  if (identity.kind === 'profile') {
    return [...common, identityField('profile', identity.profile)].join(':')
  }

  if (identity.kind === 'session') {
    return [...common, identityField('profile', identity.profile), identityField('session', identity.sessionId!)].join(
      ':'
    )
  }

  if (identity.kind === 'subagent') {
    return [
      ...common,
      identityField('profile', identity.profile),
      identityField('session', identity.sessionId!),
      identityField('subagent', identity.subagentId!)
    ].join(':')
  }

  return [
    ...common,
    identityField('profile', identity.profile),
    identityField('board', identity.board!),
    identityField('task', identity.taskId!),
    optionalIdentityField('run', identity.runId),
    optionalIdentityField('worker', identity.workerId)
  ].join(':')
}

const SHAPES: Readonly<Record<LunarCityPopulation, ScenarioShape>> = Object.freeze({
  25: {
    activity: { active: 8, idle: 16, unavailable: 1 },
    kinds: { profile: 19, session: 2, subagent: 1, task: 2, worker: 1 },
    lod: { near: 24, mid: 0, far: 0, aggregate: 1 }
  },
  100: {
    activity: { active: 25, idle: 70, unavailable: 5 },
    kinds: { profile: 70, session: 12, subagent: 6, task: 8, worker: 4 },
    lod: { near: 24, mid: 32, far: 28, aggregate: 16 }
  },
  250: {
    activity: { active: 60, idle: 180, unavailable: 10 },
    kinds: { profile: 180, session: 25, subagent: 15, task: 20, worker: 10 },
    lod: { near: 24, mid: 48, far: 82, aggregate: 96 }
  }
})

const FIXTURE_PID_BASE: Readonly<Record<string, number>> = Object.freeze({
  local: 41_000,
  'remote-lab': 42_000,
  'remote-archive': 43_000
})

function deterministicWorkerId(connectionId: string, ordinal: number): string {
  return `pid:${(FIXTURE_PID_BASE[connectionId] ?? 49_000) + ordinal + 1}`
}

export function buildPopulationContract(population: LunarCityPopulation): PopulationContract {
  const shape = SHAPES[population]
  const entities: PopulationEntity[] = []
  const connections = ['local', 'remote-lab', 'remote-archive'] as const
  const labels = { local: 'This device', 'remote-lab': 'Hermes Revenue Lab', 'remote-archive': 'Hermes Desktop' }
  const profilesByConnection = new Map<string, string[]>()
  let ordinal = 0

  for (const kind of ['profile', 'session', 'subagent', 'task', 'worker'] as const) {
    for (let index = 0; index < shape.kinds[kind]; index += 1) {
      const collision = kind === 'profile' && index < 2
      const connectionId = collision ? connections[index]! : connections[(index + ordinal) % connections.length]!
      const implicitDefault = kind === 'profile' && [2, 3, 4].includes(index)

      const durableId = implicitDefault
        ? 'default'
        : collision
          ? 'shared-steward'
          : `${kind}-${String(index).padStart(3, '0')}`

      const owners = profilesByConnection.get(connectionId) ?? []
      const profile = kind === 'profile' ? durableId : owners[index % owners.length]!
      const groupIndex = kind === 'profile' && index < GROUP_DISTRICTS.length - 1 ? index : -1
      const groups = groupIndex >= 0 ? [GROUP_DISTRICTS[groupIndex]![0]] : []

      // Exercise multi-membership while retaining a true no-group profile.
      if (kind === 'profile' && index === 0) {
        groups.push(GROUP_DISTRICTS.at(-1)![0])
      }

      const ownerSession = `session-${String(index % Math.max(1, shape.kinds.session)).padStart(3, '0')}`
      const taskId = `task-${String(index % Math.max(1, shape.kinds.task)).padStart(3, '0')}`
      const runId = kind === 'worker' ? String(1000 + index) : undefined
      const workerId = kind === 'worker' ? deterministicWorkerId(connectionId, index) : undefined

      const identity = {
        kind,
        connectionId,
        profile,
        durableId,
        board: 'default',
        runId,
        sessionId: kind === 'session' ? durableId : ownerSession,
        subagentId: kind === 'subagent' ? durableId : undefined,
        taskId: kind === 'task' ? durableId : taskId,
        workerId
      }

      entities.push({
        activity: 'idle',
        connectionId,
        displayId: durableId,
        durableId,
        exactKey: canonicalEntityKey(identity),
        groups: Object.freeze(groups),
        kind,
        ...(kind === 'profile' ? { leaderFamily: LEADER_FAMILIES[index % LEADER_FAMILIES.length] } : {}),
        lod: 'aggregate',
        profile,
        sourceLabel: labels[connectionId],
        title: `${kind[0]!.toUpperCase()}${kind.slice(1)} ${String(index + 1).padStart(3, '0')}`,
        ...(kind === 'session' || kind === 'subagent' ? { sessionId: identity.sessionId } : {}),
        ...(kind === 'subagent' ? { subagentId: identity.subagentId } : {}),
        ...(kind === 'task' || kind === 'worker'
          ? { board: identity.board, runId: identity.runId, taskId: identity.taskId }
          : {}),
        ...(kind === 'worker' ? { workerId: identity.workerId } : {})
      })

      if (kind === 'profile') {
        profilesByConnection.set(connectionId, [...owners, profile])
      }

      ordinal += 1
    }
  }

  for (const worker of entities.filter(row => row.kind === 'worker')) {
    const task =
      entities.find(row => row.kind === 'task' && row.connectionId === worker.connectionId) ??
      entities.find(row => row.kind === 'task')!

    worker.taskId = task.taskId
    worker.profile = task.profile
    task.runId = worker.runId
    task.exactKey = canonicalEntityKey({
      kind: task.kind,
      connectionId: task.connectionId,
      durableId: task.durableId,
      profile: task.profile,
      board: task.board,
      runId: task.runId,
      taskId: task.taskId
    })
    worker.exactKey = canonicalEntityKey({
      kind: worker.kind,
      connectionId: worker.connectionId,
      durableId: worker.durableId,
      profile: worker.profile,
      board: worker.board,
      runId: worker.runId,
      taskId: worker.taskId,
      workerId: worker.workerId
    })
  }

  assignBuckets(entities, 'activity', shape.activity, ['active', 'idle', 'unavailable'])
  assignBuckets(entities, 'lod', shape.lod, ['near', 'mid', 'far', 'aggregate'])
  const unavailable = entities.find(row => row.activity === 'unavailable')
  const unavailableProfile = entities.findLast(row => row.kind === 'profile' && row.activity !== 'active')

  if (unavailable && unavailableProfile && unavailable.kind !== 'profile') {
    unavailable.activity = unavailableProfile.activity
    unavailableProfile.activity = 'unavailable'
  }

  const requireActive = (entity: PopulationEntity): void => {
    if (entity.activity === 'active') {
      return
    }

    const donor = entities.find(row => row.kind === 'profile' && row.activity === 'active')

    if (donor) {
      donor.activity = entity.activity
      entity.activity = 'active'
    }
  }

  for (const subagent of entities.filter(row => row.kind === 'subagent')) {
    requireActive(subagent)
  }

  for (const worker of entities.filter(row => row.kind === 'worker')) {
    requireActive(worker)

    const task = entities.find(row => row.kind === 'task' && row.taskId === worker.taskId)

    if (task) {
      requireActive(task)
    }
  }

  const unsigned = {
    activity: shape.activity,
    entities,
    entitiesByKind: shape.kinds,
    groups: GROUP_DISTRICTS,
    leaderFamilies: LEADER_FAMILIES,
    lod: shape.lod,
    population,
    version: CONTRACT_VERSION
  }

  const digest = createHash('sha256').update(stableJson(unsigned)).digest('hex')

  return deepFreeze({ ...unsigned, digest })
}

function assignBuckets<K extends 'activity' | 'lod', V extends PopulationEntity[K]>(
  entities: PopulationEntity[],
  field: K,
  counts: Record<V, number>,
  order: readonly V[]
): void {
  let cursor = 0

  for (const value of order) {
    for (let count = 0; count < counts[value]; count += 1) {
      entities[cursor++]![field] = value
    }
  }
}

export interface LunarCityPopulationFixture {
  cleanup: () => void
  contract: PopulationContract
  contractPath: string
  hermesHome: string
  root: string
  sourceHomes: Readonly<Record<string, string>>
  userDataDir: string
}

export function createLunarCityPopulationFixture(population: LunarCityPopulation): LunarCityPopulationFixture {
  const contract = buildPopulationContract(population)
  const root = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-lunar-city-${population}-`))

  const sourceHomes = {
    local: path.join(root, 'sources', 'local', 'hermes-home'),
    'remote-lab': path.join(root, 'sources', 'remote-lab', 'hermes-home'),
    'remote-archive': path.join(root, 'sources', 'remote-archive', 'hermes-home')
  }

  const userDataDir = path.join(root, 'electron-user-data')
  const contractPath = path.join(root, `${CONTRACT_VERSION}.json`)

  try {
    fs.mkdirSync(userDataDir, { recursive: true })

    for (const home of Object.values(sourceHomes)) {
      fs.mkdirSync(path.join(home, 'profiles'), { recursive: true })
      fs.writeFileSync(
        path.join(home, 'config.yaml'),
        '# Deterministic credential-free Lunar City E2E fixture\n',
        'utf8'
      )
    }

    for (const entity of contract.entities.filter(row => row.kind === 'profile')) {
      const home = sourceHomes[entity.connectionId as keyof typeof sourceHomes]
      const profileDir = entity.profile === 'default' ? home : path.join(home, 'profiles', entity.profile)
      fs.mkdirSync(profileDir, { recursive: true })
      fs.writeFileSync(
        path.join(profileDir, 'SOUL.md'),
        `# ${entity.title}\n\nDeterministic E2E fixture persona.\n`,
        'utf8'
      )
      fs.writeFileSync(path.join(profileDir, 'profile.yaml'), profileYaml(entity), 'utf8')
      fs.writeFileSync(
        path.join(profileDir, 'desktop.json'),
        `${JSON.stringify(
          {
            lunarCityFixture: {
              activity: entity.activity,
              groups: entity.groups,
              leaderFamily: entity.leaderFamily,
              version: CONTRACT_VERSION
            }
          },
          null,
          2
        )}\n`,
        'utf8'
      )
    }

    fs.writeFileSync(contractPath, `${JSON.stringify(contract, null, 2)}\n`, 'utf8')
    seedStandardStores(contractPath, sourceHomes)

    return {
      cleanup: () => fs.rmSync(root, { recursive: true, force: true }),
      contract,
      contractPath,
      hermesHome: sourceHomes.local,
      root,
      sourceHomes: deepFreeze({ ...sourceHomes }),
      userDataDir
    }
  } catch (error) {
    fs.rmSync(root, { recursive: true, force: true })
    throw error
  }
}

export interface StandardGatewaySource {
  close: () => Promise<void>
  connectionId: string
  pid: number
  token: string
  url: string
}

export interface PopulationGateways {
  close: () => Promise<void>
  sources: readonly StandardGatewaySource[]
}

export interface StandardPopulationProjection {
  activity: Readonly<Record<Activity, number>>
  byKind: Readonly<Record<'profile' | 'session' | 'task' | 'worker', number>>
  digest: string
  entityKeys: readonly string[]
  groups: readonly string[]
  leaderFamilies: readonly string[]
  sourceMix: Readonly<Record<string, number>>
}

function projectionFromRows(rows: readonly PopulationEntity[]): StandardPopulationProjection {
  const persisted = rows.filter(
    (row): row is PopulationEntity & { kind: 'profile' | 'session' | 'task' | 'worker' } => row.kind !== 'subagent'
  )

  const projection = {
    activity: {
      active: persisted.filter(row => row.activity === 'active').length,
      idle: persisted.filter(row => row.activity === 'idle').length,
      unavailable: persisted.filter(row => row.activity === 'unavailable').length
    },
    byKind: {
      profile: persisted.filter(row => row.kind === 'profile').length,
      session: persisted.filter(row => row.kind === 'session').length,
      task: persisted.filter(row => row.kind === 'task').length,
      worker: persisted.filter(row => row.kind === 'worker').length
    },
    entityKeys: persisted.map(row => row.exactKey).sort(),
    groups: [...new Set(persisted.flatMap(row => row.groups))].sort(),
    leaderFamilies: [...new Set(persisted.flatMap(row => (row.leaderFamily ? [row.leaderFamily] : [])))].sort(),
    sourceMix: Object.fromEntries(
      [...new Set(persisted.map(row => row.connectionId))]
        .sort()
        .map(connectionId => [connectionId, persisted.filter(row => row.connectionId === connectionId).length])
    )
  }

  return deepFreeze({ ...projection, digest: createHash('sha256').update(stableJson(projection)).digest('hex') })
}

export function expectedStandardProjection(contract: PopulationContract): StandardPopulationProjection {
  return projectionFromRows(contract.entities)
}

/** Read only standard authenticated REST routes and reconstruct production identities. */
export async function readStandardPopulation(gateways: PopulationGateways): Promise<StandardPopulationProjection> {
  const rows: PopulationEntity[] = []

  for (const source of gateways.sources) {
    const headers = { 'X-Hermes-Session-Token': source.token }

    const profiles = (await (await fetch(`${source.url}/api/profiles`, { headers })).json()) as {
      profiles: Array<{ name: string }>
    }

    const sessions = (await (
      await fetch(`${source.url}/api/profiles/sessions?limit=500&profile=all`, { headers })
    ).json()) as {
      sessions: Array<{ ended_at?: null | number; id: string; is_active?: boolean; profile: string }>
    }

    const board = (await (await fetch(`${source.url}/api/plugins/kanban/board?board=default`, { headers })).json()) as {
      columns: Array<{
        tasks: Array<{ assignee: string; current_run_id?: null | number | string; id: string; status: string }>
      }>
    }

    const workers = (await (
      await fetch(`${source.url}/api/plugins/kanban/workers/active?board=default`, { headers })
    ).json()) as {
      workers: Array<{ profile: string; run_id: number | string; task_id: string; worker_pid: number }>
    }

    for (const profile of profiles.profiles) {
      const overlay = (await (
        await fetch(`${source.url}/api/profiles/${encodeURIComponent(profile.name)}/desktop-overlay`, { headers })
      ).json()) as {
        desktop?: {
          lunarCityFixture?: {
            activity?: Activity
            groups?: string[]
            leaderFamily?: (typeof LEADER_FAMILIES)[number]
          }
        }
      }

      const metadata = overlay.desktop?.lunarCityFixture

      const identity = {
        connectionId: source.connectionId,
        durableId: profile.name,
        kind: 'profile' as const,
        profile: profile.name
      }

      rows.push({
        activity: metadata?.activity ?? 'unavailable',
        connectionId: source.connectionId,
        displayId: profile.name,
        durableId: profile.name,
        exactKey: canonicalEntityKey(identity),
        groups: metadata?.groups ?? [],
        kind: 'profile',
        leaderFamily: metadata?.leaderFamily,
        lod: 'aggregate',
        profile: profile.name,
        sourceLabel: source.connectionId,
        title: profile.name
      })
    }

    for (const session of sessions.sessions) {
      const identity = {
        connectionId: source.connectionId,
        durableId: session.id,
        kind: 'session' as const,
        profile: session.profile,
        sessionId: session.id
      }

      rows.push({
        activity: session.ended_at == null && session.is_active ? 'active' : 'idle',
        connectionId: source.connectionId,
        displayId: session.id,
        durableId: session.id,
        exactKey: canonicalEntityKey(identity),
        groups: [],
        kind: 'session',
        lod: 'aggregate',
        profile: session.profile,
        sessionId: session.id,
        sourceLabel: source.connectionId,
        title: session.id
      })
    }

    for (const task of board.columns.flatMap(column => column.tasks)) {
      const runId = task.current_run_id == null ? undefined : String(task.current_run_id)

      const identity = {
        board: 'default',
        connectionId: source.connectionId,
        durableId: task.id,
        kind: 'task' as const,
        profile: task.assignee,
        runId,
        taskId: task.id
      }

      rows.push({
        activity: task.status === 'running' ? 'active' : 'idle',
        board: 'default',
        connectionId: source.connectionId,
        displayId: task.id,
        durableId: task.id,
        exactKey: canonicalEntityKey(identity),
        groups: [],
        kind: 'task',
        lod: 'aggregate',
        profile: task.assignee,
        runId,
        sourceLabel: source.connectionId,
        taskId: task.id,
        title: task.id
      })
    }

    for (const worker of workers.workers) {
      const runId = String(worker.run_id)
      const workerId = `pid:${worker.worker_pid}`

      const identity = {
        board: 'default',
        connectionId: source.connectionId,
        durableId: workerId,
        kind: 'worker' as const,
        profile: worker.profile,
        runId,
        taskId: worker.task_id,
        workerId
      }

      rows.push({
        activity: 'active',
        board: 'default',
        connectionId: source.connectionId,
        displayId: workerId,
        durableId: workerId,
        exactKey: canonicalEntityKey(identity),
        groups: [],
        kind: 'worker',
        lod: 'aggregate',
        profile: worker.profile,
        runId,
        sourceLabel: source.connectionId,
        taskId: worker.task_id,
        title: workerId,
        workerId
      })
    }
  }

  return projectionFromRows(rows)
}

const SAFE_GATEWAY_ENV = ['PATH', 'TMPDIR', 'TEMP', 'TMP', 'LANG', 'LC_ALL', 'SHELL', 'USER', 'LOGNAME'] as const

function standardHermesBinary(): string {
  const binary = path.join(os.homedir(), '.local', 'bin', 'hermes')

  if (!fs.existsSync(binary)) {
    throw new Error('The authorized ~/.local/bin/hermes launcher is required for standard-route fixtures')
  }

  const version = spawnSync(binary, ['--version-local'], { encoding: 'utf8' })

  if (version.error || version.status !== 0 || !version.stdout.trim()) {
    throw new Error('Hermes --version-local source verification failed; no fixture gateway was started')
  }

  return binary
}

async function reservePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address() as net.AddressInfo
      server.close(error => (error ? reject(error) : resolve(address.port)))
    })
  })
}

async function stopChild(child: ChildProcess): Promise<void> {
  if (!child.pid || child.exitCode !== null) {
    return
  }

  try {
    process.kill(-child.pid, 'SIGTERM')
  } catch {
    child.kill('SIGTERM')
  }

  await new Promise<void>(resolve => {
    const timer = setTimeout(resolve, 2_000)
    child.once('exit', () => {
      clearTimeout(timer)
      resolve()
    })
  })
}

/** Start credential-free real `hermes serve` sources and register their standard desktop routes. */
export async function startPopulationGateways(fixture: LunarCityPopulationFixture): Promise<PopulationGateways> {
  const sources: StandardGatewaySource[] = []

  try {
    for (const connectionId of Object.keys(fixture.sourceHomes)) {
      const port = await reservePort()
      const url = `http://127.0.0.1:${port}`
      const token = `lunar-city-e2e-${connectionId}`

      const env = Object.fromEntries(
        SAFE_GATEWAY_ENV.flatMap(key => (process.env[key] === undefined ? [] : [[key, process.env[key]!]]))
      )

      const child = spawn(
        standardHermesBinary(),
        ['serve', '--host', '127.0.0.1', '--port', String(port), '--skip-build'],
        {
          cwd: REPO_ROOT,
          detached: true,
          env: {
            ...env,
            HERMES_DASHBOARD_SESSION_TOKEN: token,
            HERMES_HOME: fixture.sourceHomes[connectionId]!,
            HERMES_KANBAN_HOME: fixture.sourceHomes[connectionId]!
          },
          stdio: ['ignore', 'pipe', 'pipe']
        }
      )

      let log = ''
      child.stdout?.on('data', chunk => (log += String(chunk)))
      child.stderr?.on('data', chunk => (log += String(chunk)))
      const deadline = Date.now() + 60_000

      while (Date.now() < deadline) {
        if (child.exitCode !== null) {
          throw new Error(`${connectionId} gateway exited (${child.exitCode}):\n${log}`)
        }

        try {
          const response = await fetch(`${url}/api/status`, { headers: { 'X-Hermes-Session-Token': token } })

          if (response.ok) {
            break
          }
        } catch {
          // Standard gateway is still starting.
        }

        await new Promise(resolve => setTimeout(resolve, 250))
      }

      if (Date.now() >= deadline) {
        throw new Error(`${connectionId} gateway did not become ready:\n${log}`)
      }

      sources.push({ close: () => stopChild(child), connectionId, pid: child.pid!, token, url })
    }

    fs.writeFileSync(
      path.join(fixture.userDataDir, 'connections.json'),
      `${JSON.stringify(
        {
          version: 2,
          primary: 'local',
          launchMode: 'primary',
          lastUsed: 'local',
          connections: sources.map(source => ({
            id: source.connectionId,
            kind: source.connectionId === 'local' ? 'local' : 'remote',
            label: source.connectionId,
            ...(source.connectionId === 'local'
              ? {}
              : { authMode: 'token', token: { encoding: 'plain', value: source.token }, url: source.url })
          }))
        },
        null,
        2
      )}\n`,
      { encoding: 'utf8', mode: 0o600 }
    )

    return {
      sources,
      close: async () => {
        await Promise.allSettled(sources.map(source => source.close()))
      }
    }
  } catch (error) {
    await Promise.allSettled(sources.map(source => source.close()))
    throw error
  }
}

function profileYaml(entity: PopulationEntity): string {
  const groups = entity.groups.length
    ? `\n      groups:\n${entity.groups.map(group => `        - ${JSON.stringify(group)}`).join('\n')}\n      group: ${JSON.stringify(entity.groups[0])}`
    : ''

  return `display_name: ${JSON.stringify(entity.title)}
description: Deterministic Lunar City population fixture
ui_meta:
  hermes-bots:
    title: ${JSON.stringify(entity.title)}
    leader_family: ${JSON.stringify(entity.leaderFamily)}${groups}
_ui_meta_revisions:
  hermes-bots: 1
`
}

function seedStandardStores(contractPath: string, sourceHomes: Record<string, string>): void {
  const script = String.raw`
import json, os, pathlib, sqlite3, sys
from hermes_state import SessionDB
from hermes_cli.kanban_db_connect import connect

contract = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
homes = json.loads(sys.argv[2])
rows = contract["entities"]

for connection_id, home_raw in homes.items():
    home = pathlib.Path(home_raw)
    os.environ["HERMES_HOME"] = str(home)
    os.environ["HERMES_KANBAN_HOME"] = str(home)
    for row in rows:
        if row["connectionId"] != connection_id or row["kind"] != "session":
            continue
        profile_home = home if row["profile"] == "default" else home / "profiles" / row["profile"]
        profile_home.mkdir(parents=True, exist_ok=True)
        db = SessionDB(db_path=profile_home / "state.db")
        try:
            db.create_session(row["sessionId"], "desktop", profile_name=row["profile"], cwd=str(home))
            if row["activity"] != "active":
                db.end_session(row["sessionId"], "completed")
        finally:
            db.close()

    db = SessionDB(db_path=home / "state.db")
    db.close()
    conn = connect(db_path=home / "kanban.db")
    try:
        now = 1788172800
        task_rows = [row for row in rows if row["connectionId"] == connection_id and row["kind"] == "task"]
        workers = [row for row in rows if row["connectionId"] == connection_id and row["kind"] == "worker"]
        for index, row in enumerate(task_rows):
            status = "running" if row["activity"] == "active" else "ready"
            conn.execute(
                "INSERT INTO tasks (id,title,assignee,status,created_at,workspace_kind,project_id,session_id) VALUES (?,?,?,?,?,?,?,?)",
                (row["durableId"], row["title"], row["profile"], status, now + index, "scratch", "fixture-project", f"session-{index:03d}"),
            )
        for index, row in enumerate(workers):
            task = next(task_row for task_row in task_rows if task_row["taskId"] == row["taskId"])
            run_id = int(row["runId"])
            status = "running"
            conn.execute(
                "INSERT INTO task_runs (id,task_id,profile,status,claim_lock,worker_pid,last_heartbeat_at,started_at,ended_at,outcome) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, task["durableId"], row["profile"], status, row["durableId"], int(row["workerId"].split(":", 1)[1]), now, now, None, None),
            )
            conn.execute("UPDATE tasks SET status='running', current_run_id=?, claim_lock=?, last_heartbeat_at=? WHERE id=?", (run_id, row["durableId"], now, task["durableId"]))
        conn.commit()
    finally:
        conn.close()
`

  const executable = path.join(REPO_ROOT, '.venv', 'bin', 'python')

  if (!fs.existsSync(executable)) {
    throw new Error('Current checkout .venv/bin/python is required for isolated fixture seeding')
  }

  const result = spawnSync(executable, ['-c', script, contractPath, JSON.stringify(sourceHomes)], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    env: { ...process.env, PYTHONPATH: REPO_ROOT }
  })

  if (result.status !== 0) {
    throw new Error(`Could not seed standard Hermes stores:\n${result.stderr || result.stdout}`)
  }
}

export interface PackagedBuildStamp {
  builtAt: string
  commit: string
  dirty: false
  schemaVersion: 1
  source: 'ci' | 'local'
}

export interface GpuPackagedEligibilityInput {
  binaryExists: boolean
  binaryPath: string
  headSha: string
  stamp: {
    builtAt?: string
    commit: string
    dirty: boolean
    schemaVersion?: number
    source: string
  }
}

export function assertGpuPackagedEligibility(input: GpuPackagedEligibilityInput): {
  buildSha: string
  executablePath: string
} {
  if (!input.binaryExists) {
    throw new Error(`Packaged binary is missing: ${input.binaryPath}`)
  }

  if (/(?:node_modules[\\/]\.bin[\\/]electron|(^|[\\/])electron(?:\.exe)?$)/iu.test(input.binaryPath)) {
    throw new Error('Refusing dev Electron for packaged GPU evidence')
  }

  if (input.stamp.dirty) {
    throw new Error('Refusing dirty packaged build stamp')
  }

  if (!['ci', 'local'].includes(input.stamp.source) || /^0{40}$/u.test(input.stamp.commit)) {
    throw new Error('Refusing fallback packaged build stamp')
  }

  if (
    input.stamp.schemaVersion !== 1 ||
    typeof input.stamp.builtAt !== 'string' ||
    !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/u.test(input.stamp.builtAt)
  ) {
    throw new Error('Refusing non-canonical packaged build stamp')
  }

  if (!EXACT_SHA.test(input.stamp.commit) || !EXACT_SHA.test(input.headSha)) {
    throw new Error('Packaged build and HEAD must be exact 40-character SHAs')
  }

  if (input.stamp.commit !== input.headSha) {
    throw new Error('Packaged build SHA does not match checkout HEAD')
  }

  return { buildSha: input.stamp.commit, executablePath: input.binaryPath }
}

export function gpuPackagedLaunchOptions(input: {
  env: Record<string, string>
  executablePath: string
  userDataDir?: string
}): { args: string[]; env: Record<string, string>; executablePath: string } {
  const allowed = new Set([
    ...SAFE_GATEWAY_ENV,
    'HERMES_HOME',
    'HERMES_DESKTOP_APP_NAME',
    'HERMES_DESKTOP_SKIP_QUIT_CONFIRM',
    'HERMES_LUNAR_CITY_PERF_ACCEPTANCE',
    'HERMES_LUNAR_CITY_PERF_NONCE'
  ])

  const env = Object.fromEntries(Object.entries(input.env).filter(([key]) => allowed.has(key)))

  const args = ['--no-sandbox', ...(input.userDataDir ? [`--user-data-dir=${input.userDataDir}`] : [])]

  return { args, env, executablePath: input.executablePath }
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(',')}]`
  }

  if (value && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => `${JSON.stringify(key)}:${stableJson(child)}`)
      .join(',')}}`
  }

  return JSON.stringify(value)
}

function deepFreeze<T>(value: T): T {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value)

    for (const child of Object.values(value)) {
      deepFreeze(child)
    }
  }

  return value
}
