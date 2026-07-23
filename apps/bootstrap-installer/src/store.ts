import { atom, computed } from 'nanostores'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import { invoke } from '@tauri-apps/api/core'

/*
 * Bootstrap state store — single source of truth for installer screens.
 *
 * Lives in nanostores per the project's TypeScript guidelines (apps/desktop
 * AGENTS.md): "Prefer small nanostores over component state when state is
 * shared, reused, or read by distant UI."
 *
 * One channel from Rust ('bootstrap' event), discriminated by payload.type.
 * We translate those events into typed atom updates here so the rest of
 * the app only deals with React-friendly state.
 */

// ---------------------------------------------------------------------------
// Types — mirror src-tauri/src/events.rs
// ---------------------------------------------------------------------------

export interface StageInfo {
  name: string
  title: string
  category: string
  needs_user_input: boolean
}

export type StageState = 'running' | 'succeeded' | 'skipped' | 'failed'

export interface StageRecord {
  info: StageInfo
  state: StageState | null
  durationMs?: number
  /** Wall-clock time the stage entered `running`, stamped client-side so the UI
   * can tick a live elapsed timer for long steps. Preserved across repeated
   * running events. */
  startedAt?: number
  error?: string
}

export interface BootstrapStateModel {
  status: 'idle' | 'running' | 'completed' | 'failed'
  protocolVersion: number | null
  stages: Record<string, StageRecord>
  stageOrder: string[]
  currentStage: string | null
  installRoot: string | null
  error: string | null
  logs: Array<{ stage?: string; line: string; stream?: 'stdout' | 'stderr' }>
}

const INITIAL: BootstrapStateModel = {
  status: 'idle',
  protocolVersion: null,
  stages: {},
  stageOrder: [],
  currentStage: null,
  installRoot: null,
  error: null,
  logs: []
}

// ---------------------------------------------------------------------------
// Atoms
// ---------------------------------------------------------------------------

export type Route = 'welcome' | 'progress' | 'success' | 'failure'

/// How the installer was launched, mirrored from src-tauri AppMode.
/// 'install' = first-run onboarding (bare launch). 'update' = driven by the
/// desktop app handing off via `Hermes-Setup.exe --update`.
export type AppMode = 'install' | 'update'

export const $route = atom<Route>('welcome')
export const $mode = atom<AppMode>('install')
export const $bootstrap = atom<BootstrapStateModel>(INITIAL)
export const $logPath = atom<string | null>(null)
export const $hermesHome = atom<string | null>(null)

export const $progress = computed($bootstrap, (b) => {
  const total = b.stageOrder.length
  if (total === 0) return { done: 0, total: 0, fraction: 0 }
  let done = 0
  for (const name of b.stageOrder) {
    const s = b.stages[name]?.state
    if (s === 'succeeded' || s === 'skipped' || s === 'failed') done += 1
  }
  return { done, total, fraction: done / total }
})

/** Apply a stage transition: stamp `startedAt` on the running edge, track the
 * active stage. Shared by the live Rust handler and the fake-boot preview so the
 * two behave identically. */
function withStageState(
  cur: BootstrapStateModel,
  name: string,
  state: StageState,
  durationMs?: number,
  error?: string
): BootstrapStateModel {
  const existing = cur.stages[name]
  if (!existing) return cur
  return {
    ...cur,
    stages: {
      ...cur.stages,
      [name]: {
        ...existing,
        state,
        startedAt: state === 'running' ? (existing.startedAt ?? Date.now()) : existing.startedAt,
        durationMs,
        error
      }
    },
    currentStage: state === 'running' ? name : cur.currentStage
  }
}

// ---------------------------------------------------------------------------
// Tauri event subscription
// ---------------------------------------------------------------------------

interface BootstrapManifestEvent {
  type: 'manifest'
  stages: StageInfo[]
  protocolVersion: number | null
}

interface BootstrapStageEvent {
  type: 'stage'
  name: string
  state: StageState
  durationMs?: number
  error?: string
}

interface BootstrapLogEvent {
  type: 'log'
  stage?: string
  line: string
  stream?: 'stdout' | 'stderr'
}

interface BootstrapCompleteEvent {
  type: 'complete'
  installRoot: string
  marker: unknown
}

interface BootstrapFailedEvent {
  type: 'failed'
  stage?: string
  error: string
}

type BootstrapEvent =
  | BootstrapManifestEvent
  | BootstrapStageEvent
  | BootstrapLogEvent
  | BootstrapCompleteEvent
  | BootstrapFailedEvent

let unlisten: UnlistenFn | null = null

export async function initialize(): Promise<void> {
  if (unlisten) return

  // Dev-only isolated preview (see runFakeBoot): drive the screens in a plain
  // browser, no Tauri backend, no real install.
  const fake = fakeMode()
  if (fake) {
    unlisten = () => {}
    $logPath.set('~/.hermes/logs/bootstrap-installer.log')
    $hermesHome.set('~/.hermes')
    $mode.set(fake === 'update' ? 'update' : 'install')
    // Update auto-runs (it's a hand-off); install/failure wait for the welcome click.
    if (fake === 'update') void runFakeBoot('update')
    return
  }

  // Pull static info on mount for the diagnostics footer.
  try {
    const [logPath, hermesHome, mode] = await Promise.all([
      invoke<string>('get_log_path'),
      invoke<string>('get_hermes_home'),
      invoke<AppMode>('get_mode')
    ])
    $logPath.set(logPath)
    $hermesHome.set(hermesHome)
    $mode.set(mode)
  } catch (err) {
    console.warn('failed to fetch installer paths', err)
  }

  unlisten = await listen<BootstrapEvent>('bootstrap', (event) => {
    const payload = event.payload
    const cur = $bootstrap.get()
    switch (payload.type) {
      case 'manifest': {
        const stages: Record<string, StageRecord> = {}
        const order: string[] = []
        for (const s of payload.stages) {
          stages[s.name] = { info: s, state: null }
          order.push(s.name)
        }
        $bootstrap.set({
          ...cur,
          status: 'running',
          protocolVersion: payload.protocolVersion,
          stages,
          stageOrder: order,
          currentStage: null,
          installRoot: null,
          error: null,
          logs: []
        })
        $route.set('progress')
        break
      }
      case 'stage': {
        if (!cur.stages[payload.name]) {
          console.warn('stage event for unknown stage', payload.name)
          break
        }
        $bootstrap.set(
          withStageState(cur, payload.name, payload.state, payload.durationMs, payload.error)
        )
        break
      }
      case 'log': {
        const logs = [...cur.logs, { stage: payload.stage, line: payload.line, stream: payload.stream }]
        // Keep the rolling buffer bounded so the UI doesn't get OOM'd
        // during a long install (playwright chromium download is ~10k lines).
        const trimmed = logs.length > 2000 ? logs.slice(-2000) : logs
        $bootstrap.set({ ...cur, logs: trimmed })
        break
      }
      case 'complete':
        $bootstrap.set({
          ...cur,
          status: 'completed',
          installRoot: payload.installRoot,
          currentStage: null
        })
        // Install: show the "launch Hermes" success screen. Update: this is a
        // hand-off — the installer relaunches the desktop and exits within a
        // few hundred ms, so routing to success just flashes that screen
        // before the window closes. Stay on progress until we exit.
        if ($mode.get() !== 'update') {
          $route.set('success')
        }
        break
      case 'failed':
        $bootstrap.set({
          ...cur,
          status: 'failed',
          error: payload.error,
          currentStage: null
        })
        $route.set('failure')
        break
    }
  })

  // Update mode is a hand-off, not a user-initiated flow: the desktop already
  // exited and re-launched us as `--update`. Kick the update immediately so
  // the user lands on progress, not a redundant "click to update" screen.
  if ($mode.get() === 'update') {
    void startUpdate()
  }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export async function startInstall(opts?: { branch?: string }): Promise<void> {
  const fake = fakeMode()
  if (fake) {
    void runFakeBoot(fake === 'failure' ? 'failure' : 'install')
    return
  }
  // Reset before kicking off so a retry from the failure screen clears
  // the previous run's state.
  $bootstrap.set(INITIAL)
  $route.set('progress')
  await invoke('start_bootstrap', {
    args: {
      commit: null,
      branch: opts?.branch ?? null,
      include_desktop: true,
      hermes_home: null
    }
  })
}

export async function startUpdate(): Promise<void> {
  if (fakeMode()) {
    void runFakeBoot('update')
    return
  }
  // Update is driven by the desktop handing off (Hermes-Setup.exe --update);
  // there's no welcome click. Reset + jump straight to progress, then let the
  // Rust side stream the synthetic update manifest.
  $bootstrap.set(INITIAL)
  $route.set('progress')
  await invoke('start_update')
}

export async function cancelInstall(): Promise<void> {
  if (fakeMode()) {
    fakeCancelled = true
    return
  }
  await invoke('cancel_bootstrap')
}

export async function launchHermesDesktop(): Promise<void> {
  if (fakeMode()) throw new Error('Preview mode — launching is disabled.')
  const installRoot = $bootstrap.get().installRoot
  if (!installRoot) throw new Error('no install root')
  await invoke('launch_hermes_desktop', { installRoot })
}

export async function openLogDir(): Promise<void> {
  if (fakeMode()) return
  await invoke('open_log_dir')
}

// ---------------------------------------------------------------------------
// Dev-only isolated preview ("fake boot")
//
// Synthesises the manifest + stage/log events Rust normally streams, so the
// whole reskin can be reviewed in a plain browser (`npm run dev`):
//   ?fake=install   welcome → [ INSTALL ] → success
//   ?fake=update    auto-runs the granular update flow
//   ?fake=failure   install that fails partway
// Gated on import.meta.env.DEV → stripped from the shipped Tauri bundle.
// ---------------------------------------------------------------------------

type FakeMode = 'install' | 'update' | 'failure'

function fakeMode(): FakeMode | null {
  if (!import.meta.env.DEV || typeof window === 'undefined') return null
  const v = new URLSearchParams(window.location.search).get('fake')
  return v === 'install' || v === 'update' || v === 'failure' ? v : null
}

interface FakeStage {
  name: string
  title: string
}

const FAKE_INSTALL_STAGES: FakeStage[] = [
  { name: 'system-packages', title: 'System packages' },
  { name: 'uv', title: 'uv' },
  { name: 'python', title: 'Python environment' },
  { name: 'repo', title: 'Hermes repository' },
  { name: 'dependencies', title: 'Python dependencies' },
  { name: 'node', title: 'Node runtime' },
  { name: 'desktop', title: 'Desktop app' }
]

const FAKE_UPDATE_STAGES: FakeStage[] = [
  { name: 'handoff', title: 'Preparing to update' },
  { name: 'update', title: 'Downloading the latest version' },
  { name: 'rebuild', title: 'Rebuilding the desktop app' },
  { name: 'install', title: 'Installing the update' }
]

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

let fakeRunning = false
let fakeCancelled = false

const fakeStage = (name: string, state: StageState, durationMs?: number, error?: string) =>
  $bootstrap.set(withStageState($bootstrap.get(), name, state, durationMs, error))

const fakeLog = (stage: string, line: string) =>
  $bootstrap.set({ ...$bootstrap.get(), logs: [...$bootstrap.get().logs, { stage, line, stream: 'stdout' }] })

const fakeFail = (error: string) =>
  $bootstrap.set({ ...$bootstrap.get(), status: 'failed', error, currentStage: null })

async function runFakeBoot(kind: FakeMode): Promise<void> {
  if (fakeRunning) return
  fakeRunning = true
  fakeCancelled = false
  try {
    const stages = kind === 'update' ? FAKE_UPDATE_STAGES : FAKE_INSTALL_STAGES
    const cancelled = () => {
      if (!fakeCancelled) return false
      fakeFail(kind === 'update' ? 'Update cancelled.' : 'Install cancelled.')
      $route.set('failure')
      return true
    }

    $bootstrap.set({
      ...INITIAL,
      status: 'running',
      stageOrder: stages.map((s) => s.name),
      stages: Object.fromEntries(
        stages.map((s): [string, StageRecord] => [
          s.name,
          { info: { ...s, category: kind, needs_user_input: false }, state: null }
        ])
      )
    })
    $route.set('progress')

    // Blow up midway in the failure preview so the failure screen shows.
    const failAt = kind === 'failure' ? stages[Math.floor(stages.length / 2)]?.name : null

    for (const s of stages) {
      if (cancelled()) return
      fakeStage(s.name, 'running')

      const durationMs = 700 + Math.floor(Math.random() * 2200)
      const lines = Math.max(2, Math.round(durationMs / 450))
      for (let l = 0; l < lines; l++) {
        await sleep(durationMs / lines)
        if (cancelled()) return
        fakeLog(s.name, `[${s.name}] ${s.title.toLowerCase()} — step ${l + 1}/${lines}…`)
      }

      if (s.name === failAt) {
        fakeStage(s.name, 'failed', durationMs, 'Simulated failure for preview.')
        fakeFail('Simulated failure for preview (fake boot).')
        $route.set('failure')
        return
      }
      fakeStage(s.name, 'succeeded', durationMs)
    }

    $bootstrap.set({ ...$bootstrap.get(), status: 'completed', currentStage: null })
    // Install lands on success; update stays on progress (the real updater
    // relaunches the desktop and exits from there).
    if (kind !== 'update') $route.set('success')
  } finally {
    fakeRunning = false
  }
}
