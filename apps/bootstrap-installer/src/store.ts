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
  logs: Array<{ stage?: string; line: string }>
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
        const existing = cur.stages[payload.name]
        if (!existing) {
          console.warn('stage event for unknown stage', payload.name)
          break
        }
        const next: StageRecord = {
          ...existing,
          state: payload.state,
          durationMs: payload.durationMs,
          error: payload.error
        }
        $bootstrap.set({
          ...cur,
          stages: { ...cur.stages, [payload.name]: next },
          currentStage:
            payload.state === 'running' ? payload.name : cur.currentStage
        })
        break
      }
      case 'log': {
        const logs = [...cur.logs, { stage: payload.stage, line: payload.line }]
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
  // Update is driven by the desktop handing off (Hermes-Setup.exe --update);
  // there's no welcome click. Reset + jump straight to progress, then let the
  // Rust side stream the synthetic update manifest.
  $bootstrap.set(INITIAL)
  $route.set('progress')
  await invoke('start_update')
}

export async function cancelInstall(): Promise<void> {
  await invoke('cancel_bootstrap')
}

export async function launchHermesDesktop(): Promise<void> {
  const installRoot = $bootstrap.get().installRoot
  if (!installRoot) throw new Error('no install root')
  await invoke('launch_hermes_desktop', { installRoot })
}

export async function openLogDir(): Promise<void> {
  await invoke('open_log_dir')
}
