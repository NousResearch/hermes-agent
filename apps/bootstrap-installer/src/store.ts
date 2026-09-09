import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import { atom } from 'nanostores'

/*
 * Bootstrap state store — single source of truth for the setup screens.
 *
 * North Forge's bootstrap is ONE monolithic `bootstrap-north-forge.ps1`
 * run (no `-Manifest` / per-stage JSON protocol), so there is no stage
 * list: the progress screen shows an indeterminate bar + the live log.
 * One channel from Rust ('bootstrap' event), discriminated by payload.type.
 */

// ---------------------------------------------------------------------------
// Types — mirror src-tauri/src/repo.rs + events.rs
// ---------------------------------------------------------------------------

export interface DriveInfo {
  letter: string
  path: string
  isSystem: boolean
  hasCheckout: boolean
  checkoutPath: string | null
}

export interface RepoInfo {
  repoRoot: string | null
  parent: string | null
  leaf: string | null
  venvDir: string | null
  dataDir: string | null
  bootstrapScript: string | null
  launcherCmd: string | null
  bootstrapped: boolean
  onSystemDrive: boolean
  source: string
  drives: DriveInfo[]
}

export interface BootstrapStateModel {
  status: 'idle' | 'running' | 'completed' | 'failed'
  repoRoot: string | null
  venvDir: string | null
  dataDir: string | null
  launcherCmd: string | null
  error: string | null
  logs: Array<{ line: string; stream?: 'stdout' | 'stderr' }>
}

const INITIAL: BootstrapStateModel = {
  status: 'idle',
  repoRoot: null,
  venvDir: null,
  dataDir: null,
  launcherCmd: null,
  error: null,
  logs: []
}

// ---------------------------------------------------------------------------
// Atoms
// ---------------------------------------------------------------------------

export type Route = 'welcome' | 'location' | 'progress' | 'success' | 'failure'

export const $route = atom<Route>('welcome')
export const $bootstrap = atom<BootstrapStateModel>(INITIAL)
export const $repo = atom<RepoInfo | null>(null)
export const $logPath = atom<string | null>(null)
/** Non-fatal note shown on the location screen (e.g. a rejected drive pick). */
export const $locationNote = atom<string | null>(null)

// ---------------------------------------------------------------------------
// Tauri event subscription
// ---------------------------------------------------------------------------

interface BootstrapStartedEvent {
  type: 'started'
  script: string
  repoRoot: string
}

interface BootstrapLogEvent {
  type: 'log'
  line: string
  stream?: 'stdout' | 'stderr'
}

interface BootstrapCompleteEvent {
  type: 'complete'
  repoRoot: string
  venvDir: string
  dataDir: string
  launcherCmd?: string
}

interface BootstrapFailedEvent {
  type: 'failed'
  error: string
}

type BootstrapEvent =
  | BootstrapStartedEvent
  | BootstrapLogEvent
  | BootstrapCompleteEvent
  | BootstrapFailedEvent

let unlisten: UnlistenFn | null = null

export async function initialize(): Promise<void> {
  if (unlisten) {return}

  const fake = fakeMode()

  if (fake) {
    unlisten = () => {}
    $logPath.set('%LOCALAPPDATA%\\hermes\\logs\\north-forge-setup.log')
    $repo.set(FAKE_REPO)

    return
  }

  try {
    const [logPath, repo] = await Promise.all([
      invoke<string>('get_log_path'),
      invoke<RepoInfo>('detect_repo')
    ])

    $logPath.set(logPath)
    $repo.set(repo)
  } catch (err) {
    console.warn('failed to fetch setup context', err)
  }

  unlisten = await listen<BootstrapEvent>('bootstrap', (event) => {
    const payload = event.payload
    const cur = $bootstrap.get()

    switch (payload.type) {
      case 'started': {
        $bootstrap.set({ ...INITIAL, status: 'running', repoRoot: payload.repoRoot })
        $route.set('progress')

        break
      }

      case 'log': {
        const logs = [...cur.logs, { line: payload.line, stream: payload.stream }]
        // Keep the rolling buffer bounded so a long install doesn't OOM the UI.
        const trimmed = logs.length > 4000 ? logs.slice(-4000) : logs
        $bootstrap.set({ ...cur, logs: trimmed })

        break
      }

      case 'complete': {
        $bootstrap.set({
          ...cur,
          status: 'completed',
          repoRoot: payload.repoRoot,
          venvDir: payload.venvDir,
          dataDir: payload.dataDir,
          launcherCmd: payload.launcherCmd ?? null
        })
        $route.set('success')

        break
      }

      case 'failed': {
        $bootstrap.set({ ...cur, status: 'failed', error: payload.error })
        $route.set('failure')

        break
      }
    }
  })
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/** Re-run auto-detection (env → installer-relative → drive scan). */
export async function refreshRepo(): Promise<void> {
  if (fakeMode()) {return}

  try {
    $repo.set(await invoke<RepoInfo>('detect_repo'))
  } catch (err) {
    console.warn('detect_repo failed', err)
  }
}

/** Accept a folder/drive the user picked. Returns an error string, or null on success. */
export async function chooseLocation(path: string): Promise<string | null> {
  $locationNote.set(null)

  if (fakeMode()) {
    $repo.set({ ...FAKE_REPO, repoRoot: path, source: 'picked' })

    return null
  }

  try {
    const repo = await invoke<RepoInfo>('set_repo_root', { path })
    $repo.set(repo)

    return null
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    $locationNote.set(msg)

    return msg
  }
}

export async function startInstall(): Promise<void> {
  const fake = fakeMode()

  if (fake) {
    void runFakeBoot(fake === 'failure' ? 'failure' : 'install')

    return
  }

  $bootstrap.set({ ...INITIAL, status: 'running' })
  $route.set('progress')
  await invoke('start_bootstrap', {
    args: { repo_root: $repo.get()?.repoRoot ?? null }
  })
}

export async function cancelInstall(): Promise<void> {
  if (fakeMode()) {
    fakeCancelled = true

    return
  }

  await invoke('cancel_bootstrap')
}

export async function launchNorthForge(): Promise<void> {
  if (fakeMode()) {throw new Error('Preview mode — launching is disabled.')}
  const repoRoot = $bootstrap.get().repoRoot ?? $repo.get()?.repoRoot

  if (!repoRoot) {throw new Error('no checkout resolved')}
  await invoke('launch_north_forge', { repoRoot })
}

export async function openLogDir(): Promise<void> {
  if (fakeMode()) {return}
  await invoke('open_log_dir')
}

// ---------------------------------------------------------------------------
// Dev-only isolated preview ("fake boot")
//
//   ?fake=install   welcome → location → [ INSTALL ] → success
//   ?fake=failure   a bootstrap that fails partway
// Gated on import.meta.env.DEV → stripped from the shipped Tauri bundle.
// ---------------------------------------------------------------------------

type FakeMode = 'install' | 'failure'

function fakeMode(): FakeMode | null {
  if (!import.meta.env.DEV || typeof window === 'undefined') {return null}
  const v = new URLSearchParams(window.location.search).get('fake')

  return v === 'install' || v === 'failure' ? v : null
}

const FAKE_REPO: RepoInfo = {
  repoRoot: 'E:\\north-forge-agent',
  parent: 'E:\\',
  leaf: 'north-forge-agent',
  venvDir: 'E:\\north-forge-agent-venv',
  dataDir: 'E:\\north-forge-agent-data',
  bootstrapScript: 'E:\\north-forge-agent\\scripts\\bootstrap-north-forge.ps1',
  launcherCmd: 'E:\\north-forge-agent\\north-forge.cmd',
  bootstrapped: false,
  onSystemDrive: false,
  source: 'scan',
  drives: [
    { letter: 'C:', path: 'C:\\', isSystem: true, hasCheckout: false, checkoutPath: null },
    { letter: 'E:', path: 'E:\\', isSystem: false, hasCheckout: true, checkoutPath: 'E:\\north-forge-agent' }
  ]
}

const FAKE_LINES = [
  'North Forge bootstrap',
  '  repo : E:\\north-forge-agent',
  '  venv : E:\\north-forge-agent-venv',
  '  data : E:\\north-forge-agent-data   (HERMES_HOME)',
  'creating venv...',
  'installing North Forge (editable) into the venv - this can take a minute...',
  'Resolved 214 packages',
  'Installed 214 packages in 3.1s',
  '  skin : north-forge  (set as active skin)',
  'READY. venv has \'hermes\' (import ok).'
]

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

let fakeRunning = false
let fakeCancelled = false

async function runFakeBoot(kind: FakeMode): Promise<void> {
  if (fakeRunning) {return}
  fakeRunning = true
  fakeCancelled = false

  try {
    $bootstrap.set({ ...INITIAL, status: 'running', repoRoot: FAKE_REPO.repoRoot })
    $route.set('progress')

    const failAt = kind === 'failure' ? Math.floor(FAKE_LINES.length / 2) : -1

    for (let i = 0; i < FAKE_LINES.length; i++) {
      await sleep(650)

      if (fakeCancelled) {
        $bootstrap.set({ ...$bootstrap.get(), status: 'failed', error: 'Bootstrap cancelled.' })
        $route.set('failure')

        return
      }

      $bootstrap.set({
        ...$bootstrap.get(),
        logs: [...$bootstrap.get().logs, { line: FAKE_LINES[i]!, stream: 'stdout' }]
      })

      if (i === failAt) {
        $bootstrap.set({
          ...$bootstrap.get(),
          status: 'failed',
          error: 'bootstrap-north-forge.ps1 exited with code 1.\n\neditable install failed (exit 1).'
        })
        $route.set('failure')

        return
      }
    }

    $bootstrap.set({
      ...$bootstrap.get(),
      status: 'completed',
      repoRoot: FAKE_REPO.repoRoot,
      venvDir: FAKE_REPO.venvDir,
      dataDir: FAKE_REPO.dataDir,
      launcherCmd: FAKE_REPO.launcherCmd
    })
    $route.set('success')
  } finally {
    fakeRunning = false
  }
}
