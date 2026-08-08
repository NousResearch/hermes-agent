// Dev-only update-flow simulator. The real update flow quits the app (or a
// remote backend) — you can't iterate on the overlay's applying/terminal
// states against it. This drives the SAME stores the real flow writes
// (`$updateApply` / `$backendUpdateApply` + the overlay atoms) with realistic
// stage transitions and log lines, so every state is inspectable on demand.
//
// From the devtools console:
//
//   __SIMULATE_UPDATE__()                        // client apply → restart
//   __SIMULATE_UPDATE__('backend')               // backend apply → restart
//   __SIMULATE_UPDATE__('client', { end: 'error' })    // land on the error state
//   __SIMULATE_UPDATE__('client', { end: 'manual' })   // CLI-install manual state
//   __SIMULATE_UPDATE__('client', { end: 'guiSkew' })  // Linux GUI-skew state
//   __SIMULATE_UPDATE__('client', { speed: 5 })        // 5× faster
//   __SIMULATE_UPDATE__.stop()                   // reset everything
//
// Registered from `src/debug/index.ts`, which vite aliases to a noop outside
// dev — none of this reaches a shipped renderer.

import type { DesktopUpdateStage } from '@/global'
import {
  $backendUpdateApply,
  $updateApply,
  $updateOverlayOpen,
  $updateOverlayTarget,
  resetUpdateApplyState,
  type UpdateApplyState
} from '@/store/updates'

type SimulateEnd = 'restart' | 'error' | 'manual' | 'guiSkew'

interface SimulateOptions {
  end?: SimulateEnd
  /** Playback speed multiplier — 2 = twice as fast. */
  speed?: number
}

interface SimStep {
  stage: DesktopUpdateStage
  message: string
  holdMs: number
}

// Message rhythm mirrors a real run: stage headline first, then the streamed
// log lines the electron main process forwards (each replaces `message`).
const APPLY_STEPS: SimStep[] = [
  { stage: 'prepare', message: 'Starting update…', holdMs: 900 },
  { stage: 'update', message: 'Updating Hermes (git + dependencies)…', holdMs: 1100 },
  { stage: 'update', message: 'From github.com:NousResearch/hermes-agent', holdMs: 800 },
  { stage: 'update', message: 'Updating a1b2c3d..e4f5a6b — fast-forward', holdMs: 900 },
  { stage: 'update', message: '→ Updating Python dependencies...', holdMs: 1400 },
  { stage: 'update', message: 'Installed 4 packages in 812ms', holdMs: 900 },
  { stage: 'rebuild', message: 'Rebuilding the desktop app…', holdMs: 1200 },
  { stage: 'rebuild', message: 'vite v6.3.5 building for production...', holdMs: 1300 },
  { stage: 'rebuild', message: '✓ 2841 modules transformed.', holdMs: 1000 },
  { stage: 'rebuild', message: '• electron-builder packaging platform=darwin arch=arm64', holdMs: 1600 }
]

const END_STEPS: Record<SimulateEnd, SimStep> = {
  restart: { stage: 'restart', message: 'Installing the updated app and restarting…', holdMs: 0 },
  error: { stage: 'error', message: 'hermes update failed (exit 1). See ~/.hermes/logs/update.log for details.', holdMs: 0 },
  manual: { stage: 'manual', message: 'hermes update --branch main', holdMs: 0 },
  guiSkew: {
    stage: 'guiSkew',
    message: 'Backend updated, but the desktop app package was not changed. Update or reinstall the Hermes desktop app to match.',
    holdMs: 0
  }
}

let runToken = 0

const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))

function applyStateFor(step: SimStep, current: UpdateApplyState): UpdateApplyState {
  const terminal = step.stage === 'error' || step.stage === 'manual' || step.stage === 'guiSkew'

  return {
    applying: !terminal,
    stage: step.stage,
    message: step.message,
    percent: null,
    error: step.stage === 'error' ? 'apply-failed' : null,
    command: step.stage === 'manual' ? step.message : null,
    log: [...current.log, { stage: step.stage, message: step.message, at: Date.now() }].slice(-50)
  }
}

async function simulateUpdate(target: 'client' | 'backend' = 'client', options: SimulateOptions = {}): Promise<void> {
  const token = ++runToken
  const speed = Math.max(0.1, options.speed ?? 1)
  const $apply = target === 'backend' ? $backendUpdateApply : $updateApply

  resetUpdateApplyState()
  $updateOverlayTarget.set(target)
  $updateOverlayOpen.set(true)

  const steps = [...APPLY_STEPS, END_STEPS[options.end ?? 'restart']]

  for (const step of steps) {
    if (token !== runToken) {
      return
    }

    $apply.set(applyStateFor(step, $apply.get()))
    await sleep(step.holdMs / speed)
  }
}

function stopSimulation(): void {
  runToken += 1
  resetUpdateApplyState()
  $updateOverlayOpen.set(false)
}

export function installUpdateSimulator(): void {
  const api = Object.assign(simulateUpdate, { stop: stopSimulation })

  ;(window as unknown as Record<string, unknown>).__SIMULATE_UPDATE__ = api

  // `npm run dev:fake-update` (VITE_SIMULATE_UPDATE=1) auto-runs the client
  // apply loop shortly after boot, so seeing the applying UI is one shell
  // command — no console typing. The value selects the terminal state:
  // 1/restart (default), error, manual, guiSkew, or backend.
  const auto = String(import.meta.env.VITE_SIMULATE_UPDATE ?? '').trim()

  if (!auto) {
    return
  }

  const target = auto === 'backend' ? 'backend' : 'client'
  const end: SimulateEnd = auto === 'error' || auto === 'manual' || auto === 'guiSkew' ? auto : 'restart'

  // Small delay so the shell has mounted before the overlay opens.
  window.setTimeout(() => {
    void simulateUpdate(target, { end })
  }, 1500)
}
