// Demo / fixture mode entry point. Lazy-loaded from main.tsx only when
// isDemoMode() is true (so production builds tree-shake it out). Installs a fake
// preload bridge + gateway — the app then runs with canned data and no backend —
// plus a window.__demo scripting API for e2e tests and demo tooling.
import { installBridge } from './bridge'
import { DEMO_TURN } from './fixtures'
import { control, type DemoTurn, installGateway } from './gateway'

declare global {
  interface Window {
    __demo?: {
      emit: (event: { type: string; session_id?: string; payload?: Record<string, unknown> }) => boolean
      playTurn: (turn?: DemoTurn, sessionId?: string) => Promise<void>
      term: (data: string) => void
    }
  }
}

let installed = false

export function installDemo(): void {
  if (installed) {
    return
  }

  installed = true

  // Returning users skip onboarding; setup.runtime_check (gateway) also reports
  // ready, so the onboarding overlay never reappears once the gateway connects.
  try {
    window.localStorage.setItem('hermes-desktop-onboarded-v1', '1')
  } catch {
    // localStorage unavailable — degrade silently
  }

  installGateway()
  installBridge()

  window.__demo = {
    emit: event => control.emit(event),
    playTurn: (turn = DEMO_TURN, sessionId) => control.playTurn(turn, sessionId),
    term: data => control.term(data)
  }
}
