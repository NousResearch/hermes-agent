import type { DesktopDiagnosticInput } from '@/global'

export function emitDesktopDiagnostic(payload: DesktopDiagnosticInput): void {
  if (typeof window === 'undefined') {
    return
  }

  void window.hermesDesktop?.diagnostics?.event(payload).catch(() => undefined)
}

export function sendDesktopHeartbeat(payload: Record<string, unknown> = {}): void {
  if (typeof window === 'undefined') {
    return
  }

  void window.hermesDesktop?.diagnostics?.heartbeat(payload).catch(() => undefined)
}
