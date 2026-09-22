import { getTerminalBackends, selectTerminalBackend } from '@/hermes'
import { translateNow } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'
import { requestRoute } from '@/store/recovery-requests'
import type { TerminalBackendInfo } from '@/types/hermes'

/**
 * Proactive warning when the selected terminal backend (Docker, SSH, …)
 * fails its probe. Without this the only signal is a "Needs setup" pill inside
 * Skills → Tools → Terminal, so the user learns that shell commands cannot run
 * only when a tool call fails. One toast per boot, never for the Local backend
 * (it needs no probe), and silent when the probe request itself fails — that
 * is an older/remote backend, not a broken terminal.
 */

const TERMINAL_TOOLSET_ROUTE = '/capabilities?tab=toolsets'
const TOAST_ID = 'terminal-backend-unavailable'

/** The active backend row when it is selected but not ready, else null. */
export function unavailableTerminalBackend(backends: readonly TerminalBackendInfo[]): TerminalBackendInfo | null {
  const active = backends.find(backend => backend.active)

  return active && active.name !== 'local' && active.status !== 'ready' ? active : null
}

export async function warnIfTerminalBackendUnavailable(): Promise<boolean> {
  let backends: TerminalBackendInfo[]

  try {
    backends = (await getTerminalBackends()).backends
  } catch {
    return false
  }

  const broken = unavailableTerminalBackend(backends)

  if (!broken) {
    return false
  }

  const copy = 'settings.toolsets.terminalBackend'

  notify({
    id: TOAST_ID,
    kind: 'warning',
    title: translateNow(`${copy}.unavailableTitle`),
    message: translateNow(`${copy}.unavailableMessage`, broken.label),
    detail: broken.detail || undefined,
    action: { label: translateNow(`${copy}.openBackendSettings`), onClick: () => requestRoute(TERMINAL_TOOLSET_ROUTE) },
    secondaryAction: {
      label: translateNow(`${copy}.useLocal`),
      onClick: () =>
        void selectTerminalBackend('local')
          .then(() => notify({ kind: 'success', message: translateNow(`${copy}.switchedToLocal`) }))
          .catch(err => notifyError(err, translateNow(`${copy}.failedSelect`, 'local')))
    }
  })

  return true
}
