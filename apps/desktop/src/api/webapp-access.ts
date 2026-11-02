import { hermesApi, profileScoped } from './client'

// Web access: turn on the Webapp the gateway on THIS machine serves, for other devices, behind Nous
// sign-in. The backend owns every fact here; the pane renders /status and asks for changes.

export type WebAccessMode = 'lan' | 'public'

export interface WebAccessStatus {
  /** `webapp.enabled` in the serving gateway's config: what it serves from its next start. */
  enabled: boolean
  /** What the gateway serves now; `stopped` when none answers. */
  gateway: { reason: null | string; state: 'off' | 'serving' | 'starting' | 'stopped' | 'unavailable'; url: null | string }
  lan_address: null | string
  /** Who the configured web app serves; `local` is a loopback bind, reachable from this computer only. */
  mode: 'local' | null | WebAccessMode
  nous_signed_in: boolean
  port: number
  public_url: string
  /** The newest restart of this gateway exited without replacing it; `pid` is the `restart_pid` that
   *  restart answered, `reason` the last line it wrote (or its exit code). */
  restart_failure?: null | { exit_code: number; pid: number; reason: string }
}

/** Start/stop save the config and restart the gateway; a refused restart keeps the config.
 *  `restart_busy`: a running chat refused it. `restart_error` is the CLI's text, for logs only. */
export interface WebAccessChange extends WebAccessStatus {
  restart_busy?: boolean
  restart_error?: string
  restart_pid?: number
  restart_started: boolean
}

/** Start builds the Webapp bundle first, as `hermes webapp` does; that build waits up to 30 minutes
 *  for another build of it. */
const START_REQUEST_TIMEOUT_MS = 35 * 60_000

export function getWebAccessStatus(): Promise<WebAccessStatus> {
  return hermesApi<WebAccessStatus>({ ...profileScoped(), path: '/api/webapp-access/status' })
}

export function startWebAccess(mode: WebAccessMode, publicUrl?: string): Promise<WebAccessChange> {
  return hermesApi<WebAccessChange>({
    ...profileScoped(),
    body: { mode, public_url: publicUrl },
    method: 'POST',
    path: '/api/webapp-access/start',
    timeoutMs: START_REQUEST_TIMEOUT_MS
  })
}

export function stopWebAccess(): Promise<WebAccessChange> {
  return hermesApi<WebAccessChange>({ ...profileScoped(), method: 'POST', path: '/api/webapp-access/stop' })
}

/** Restart the gateway onto the saved config, and nothing else. */
export function restartWebAccess(): Promise<WebAccessChange> {
  return hermesApi<WebAccessChange>({ ...profileScoped(), method: 'POST', path: '/api/webapp-access/restart' })
}
