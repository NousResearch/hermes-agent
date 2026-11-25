// Host-level backend discovery — the Desktop half of multiplex-only.
//
// Exactly ONE `hermes serve` runs per HOST and multiplexes every profile, so
// Desktop must ATTACH to a backend that is already listening and spawn only
// when the host has none. A backend registered by another profile is still THE
// backend: refusing it is what produced a second process per profile.
//
// Nothing new has to be published for this. The backend already writes a
// machine-root `spawn-ledger.json` entry AFTER its socket binds
// (`hermes_cli/process_identity.py::register_self`, called from
// `web_server.py` with `detail={host, port, profile}`), and an ungated
// loopback backend serves its session token at `GET /`
// (`window.__HERMES_SESSION_TOKEN__`, `web_server_dashboard.py`). The ledger is
// DISCOVERY ONLY — a record is never trusted as proof of a usable backend; the
// HTTP probe and the token handshake are the boundary that validates it.
//
// These helpers are pure / dependency-injected so the decision is testable
// without Electron.

/** Ledger record for a backend that might be attachable. */
export interface HostBackendRecord {
  createTime: number | null
  /** Bind host as recorded; `0.0.0.0`/`::`/empty all dial back on loopback. */
  host: string
  pid: number
  port: number
  profile: string
  purpose: string
  registeredAt: number
}

/** Filename the CLI writes under the machine Hermes root. */
export const SPAWN_LEDGER_FILENAME = 'spawn-ledger.json'

/** Ledger purposes that denote a JSON-RPC/WebSocket backend we can attach to. */
const ATTACHABLE_PURPOSES = new Set(['dashboard', 'serve'])

/** Bind hosts reachable from this machine over loopback. */
const LOOPBACK_DIALABLE = new Set(['', '0.0.0.0', '127.0.0.1', '::', '::1', 'localhost'])

function asInteger(value: unknown): number | null {
  return Number.isInteger(value) ? Number(value) : null
}

/**
 * Attachable backend records from the raw `spawn-ledger.json` text.
 *
 * A record without a bound port predates the structured detail (or belongs to
 * a purpose that never binds) and is skipped: a port is the whole point.
 * A record marked `isolated` (`hermes serve --isolated`, e.g. the backend
 * another machine's Desktop spawned here over SSH) opted out of the host
 * singleton and belongs to that client, so it is skipped too; the CLI's
 * `_attach_to_host_backend` honours the same flag.
 * Unreadable/corrupt JSON yields `[]` — discovery degrades to "spawn", never
 * to a wrong attach.
 */
export function parseSpawnLedger(contents: unknown): HostBackendRecord[] {
  let parsed: unknown

  try {
    parsed = JSON.parse(String(contents ?? ''))
  } catch {
    return []
  }

  if (!Array.isArray(parsed)) {
    return []
  }

  const records: HostBackendRecord[] = []

  for (const value of parsed) {
    if (!value || typeof value !== 'object') {
      continue
    }

    const entry = value as Record<string, unknown>
    const pid = asInteger(entry.pid)
    const port = asInteger(entry.port)
    const purpose = String(entry.purpose ?? '')
    const host = String(entry.host ?? '')

    if (
      pid === null ||
      pid <= 0 ||
      port === null ||
      port <= 0 ||
      port > 65535 ||
      entry.isolated === true ||
      !ATTACHABLE_PURPOSES.has(purpose) ||
      !LOOPBACK_DIALABLE.has(host.toLowerCase())
    ) {
      continue
    }

    records.push({
      createTime: typeof entry.create_time === 'number' ? entry.create_time : null,
      host,
      pid,
      port,
      profile: String(entry.profile ?? ''),
      purpose,
      registeredAt: typeof entry.registered_at === 'number' ? entry.registered_at : 0
    })
  }

  return records
}
