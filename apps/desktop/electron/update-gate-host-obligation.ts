// Host update→restart obligation record, read by the Desktop update gate.
//
// `hermes update` arms a HOST-scoped restart obligation before it touches the
// fleet (hermes_cli/update_cmd_fleet.py::_write_fleet_restart_pending_marker)
// and discharges it only after post-restart verification proves the whole
// fleet is on the pulled code (update_cmd_fleet.py::_verify_fleet_after_update
// → _clear_fleet_restart_pending_marker). The record lives beside the host
// rendezvous records in the gateway-locks state dir — the one cross-profile
// state root the tree has (hermes_cli/update_host_obligation.py).
//
// An EXTERNAL update (a maintenance script, a second profile's CLI, the
// gateway's own /update) never touches this Desktop's in-process
// `updateInFlight` flag, and the live-update marker it holds is released
// before the fleet-restart tail (#126177): in that window a Desktop backend
// respawn raced the bouncing gateway and aborted with "Hermes Desktop is
// quitting." This record is the cross-process signal that covers it — present
// and not discharged means the restart is still owed and the gate stays
// closed.
//
// Mirrors `hostRendezvousDirectory` (host-published-token.ts) for the
// directory and hermes_cli/update_host_obligation.py for the record shape:
// a `restarted` object means the restart already happened; anything else
// present means the obligation is still live. A record whose terms cannot be
// read (corrupt, foreign version) stays live — fail closed, same contract as
// the Python reader (`host_obligation_present`).

import fs from 'fs'
import path from 'path'

import { hostRendezvousDirectory } from './host-published-token'

export const HOST_UPDATE_RESTART_RECORD = 'host-update-restart.json'

export interface HostRendezvousEnv {
  home: string
  lockDir?: string
  platform: string
  stateHome?: string
}

export interface HostObligationIo {
  exists: (target: string) => boolean
  readFile: (target: string) => string
}

/**
 * True when the host still owes the fleet a restart onto the pulled code.
 *
 * Never throws: an unreadable state dir means no signal (the gate falls back
 * to the marker/in-flight/hand-off signals), while an unreadable RECORD means
 * the restart terms are unknown and the gate stays closed.
 */
export function hostFleetRestartPending(
  env: HostRendezvousEnv,
  io: HostObligationIo = {
    exists: target => fs.existsSync(target),
    readFile: target => fs.readFileSync(target, 'utf8')
  }
): boolean {
  const record = path.join(hostRendezvousDirectory(env), HOST_UPDATE_RESTART_RECORD)

  let text: string

  try {
    text = io.readFile(record)
  } catch {
    // Absent (nothing owed) and unreadable-but-present are different states
    // for the gate: nothing owed opens it, unknown terms must not.
    return io.exists(record)
  }

  let parsed: unknown

  try {
    parsed = JSON.parse(text)
  } catch {
    return true
  }

  // Discharged only by a recorded restart proof; a version we do not
  // understand is an obligation whose terms are unknown — still owed.
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return true
  }

  return (parsed as Record<string, unknown>).restarted === undefined
}
