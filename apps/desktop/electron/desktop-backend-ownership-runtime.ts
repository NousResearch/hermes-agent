import fs from 'node:fs'
import path from 'node:path'

import {
  type BackendOutputTail,
  claimDecision,
  isPidOnlyStartMarker,
  pidOnlyStartMarker,
  REAP_PROBE_TIMEOUT_MS,
  type StartMarkerProbe
} from './backend-claim'
import { backendCommandMatches, createBackendOwnership } from './backend-ownership'
import { createParentStartMarkerResolver } from './parent-process-identity'

export interface DesktopBackendOwnershipRuntimeDeps {
  ownershipPath: string
  isWindows: boolean
  execText: (command: string, args: string[]) => Promise<string>
  processStartMarker: (pid: number, timeoutMs?: number) => Promise<string>
  probeStartMarker: (pid: number) => Promise<StartMarkerProbe>
  forceKillProcessTree: (pid: number) => void
  stopBackendChild: (child: any) => void
  waitForBackendExit: (child: any) => Promise<any>
  rememberLog: (message: string) => void
}

export function createDesktopBackendOwnershipRuntime(deps: DesktopBackendOwnershipRuntimeDeps) {
  const DESKTOP_BACKEND_OWNERSHIP_PATH = deps.ownershipPath
  const IS_WINDOWS = deps.isWindows
  const {
    execText,
    processStartMarker,
    probeStartMarker,
    forceKillProcessTree,
    stopBackendChild,
    waitForBackendExit,
    rememberLog
  } = deps
  let backendOrphanReapPromise = null

  function writeBackendOwnership(contents) {
    fs.mkdirSync(path.dirname(DESKTOP_BACKEND_OWNERSHIP_PATH), { recursive: true })
    const tempPath = `${DESKTOP_BACKEND_OWNERSHIP_PATH}.${process.pid}.tmp`

    try {
      fs.writeFileSync(tempPath, contents, { encoding: 'utf8', mode: 0o600 })
      fs.renameSync(tempPath, DESKTOP_BACKEND_OWNERSHIP_PATH)
    } finally {
      try {
        fs.rmSync(tempPath, { force: true })
      } catch {
        void 0
      }
    }
  }

  // execText and processStartMarker moved to backend-claim.ts (#93608) so the
  // claim/probe policy is testable — including on Windows CI with real
  // PowerShell — without booting Electron. This runtime calls through that module.

  async function backendCommandForPid(pid) {
    try {
      const command = IS_WINDOWS ? 'powershell.exe' : 'ps'

      const args = IS_WINDOWS
        ? [
            '-NoProfile',
            '-NonInteractive',
            '-Command',
            `(Get-CimInstance Win32_Process -Filter 'ProcessId = ${pid}').CommandLine`
          ]
        : ['-p', String(pid), '-o', 'command=']

      return (await execText(command, args)) || null
    } catch {
      return null
    }
  }

  async function processIdentityMatches(identity, timeoutMs: number = 30_000) {
    // Degraded PID-only identity (#93608): the start-marker probe failed while
    // the child was verifiably alive, so only PID liveness can be checked here.
    // backendIdentityMatches layers the command-line check on top before
    // anything destructive relies on the answer.
    if (isPidOnlyStartMarker(identity.startMarker)) {
      try {
        process.kill(identity.pid, 0)

        return true
      } catch (error) {
        const code = (error as NodeJS.ErrnoException | null)?.code

        return code === 'ESRCH' || code === 'ENOENT' ? false : code === 'EPERM' ? true : undefined
      }
    }

    try {
      return (await processStartMarker(identity.pid, timeoutMs)) === identity.startMarker
    } catch (error) {
      return error?.code === 'ENOENT' || error?.code === 'ESRCH' ? false : undefined
    }
  }

  async function backendIdentityMatches(identity) {
    const processMatches = await processIdentityMatches(identity, REAP_PROBE_TIMEOUT_MS)

    if (processMatches !== true) {
      return processMatches
    }

    const command = await backendCommandForPid(identity.pid)

    return command === null ? undefined : backendCommandMatches(command)
  }

  // True when the recorded parent Electron is still running (same PID AND start
  // marker); false when it is gone or its PID was reused; undefined when the
  // ownership record predates parent tracking. Undefined deliberately falls back
  // to the pre-parent reap behaviour so legacy orphan cleanup keeps working.
  async function backendParentMatches(entry) {
    if (!Number.isInteger(entry.parentPid) || typeof entry.parentStartMarker !== 'string' || !entry.parentStartMarker) {
      return undefined
    }

    try {
      return (await processStartMarker(entry.parentPid, REAP_PROBE_TIMEOUT_MS)) === entry.parentStartMarker
    } catch (error) {
      return error?.code === 'ENOENT' || error?.code === 'ESRCH' ? false : undefined
    }
  }

  async function stopOwnedBackend(identity) {
    const matches = await processIdentityMatches(identity, REAP_PROBE_TIMEOUT_MS)

    if (matches === false) {
      return
    }

    if (matches !== true) {
      // Identity probe failed (not confirmed gone): preserve the record so a
      // later launch retries the stop instead of dropping it and leaking the
      // backend. reapOrphans keeps the entry when stop() throws.
      throw new Error(`Could not verify backend PID ${identity.pid} before stopping it.`)
    }

    if (IS_WINDOWS) {
      forceKillProcessTree(identity.pid)
    } else {
      try {
        process.kill(-identity.pid, 'SIGTERM')
      } catch {
        try {
          process.kill(identity.pid, 'SIGTERM')
        } catch {
          return
        }
      }

      const deadline = Date.now() + 1500

      while (Date.now() < deadline) {
        if ((await processIdentityMatches(identity, REAP_PROBE_TIMEOUT_MS)) !== true) {
          return
        }

        await new Promise(resolve => setTimeout(resolve, 50))
      }

      // Revalidate immediately before escalation so PID reuse cannot target a
      // replacement process.
      if ((await processIdentityMatches(identity, REAP_PROBE_TIMEOUT_MS)) === true) {
        try {
          process.kill(-identity.pid, 'SIGKILL')
        } catch {
          process.kill(identity.pid, 'SIGKILL')
        }
      }
    }

    await new Promise(resolve => setTimeout(resolve, 50))
    const remaining = await processIdentityMatches(identity, REAP_PROBE_TIMEOUT_MS)

    if (remaining !== false) {
      throw new Error(`Backend PID ${identity.pid} did not stop cleanly.`)
    }
  }

  const backendOwnership = createBackendOwnership({
    matchesIdentity: backendIdentityMatches,
    matchesParent: backendParentMatches,
    stop: stopOwnedBackend,
    store: {
      read: () => {
        try {
          return fs.readFileSync(DESKTOP_BACKEND_OWNERSHIP_PATH, 'utf8')
        } catch {
          return null
        }
      },
      write: writeBackendOwnership,
      // A corrupt ownership file is moved aside instead of being rewritten
      // away by the reap sweep — its records are the only pointer to any
      // still-running backends it described (#89298).
      quarantine: () => {
        const parked = `${DESKTOP_BACKEND_OWNERSHIP_PATH}.corrupt`

        try {
          fs.renameSync(DESKTOP_BACKEND_OWNERSHIP_PATH, parked)
          rememberLog(`Backend ownership file was unreadable; moved to ${parked}`)
        } catch {
          // Nothing to move (or no permission) — the sweep already skipped.
        }
      }
    }
  })

  const desktopParentStartMarker = createParentStartMarkerResolver({
    load: () => processStartMarker(process.pid),
    onError: error => {
      const detail = error instanceof Error ? error.message : String(error)

      rememberLog(
        `Could not resolve the Desktop process start marker; starting the backend with PID-only parent tracking: ${detail}`
      )
    }
  })

  async function claimBackendChild(child, command, profile, nonce, outputTail: BackendOutputTail | null = null) {
    // Probe/claim policy lives in backend-claim.ts (#93608): a marker probe
    // that fails against a LIVE child degrades to PID-only identity — matching
    // createParentStartMarkerResolver — instead of killing a healthy backend
    // over a flaky Get-Process (PS 5.1 cold starts, #87169). Only a child that
    // actually died keeps the fail-closed throw, now carrying its stderr tail.
    const probe = await probeStartMarker(child.pid)
    const decision = claimDecision(child.exitCode === null && !child.killed, probe)

    if (decision.action === 'fail') {
      stopBackendChild(child)
      await waitForBackendExit(child)
      throw new Error(
        `Hermes backend (PID ${child.pid}) died before its identity could be recorded: ${decision.reason}${outputTail?.describe() ?? ''}`
      )
    }

    let startMarker

    if (decision.action === 'degrade') {
      startMarker = pidOnlyStartMarker(child.pid)
      rememberLog(
        `WARNING: process start marker probe failed for live Hermes backend PID ${child.pid}; ` +
          `claiming with PID-only identity instead of stopping it: ${decision.reason}`
      )
    } else {
      startMarker = decision.startMarker
    }

    try {
      const identity = await backendOwnership.claim({
        command,
        nonce,
        pid: child.pid,
        profile,
        startMarker,
        // Record the spawning Electron so reapOrphans can tell an orphaned
        // backend (parent gone) from one owned by a live instance — a live
        // parent's backend is never reaped (#87295).
        parentPid: process.pid,
        parentStartMarker: await desktopParentStartMarker()
      })

      child.hermesBackendIdentity = identity

      return identity
    } catch (error) {
      stopBackendChild(child)
      await waitForBackendExit(child)
      throw new Error(
        `Could not persist ownership for the Hermes backend: ${error.message}${outputTail?.describe() ?? ''}`
      )
    }
  }

  function releaseBackendChild(child) {
    const identity = child?.hermesBackendIdentity

    if (!identity) {
      return
    }

    try {
      backendOwnership.release(identity)
    } catch (error) {
      rememberLog(`Could not release backend ownership for PID ${identity.pid}: ${error.message}`)
    }
  }

  function reapOrphanedBackendsOnce() {
    if (!backendOrphanReapPromise) {
      backendOrphanReapPromise = backendOwnership
        .reapOrphans()
        .then(pids => {
          if (pids.length) {
            rememberLog(`Reaped orphaned desktop backend PID(s): ${pids.join(', ')}`)
          }
        })
        .catch(error => {
          backendOrphanReapPromise = null
          throw error
        })
    }

    return backendOrphanReapPromise
  }

  return { claimBackendChild, desktopParentStartMarker, reapOrphanedBackendsOnce, releaseBackendChild }
}
