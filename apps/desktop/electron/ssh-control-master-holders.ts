// Who may run `ssh -O exit` on a shared ControlMaster socket.
//
// Every SshConnection built for the same scope/host/identity hashes to the
// same ControlPath, and ControlMaster=auto makes a later connection attach to
// whatever master is already listening there. `-O exit` therefore tears down
// the master for every attached connection, not just the caller. A stale
// bootstrap attempt (timed out, superseded, or rolled back) that closes late
// used to kill the master its successor had already attached to and forwarded
// the live backend through (#97264). A connection now registers as a holder of
// its ControlPath from the moment it starts opening, and only the last holder
// to close exits the master; earlier closers just release their claim.

export type ControlMasterCloseAction = 'exit-master' | 'release-only'

export interface ControlMasterHolders {
  acquire(controlPath: string, holder: object): void
  // Drops `holder` and returns what its close should do with the master.
  release(controlPath: string, holder: object): ControlMasterCloseAction
  count(controlPath: string): number
}

// Pure decision: the master belongs to whoever is still attached. Exit it only
// when the closing connection was the last holder of the socket.
export function controlMasterCloseAction(otherHolders: number): ControlMasterCloseAction {
  return otherHolders > 0 ? 'release-only' : 'exit-master'
}

export function createControlMasterHolders(): ControlMasterHolders {
  const holders = new Map<string, Set<object>>()

  return {
    acquire(controlPath, holder) {
      if (!controlPath) {
        return
      }

      const set = holders.get(controlPath) || new Set<object>()
      set.add(holder)
      holders.set(controlPath, set)
    },
    release(controlPath, holder) {
      const set = holders.get(controlPath)

      set?.delete(holder)

      if (set && set.size === 0) {
        holders.delete(controlPath)
      }

      return controlMasterCloseAction(set?.size || 0)
    },
    count(controlPath) {
      return holders.get(controlPath)?.size || 0
    }
  }
}

// Process-wide: every SshConnection in the Electron main process shares one
// view of which sockets are still in use.
export const sharedControlMasterHolders: ControlMasterHolders = createControlMasterHolders()
