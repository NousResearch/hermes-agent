interface SshBackendRetirementOptions<TOwner extends { pid?: number }, TEntry> {
  getOwner: (scope: string | null | undefined) => TOwner | undefined
  isAlive: (owner: TOwner) => Promise<boolean>
  getEntry: (key: string) => TEntry | undefined
  forgetEntry: (key: string) => Promise<void>
  teardownOwner: (scope: string | null | undefined) => Promise<void>
  withDrain: (key: string, cleanup: () => Promise<void>) => Promise<unknown>
}

/** HTTP failure never authorizes terminating an owner whose PID is alive or unverified. */
export function createSshBackendRetirement<TOwner extends { pid?: number }, TEntry>({
  getOwner, isAlive, getEntry, forgetEntry, teardownOwner, withDrain
}: SshBackendRetirementOptions<TOwner, TEntry>) {
  async function confirmDead(scope: string | null | undefined): Promise<boolean> {
    const owner = getOwner(scope)

    if (!owner) {
      return true
    }

    return Boolean(owner.pid) && !(await isAlive(owner)) && getOwner(scope) === owner
  }

  async function retireIfDead(scope: string | null | undefined): Promise<void> {
    if (await confirmDead(scope)) {
      await teardownOwner(scope)
    }
  }

  async function retirePool(key: string, entry: TEntry | undefined): Promise<void> {
    const owner = getOwner(key)

    await withDrain(key, async () => {
      if (getEntry(key) !== entry || getOwner(key) !== owner || !(await confirmDead(key))) {
        return
      }

      // The PID check yielded; both routing and ownership must still name the captured objects.
      if (getEntry(key) !== entry || getOwner(key) !== owner) {
        return
      }

      await forgetEntry(key)
      await teardownOwner(key)
    })
  }

  return { confirmDead, retireIfDead, retirePool }
}
