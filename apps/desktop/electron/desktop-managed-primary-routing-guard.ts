export interface ManagedPrimaryRoutingGuardDeps {
  managedConnectionUpdates: Map<string, unknown>
  managedConnectionRecoveries: Map<string, unknown>
  managedPrimaryRestoreOwners: Map<string, unknown>
  readManagedSshRecoveryRecords: () => Array<{ connectionId: string }>
}

export function createManagedPrimaryRoutingGuard(deps: ManagedPrimaryRoutingGuardDeps) {
  const {
    managedConnectionUpdates,
    managedConnectionRecoveries,
    managedPrimaryRestoreOwners,
    readManagedSshRecoveryRecords
  } = deps

  function assertCanMutateManagedPrimaryRouting() {
    const durableIds = readManagedSshRecoveryRecords().map(record => record.connectionId)

    const ids = new Set([
      ...managedConnectionUpdates.keys(),
      ...managedConnectionRecoveries.keys(),
      ...managedPrimaryRestoreOwners.keys(),
      ...durableIds
    ])

    if (ids.size > 0) {
      const error: any = new Error(
        `Primary connection routing cannot change while managed SSH update recovery is pending for ${[...ids].join(', ')}.`
      )

      error.code = 'managed-update-in-progress'
      throw error
    }
  }

  return assertCanMutateManagedPrimaryRouting
}
