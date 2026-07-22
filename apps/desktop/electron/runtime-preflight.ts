export type BootstrapRuntimeStatus = 'absent' | 'ready' | 'repair'

export function bootstrapRuntimeStatus(options: {
  marker: unknown
  markerSchemaVersion: number
  runtimeUsable(): boolean
}): BootstrapRuntimeStatus {
  const marker = options.marker

  if (!marker || typeof marker !== 'object') {
    return 'absent'
  }

  const { pinnedCommit, schemaVersion } = marker as { pinnedCommit?: unknown; schemaVersion?: unknown }

  if (schemaVersion !== options.markerSchemaVersion || typeof pinnedCommit !== 'string' || pinnedCommit.length < 7) {
    return 'absent'
  }

  return options.runtimeUsable() ? 'ready' : 'repair'
}
