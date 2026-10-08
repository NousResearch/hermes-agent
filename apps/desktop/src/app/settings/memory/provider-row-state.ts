import type { MemoryCatalogProvider, MemoryStatusResponse } from '@/types/hermes'

/** What the provider row should offer for the inspected provider. */
export type MemoryRowStep = 'install' | 'retry' | 'use' | null

export interface MemoryRowState {
  active: string
  /** Featured catalog providers this owner has not installed yet. */
  catalog: MemoryCatalogProvider[]
  entry?: MemoryCatalogProvider
  installed: MemoryStatusResponse['providers']
  isActive: boolean
  isInstalled: boolean
  missing: MemoryStatusResponse['providers']
  /** Catalog entries with no row at all (not even a configured-but-missing one). */
  offered: MemoryCatalogProvider[]
  selected: string
  step: MemoryRowStep
}

/** Pure derivation of the provider row from one owner's `/api/memory` snapshot. */
export function memoryRowState(data: MemoryStatusResponse | undefined, inspected: null | string): MemoryRowState {
  const providers = data?.providers ?? []
  const installed = providers.filter(provider => provider.name !== 'builtin' && provider.status !== 'missing')
  const missing = providers.filter(provider => provider.status === 'missing')

  const catalog = (data?.catalog_providers ?? []).filter(
    entry => entry.featured && !installed.some(provider => provider.name === entry.name)
  )

  const active = data?.active || 'builtin'
  const selected = inspected ?? active
  const entry = catalog.find(provider => provider.name === selected)
  const isActive = selected === active

  const ready =
    selected === 'builtin' || providers.some(provider => provider.name === selected && provider.status === 'ready')

  // One follow-up at a time: install a catalog entry, use a ready provider, or re-check one that is not.
  const step: MemoryRowStep = entry ? 'install' : isActive ? null : ready ? 'use' : 'retry'

  return {
    active,
    catalog,
    entry,
    installed,
    isActive,
    isInstalled: installed.some(provider => provider.name === selected),
    missing,
    offered: catalog.filter(provider => !providers.some(row => row.name === provider.name)),
    selected,
    step
  }
}

/** Display name: provider/catalog title, else the plugin id read as words (`agent-memory` → `Agent Memory`). */
export function memoryProviderLabel(
  name: string,
  data: MemoryStatusResponse | undefined,
  row: MemoryRowState,
  builtin: string,
  pretty: (value: string) => string
): string {
  if (name === 'builtin') {
    return builtin
  }

  return (
    data?.providers.find(provider => provider.name === name)?.title ||
    row.catalog.find(entry => entry.name === name)?.title ||
    pretty(name.replace(/-/g, ' '))
  )
}
