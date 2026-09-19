/**
 * Client-side narrowing of the vault list. The list merges local items with every
 * unlocked password manager (1Password / Bitwarden), so it can run to hundreds of
 * rows; filtering stays in the renderer because `vault.list` returns metadata only
 * (never secrets) and the whole set is already in memory.
 */
export type VaultKindFilter = 'address' | 'all' | 'login' | 'payment'

export interface VaultFilterable {
  kind: string
  label: string
  origin: null | string
  identifier?: null | string
}

const normalize = (value: null | string | undefined) => (value ?? '').trim().toLocaleLowerCase()

export function filterVaultItems<T extends VaultFilterable>(items: readonly T[], query: string, kind: VaultKindFilter): T[] {
  const needle = normalize(query)

  return items.filter(item => {
    if (kind !== 'all' && item.kind !== kind) {
      return false
    }

    if (!needle) {
      return true
    }

    // Origin matches with or without the scheme so "github" finds https://github.com.
    const origin = normalize(item.origin).replace(/^https?:\/\//, '')

    return normalize(item.label).includes(needle) || normalize(item.identifier).includes(needle) || origin.includes(needle)
  })
}
