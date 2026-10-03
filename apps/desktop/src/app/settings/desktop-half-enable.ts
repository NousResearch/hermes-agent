import type { PluginRecord } from '@/contrib/plugins-store'

/** Ids of desktop halves the install dialog should turn on. Already-loaded and
 *  broken rows are left alone; a disabled row whose id, package, or name was
 *  just installed is the one the user asked for. */
export function installedDesktopHalfIds(
  records: readonly Pick<PluginRecord, 'id' | 'name' | 'packageName' | 'status'>[],
  names: readonly (string | null | undefined)[]
): string[] {
  const wanted = new Set(names.filter((name): name is string => Boolean(name)))

  if (wanted.size === 0) {
    return []
  }

  const ids: string[] = []

  for (const record of records) {
    if (record.status !== 'disabled') {
      continue
    }

    if ([record.id, record.packageName, record.name].some(alias => alias && wanted.has(alias))) {
      ids.push(record.id)
    }
  }

  return ids
}
