import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'czesiek:rail-collapsed:v1'

function read(): Record<string, boolean> {
  try {
    const parsed: unknown = JSON.parse(storedString(KEY) ?? '{}')

    return parsed && typeof parsed === 'object' ? (parsed as Record<string, boolean>) : {}
  } catch {
    return {}
  }
}

/** Which rail cards the user folded, by card id. Global to the window: it is a layout preference. */
export const $railCollapsed = atom<Record<string, boolean>>(read())

export function isRailCardCollapsed(id: string, fallback: boolean): boolean {
  return $railCollapsed.get()[id] ?? fallback
}

export function setRailCardCollapsed(id: string, collapsed: boolean): void {
  const next = { ...$railCollapsed.get(), [id]: collapsed }

  $railCollapsed.set(next)
  persistString(KEY, JSON.stringify(next))
}
