import { ENTRY_PREFIX, type PreparedSubmission } from './prepared-submissions'

// The browser journal as localStorage holds it: one key per entry.
const storedKeys = () =>
  Array.from({ length: window.localStorage.length }, (_, index) => window.localStorage.key(index)!)

const removeStoredKeys = (remove: (key: string) => boolean) => {
  for (const key of storedKeys().filter(remove)) {
    window.localStorage.removeItem(key)
  }
}

/** The browser journal's entries, by journal key. */
export function storedJournal(): Record<string, PreparedSubmission> {
  return Object.fromEntries(
    storedKeys()
      .filter(key => key.startsWith(ENTRY_PREFIX))
      .map(key => [key.slice(ENTRY_PREFIX.length), JSON.parse(window.localStorage.getItem(key)!)])
  )
}

export const clearStoredJournal = () => removeStoredKeys(key => key.startsWith(ENTRY_PREFIX))

export const clearAllButStoredJournal = () => removeStoredKeys(key => !key.startsWith(ENTRY_PREFIX))

// Chromium gives each renderer its own cached copy of the origin's localStorage and propagates
// writes asynchronously, last writer wins per key: each window writes against its own view.
export function storageView(shared: Map<string, string>) {
  const local = new Map(shared)
  const written = new Map<string, null | string>()

  const storage = {
    get length() {
      return local.size
    },
    key: (index: number) => [...local.keys()][index] ?? null,
    getItem: (key: string) => local.get(key) ?? null,
    setItem: (key: string, value: string) => {
      local.set(key, value)
      written.set(key, value)
    },
    removeItem: (key: string) => {
      local.delete(key)
      written.set(key, null)
    },
    clear: () => undefined
  }

  const flush = () => {
    for (const [key, value] of written) {
      if (value === null) {
        shared.delete(key)
      } else {
        shared.set(key, value)
      }
    }
    written.clear()
  }

  return { storage, flush }
}
