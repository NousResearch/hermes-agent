export const STORAGE_KEY = 'hermes.desktop.preparedSubmissions.v1'

let browserOwner: string | undefined
const legacyOwners = new WeakMap<object, string>()

export async function journalOwner(): Promise<string> {
  const native = window.hermesDesktop?.preparedSubmissions

  if (native?.owner) {return native.owner()}

  // Older bridges cannot transfer recovery ownership. A page-local identity
  // still isolates new intents; retained drafts require explicit recovery.
  if (native) {
    let owner = legacyOwners.get(native)

    if (!owner) {owner = crypto.randomUUID(); legacyOwners.set(native, owner)}

    return owner
  }

  return browserOwner ??= crypto.randomUUID()
}

export async function readJournal<T>(): Promise<Record<string, T>> {
  const native = window.hermesDesktop?.preparedSubmissions
  const parsed: unknown = JSON.parse(native ? await native.read() : localStorage.getItem(STORAGE_KEY) || '{}')

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {throw new Error('Invalid prepared submission journal')}

  return parsed as Record<string, T>
}

export async function compareJournal(key: string, expected: string | null, entry: string | null, transfer = false, freshGenerated = false): Promise<boolean> {
  const native = window.hermesDesktop?.preparedSubmissions

  if (native?.compareSend) {return native.compareSend(key, expected, entry)}

  if (native) {
    // Only a first write under a freshly renderer-generated UUID is compatible
    // with the old bridge. Never emulate CAS for caller IDs, recovery, updates
    // or deletion; never fall back after a real CAS refusal or I/O failure.
    if (!freshGenerated || transfer || expected !== null || entry === null) {throw new Error('Atomic draft storage unavailable; update Desktop')}
    const current = (await readJournal<unknown>())[key]

    if (current !== undefined) {return false}
    await native.update(key, entry)

    return true
  }

  if (!navigator.locks) {throw new Error('Atomic draft storage unavailable in this browser')}

  return navigator.locks.request(STORAGE_KEY, () => {
    const journal = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    const current = Object.hasOwn(journal, key) ? JSON.stringify(journal[key]) : null

    if (current !== expected) {return false}

    if (entry === null) {delete journal[key]}
    else {Object.defineProperty(journal, key, { value: JSON.parse(entry), enumerable: true, configurable: true })}

    localStorage.setItem(STORAGE_KEY, JSON.stringify(journal))

    return true
  })
}
