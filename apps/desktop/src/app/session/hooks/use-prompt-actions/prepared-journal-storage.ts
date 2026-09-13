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

export async function compareJournal(key: string, expected: string | null, entry: string | null, transfer = false): Promise<boolean> {
  const native = window.hermesDesktop?.preparedSubmissions

  if (native?.compareAndSet) {return native.compareAndSet(key, expected, entry)}

  if (native) {
    // Compatibility only for immutable, window-owned UUID slots. Cross-window
    // recovery needs the atomic bridge and cannot degrade to read/update.
    if (transfer) {throw new Error('Atomic draft recovery unavailable; update Desktop')}
    const current = (await readJournal<unknown>())[key]

    if ((current === undefined ? null : JSON.stringify(current)) !== expected) {return false}
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
