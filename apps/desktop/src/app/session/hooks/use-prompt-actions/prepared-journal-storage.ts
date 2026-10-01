export const STORAGE_KEY = 'hermes.desktop.preparedSubmissions.v1'

let browserOwner: string | undefined

export async function journalOwner(): Promise<string> {
  const native = window.hermesDesktop?.preparedSubmissions

  if (native?.owner) {return native.owner()}

  return browserOwner ??= crypto.randomUUID()
}

export async function readJournal<T>(): Promise<Record<string, T>> {
  const native = window.hermesDesktop?.preparedSubmissions
  const parsed: unknown = JSON.parse(native ? await native.read() : localStorage.getItem(STORAGE_KEY) || '{}')

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {throw new Error('Invalid prepared submission journal')}

  return parsed as Record<string, T>
}

export async function compareJournal(key: string, expected: string | null, entry: string | null): Promise<boolean> {
  const native = window.hermesDesktop?.preparedSubmissions

  if (native) {
    // The bridge's atomic compare-send is the only native write path. Never emulate
    // it, and never fall back after a real refusal or I/O failure.
    if (!native.compareSend) {throw new Error('Atomic draft storage unavailable; update Desktop')}

    return native.compareSend(key, expected, entry)
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
