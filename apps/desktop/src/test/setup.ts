import { vi } from 'vitest'

// jsdom does not provide CSS.escape, but Chromium/Electron does. Components such
// as the thread timeline use it when querying message ids. Keep renderer tests
// aligned with the packaged runtime instead of failing on the test DOM shim.
if (!globalThis.CSS) {
  vi.stubGlobal('CSS', {})
}

if (typeof globalThis.CSS.escape !== 'function') {
  globalThis.CSS.escape = (value: string) =>
    String(value).replace(/[^a-zA-Z0-9_-]/g, char => `\\${char.codePointAt(0)?.toString(16)} `)
}
