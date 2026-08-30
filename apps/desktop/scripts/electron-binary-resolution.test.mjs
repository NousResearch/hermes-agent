import fs from 'node:fs'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

vi.mock('../e2e/test', () => ({ installErrorBannerGuard: () => {} }))

const { findElectron } = await import('../e2e/fixtures.ts')

test.skipIf(process.platform !== 'win32')(
  'resolves Electron to a native Windows executable outside npm shims',
  () => {
    const electron = findElectron()
    const normalized = electron.replaceAll('\\', '/').toLowerCase()

    expect(path.win32.isAbsolute(electron)).toBe(true)
    expect(normalized).toMatch(/electron\.exe$/)
    expect(fs.statSync(electron).isFile()).toBe(true)
    expect(normalized).not.toMatch(/^\/c\//)
    expect(normalized).not.toContain('/node_modules/.bin/')
  },
)
