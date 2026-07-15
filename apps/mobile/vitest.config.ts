import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vitest/config'

// Resolve a path relative to this config file.
const r = (p: string) => fileURLToPath(new URL(p, import.meta.url))

// Bridge unit tests. They import the real bridge modules (which pull in the
// @capacitor/* plugins); those are mocked per-test with vi.mock. A node
// environment is enough — the logic under test is pure request/URL shaping.
export default defineConfig({
  resolve: {
    alias: {
      '@': r('../desktop/src'),
      '@hermes/shared': r('../shared/src/index.ts'),
      '~mobile': r('./src/mobile'),
      '~bridge': r('./src/bridge'),
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
