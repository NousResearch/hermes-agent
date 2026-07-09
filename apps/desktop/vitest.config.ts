import { defineConfig, mergeConfig } from 'vitest/config'

import viteConfig from './vite.config'

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: 'jsdom',
      // The UI suite lives in src/ as TypeScript only. Scoping the include list
      // keeps vitest away from electron/*.test.ts and scripts/*.test.mjs (which
      // are node:test suites run by `test:desktop:platforms`) and from any
      // compiled .js twins that a stray `tsc -b` may have emitted next to the
      // sources (those double-run tests against stale strings).
      include: ['src/**/*.test.{ts,tsx}'],
      setupFiles: ['src/test/setup.ts']
    }
  })
)
