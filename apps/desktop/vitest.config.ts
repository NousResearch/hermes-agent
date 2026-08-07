import type { TestProjectConfiguration } from 'vitest/config';
import { defineConfig } from 'vitest/config'

const reactUi: TestProjectConfiguration = {
  extends: './vite.config.ts',
  test: {
    name: 'ui',
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    globals: true,
    // The first test in each file pays jsdom env init + full module transform,
    // which can exceed vitest's 5000ms default under CI/load. 15s gives the
    // cold start headroom without masking genuinely hung tests.
    testTimeout: 15_000
  }
}

const electronNative: TestProjectConfiguration = {
  test: {
    name: 'electron',
    environment: 'node',
    include: ['electron/**/*.test.ts', 'scripts/**.test.{ts,mjs}'],
    // Native Git/process tests can exceed Vitest's 5000ms default on loaded
    // workstations even when the underlying subprocess completes normally.
    // Keep a finite ceiling while allowing the same cold-start headroom as UI.
    testTimeout: 15_000,
    // These tests launch real Git and other native child processes. Running
    // files in parallel can exhaust process/I/O headroom and create false
    // timeouts, so keep files serial while tests within each file stay normal.
    fileParallelism: false
  }
}

export default defineConfig({
  test: {
    projects: [reactUi, electronNative]
  }
})
