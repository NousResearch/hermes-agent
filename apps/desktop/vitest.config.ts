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
    // React 19.1+ stubs `react`.act to undefined in the production build;
    // @testing-library/react's act() delegates to it and throws
    // "React.act is not a function" when NODE_ENV=production. Force the
    // development build for the test worker so `act` resolves to a real
    // function, regardless of any ambient NODE_ENV (e.g. a shell that
    // inherited NODE_ENV=production from a desktop launch).
    env: { NODE_ENV: 'development' },
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
    include: ['electron/**/*.test.ts', 'scripts/**.test.{ts,mjs}']
  }
}

export default defineConfig({
  test: {
    projects: [reactUi, electronNative]
  }
})
