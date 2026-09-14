import fs from 'node:fs'
import path from 'node:path'

import { defineConfig, devices } from '@playwright/test'

import {
  CROSS_ENGINE_HERMES_HOME,
  CROSS_ENGINE_PORT,
} from './e2e/dashboard-cross-engine-home'

const repoRoot = path.resolve(import.meta.dirname, '..', '..')
const python = process.env.HERMES_PYTHON?.trim() || path.join(repoRoot, '.venv', 'Scripts', 'python.exe')
const edgeCandidates = [
  process.env.MSEDGE_EXECUTABLE_PATH?.trim(),
  process.env.ProgramFiles ? path.join(process.env.ProgramFiles, 'Microsoft', 'Edge', 'Application', 'msedge.exe') : null,
  process.env['ProgramFiles(x86)'] ? path.join(process.env['ProgramFiles(x86)'], 'Microsoft', 'Edge', 'Application', 'msedge.exe') : null,
  '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  '/usr/bin/microsoft-edge',
  '/usr/bin/microsoft-edge-stable',
].filter((candidate): candidate is string => Boolean(candidate))
const hasEdge = edgeCandidates.some(candidate => fs.existsSync(candidate))

export default defineConfig({
  testDir: './e2e',
  testMatch: 'dashboard-cross-engine.spec.ts',
  globalSetup: './e2e/dashboard-cross-engine-global-setup.ts',
  globalTeardown: './e2e/dashboard-cross-engine-global-teardown.ts',
  timeout: 60_000,
  fullyParallel: true,
  reporter: [['line']],
  use: {
    baseURL: `http://127.0.0.1:${CROSS_ENGINE_PORT}`,
    trace: 'on-first-retry',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'], browserName: 'chromium' } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'], browserName: 'firefox' } },
    ...(hasEdge ? [{ name: 'edge', use: { ...devices['Desktop Chrome'], browserName: 'chromium' as const, channel: 'msedge' as const } }] : []),
  ],
  webServer: {
    command: `"${python}" -m hermes_cli.main dashboard --host 127.0.0.1 --port ${CROSS_ENGINE_PORT} --no-open --skip-build`,
    cwd: repoRoot,
    env: {
      HERMES_HOME: CROSS_ENGINE_HERMES_HOME,
      HERMES_WEB_DIST: path.join(repoRoot, 'hermes_cli', 'web_dist'),
    },
    url: `http://127.0.0.1:${CROSS_ENGINE_PORT}/api/health`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
})
