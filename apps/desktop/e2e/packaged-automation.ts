import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, _electron, type ElectronApplication, type Page } from '@playwright/test'

import { buildAppEnv, createSandbox, PACKAGED_BINARY_PATH, writeEnvFile, writeMockProviderConfig, type Sandbox } from './fixtures'
import { startMockServer, type MockServer, type MockServerOptions } from './mock-server'

// Durable packaged E2E helpers for automation-packaged.spec.ts.
//
// The packaged binary ships no bundled backend: booting it for a REAL
// regression (not just "the dialog opened") requires pointing it at a hermes
// source root that can serve the automation lifecycle RPCs and at a Python
// that can import hermes_cli. Resolve both portably from the environment /
// venv discovery — never a machine-specific committed path. Every run uses a
// fresh disposable home + mock inference only.

export const desktopRoot = path.resolve(import.meta.dirname, '..')
export const repoRoot = path.resolve(desktopRoot, '..', '..')

/** Portably resolve the hermes source root the packaged backend should use. */
export function resolveHermesRoot(): string {
  const override = process.env.HERMES_DESKTOP_HERMES_ROOT
  return override ? path.resolve(override) : repoRoot
}

/** Portably resolve a Python interpreter, mirroring the packaged app's own
 * findPythonForRoot() ladder: explicit override, then venv discovery under the
 * root. The app falls back to system Python itself when both are absent. */
export function resolvePython(root: string): string | undefined {
  const override = process.env.HERMES_DESKTOP_PYTHON
  if (override && fs.existsSync(override)) {
    return override
  }
  const winCandidates = ['.venv', 'venv'].map((v) => path.join(root, v, 'Scripts', 'python.exe'))
  const posixCandidates = ['.venv', 'venv'].map((v) => path.join(root, v, 'bin', 'python'))
  for (const candidate of [...winCandidates, ...posixCandidates]) {
    if (fs.existsSync(candidate)) {
      return candidate
    }
  }
  return undefined
}

/** Extra config.yaml sections the packaged specs share:
 * - approvals off (an aux LLM at the same mock would consume a scripted index and park a turn)
 * - a 5s loop floor (so a capped loop can observe its single tick inside the test window)
 * - title generation off (an unasserted extra main-model call per turn). */
export const PACKAGE_EXTRA_CONFIG = `
approvals:
  mode: "off"
auxiliary:
  title_generation:
    enabled: false
loops:
  min_interval_seconds: 5
`

export interface PackagedReal {
  app: ElectronApplication
  page: Page
  mock: MockServer
  mockUrl: string
  sandbox: Sandbox
  cleanup: () => Promise<void>
}

/** Launch the *packaged* binary with a real backend resolved from the source
 * root + Python (both portable) and mock inference, in a fresh disposable home. */
export async function launchPackagedReal(mockOptions: MockServerOptions = {}): Promise<PackagedReal> {
  const hermesRoot = resolveHermesRoot()
  if (!fs.existsSync(path.join(hermesRoot, 'hermes_cli', 'main.py'))) {
    throw new Error(
      `Packaged backend needs a hermes source root at HERMES_DESKTOP_HERMES_ROOT (resolved ${hermesRoot})`,
    )
  }
  const python = resolvePython(hermesRoot)
  if (!python) {
    throw new Error(
      `No Python resolved for packaged backend. Set HERMES_DESKTOP_PYTHON or provide a venv under ${hermesRoot}`,
    )
  }

  const mock = await startMockServer(mockOptions)
  const sandbox = createSandbox('pkgreal')
  writeMockProviderConfig(sandbox.hermesHome, mock.url, undefined, PACKAGE_EXTRA_CONFIG)
  writeEnvFile(sandbox.hermesHome)

  const env = buildAppEnv(sandbox, {
    HERMES_DESKTOP_HERMES_ROOT: hermesRoot,
    HERMES_DESKTOP_PYTHON: python,
  })
  // The packaged binary must use its own bundled renderer + backend, not a dev checkout.
  delete (env as Record<string, string | undefined>).HERMES_DESKTOP_DEV_SERVER
  delete (env as Record<string, string | undefined>).HERMES_DESKTOP_HERMES

  const app = await _electron.launch({
    executablePath: PACKAGED_BINARY_PATH,
    args: ['--disable-gpu', '--no-sandbox'],
    env,
  })
  const page = await app.firstWindow()

  return {
    app,
    page,
    mock,
    mockUrl: mock.url,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    },
  }
}

/** A real first message activates a session so the composer automation button
 * stops being disabled (the create dialog needs an owning session). */
export async function ensureActiveSession(page: Page, prompt: string): Promise<void> {
  const input = page.locator('[data-slot="composer-rich-input"]').first()
  await input.fill(prompt)
  await input.press('Enter')
  await expect(page.getByText('Hello from the mock inference server! The full boot chain is working.', { exact: true })).toBeVisible({ timeout: 60_000 })
}

export async function openCreateAutomation(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Add files and actions', exact: true }).first().click()
  await page.getByRole('menuitem', { name: /Create automation/ }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
}

export type AutomationType = 'Goal' | 'Loop' | 'Heartbeat'

export async function pickAutomationType(page: Page, type: AutomationType): Promise<void> {
  await page.getByRole('button', { name: type, exact: true }).click()
  await expect(page.getByRole('button', { name: /^(Start|Save|Create)/, exact: true })).toBeVisible()
}

export async function openEditDialog(page: Page, actionsButtonName: string, editMenuitem: string): Promise<void> {
  await page.getByRole('button', { name: actionsButtonName, exact: true }).click()
  await page.getByRole('menuitem', { name: editMenuitem, exact: true }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
}

/** Type unsaved changes inside an open edit dialog, Pause, then assert the dialog and draft survive. */
export async function pausePreservesUnsavedDraft(page: Page, fieldLabel: string, unsavedText: string): Promise<void> {
  const field = page.getByLabel(fieldLabel, { exact: true })
  await expect(field).toBeVisible()
  await field.fill(unsavedText)
  await page.getByRole('button', { name: /Pause/, exact: true }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await expect(page.getByLabel(fieldLabel, { exact: true })).toHaveValue(unsavedText)
}