import type { ElectronApplication, Page } from '@playwright/test'

export async function waitForPageWindowVisible(app: ElectronApplication, page: Page, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs
  const window = await app.browserWindow(page)

  try {
    while (Date.now() < deadline) {
      if (await window.evaluate(target => target.isVisible())) {
        return
      }

      const remaining = deadline - Date.now()

      if (remaining > 0) {
        await new Promise(resolve => setTimeout(resolve, Math.min(500, remaining)))
      }
    }

    throw new Error(`Page BrowserWindow did not become visible within ${timeoutMs}ms`)
  } finally {
    await window.dispose()
  }
}
