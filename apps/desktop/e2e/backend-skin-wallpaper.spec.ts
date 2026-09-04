/**
 * End-to-end coverage for a user skin that deliberately reuses a built-in
 * Desktop theme name. The real Python gateway loads the YAML, resolves the
 * relative image path, publishes gateway.ready, and Electron reads the image
 * before the renderer paints it.
 */
import fs from 'node:fs'
import path from 'node:path'

import type { ElectronApplication, Page } from '@playwright/test'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type Sandbox,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { type MockServer, startMockServer } from './mock-server'
import { allowErrorBanners, collectErrorBanners, expect, test } from './test'

const ONE_PIXEL_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  'base64'
)

let app: ElectronApplication | null = null
let page: Page | null = null
let mock: MockServer | null = null
let sandbox: Sandbox | null = null

async function setUpFixture(): Promise<void> {
  mock = await startMockServer()
  sandbox = createSandbox('backend-skin-wallpaper')
  writeMockProviderConfig(sandbox.hermesHome, mock.url, '  skin: mono')
  writeEnvFile(sandbox.hermesHome)

  const skinsDir = path.join(sandbox.hermesHome, 'skins')
  fs.mkdirSync(skinsDir, { recursive: true })
  fs.writeFileSync(path.join(skinsDir, 'wallpaper.png'), ONE_PIXEL_PNG)
  fs.writeFileSync(
    path.join(skinsDir, 'mono.yaml'),
    [
      'name: mono',
      'colors:',
      "  background: '#ff00ff'",
      "  ui_text: '#00ff00'",
      'background_image: wallpaper.png',
      'background_image_fit: contain',
      'background_image_position: top right',
      "background_overlay: '#10203099'",
      ''
    ].join('\n'),
    'utf8'
  )

  const launched = await launchDesktop(buildAppEnv(sandbox))
  app = launched.app
  page = launched.page

  // Select the built-in theme before reconnecting. gateway.ready then decorates
  // that palette with the backend skin's wallpaper without adopting its colours.
  await page.evaluate(() => {
    window.localStorage.setItem('hermes-desktop-theme-v2', 'mono')
    window.localStorage.setItem('hermes-desktop-mode-v1', 'dark')
  })
  await page.reload({ waitUntil: 'domcontentloaded' })
  await page.waitForSelector('[data-hermes-skin-wallpaper]', {
    state: 'attached',
    timeout: 120_000
  })
}

test.afterAll(async () => {
  if (app) {
    const runningApp = app

    const closed = await Promise.race([
      runningApp.close().then(
        () => true,
        () => true
      ),
      new Promise<boolean>(resolve => setTimeout(() => resolve(false), 15_000))
    ])

    if (!closed && runningApp.process().exitCode === null) {
      runningApp.process().kill()
    }
  }

  await mock?.close()
  sandbox?.cleanup()
})

test('keeps the built-in palette and renders the backend wallpaper metadata', async () => {
  // This spec owns the Electron page lifecycle, so inspect the guard before
  // cleanup rather than asking the shared post-test hook to inspect a closing page.
  allowErrorBanners()
  await setUpFixture()

  const wallpaper = page!.locator('[data-hermes-skin-wallpaper]')
  const image = wallpaper.locator('img')
  const overlay = wallpaper.locator('[data-hermes-skin-wallpaper-overlay]')

  await expect(wallpaper).toBeVisible()
  await expect(image).toHaveAttribute('src', /^data:image\/png;base64,/)
  await expect(image).toHaveCSS('object-fit', 'contain')
  await expect(image).toHaveCSS('object-position', '100% 0%')
  await expect(overlay).toHaveCSS('background-color', 'rgba(16, 32, 48, 0.6)')

  const colours = await page!.evaluate(() => ({
    background: document.documentElement.style.getPropertyValue('--theme-background-seed'),
    foreground: document.documentElement.style.getPropertyValue('--theme-foreground')
  }))

  expect(colours).toEqual({ background: '#0e0e0e', foreground: '#eaeaea' })
  expect(await collectErrorBanners(page)).toEqual([])
})
