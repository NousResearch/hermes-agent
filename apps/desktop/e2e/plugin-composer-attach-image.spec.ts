import fs from 'node:fs'
import path from 'node:path'

import { setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGOQi1gAAAGuARekSxIOAAAAAElFTkSuQmCC'

const PLUGIN = `
import { COMPOSER_AREAS, host } from '@hermes/plugin-sdk'
import { jsx } from 'react/jsx-runtime'
import { useState } from 'react'
const png = Uint8Array.from(atob('${PNG}'), char => char.charCodeAt(0))
function AttachSketch() {
  const [result, setResult] = useState('Test invalid sketch')
  return jsx('button', { type: 'button', children: result, onClick: async () => {
    const draft = host.composer.captureDraft('new')
    const invalid = result === 'Test invalid sketch'
    const bytes = invalid ? png.slice(0, 24) : png
    const attached = draft && await host.composer.attachImage(draft, new Blob([bytes], { type: 'image/png' }), { filename: 'test-sketch.png' })
    setResult(attached ? 'Sketch attached' : invalid ? 'Attach valid sketch' : 'Sketch not attached')
  } })
}
export default { id: 'attach-test-sketch', name: 'Attach test sketch', register(ctx) {
  ctx.register({ id: 'sketch', area: COMPOSER_AREAS.leading, render: () => jsx(AttachSketch, {}) })
} }
`

test('a plugin can attach a PNG to an unsent Desktop draft through the public SDK', async () => {
  test.setTimeout(120_000)
  const fixture = await setupMockBackend()
  const { page, sandbox } = fixture

  try {
    await waitForAppReady(fixture)
    const plugins = path.join(sandbox.hermesHome, 'desktop-plugins')
    const staging = path.join(sandbox.hermesHome, 'sketch-staging')
    fs.mkdirSync(plugins, { recursive: true })
    fs.mkdirSync(staging)
    fs.writeFileSync(path.join(staging, 'plugin.js'), PLUGIN)
    fs.renameSync(staging, path.join(plugins, 'attach-test-sketch'))

    const text = 'Leave the draft text alone'
    const editor = page.locator('[data-slot="composer-rich-input"]:visible').first()
    await editor.fill(text)
    const images = path.join(sandbox.userDataDir, 'composer-images')
    await page.getByRole('button', { name: 'Test invalid sketch', exact: true }).click()
    await expect(page.getByRole('button', { name: 'Attach valid sketch', exact: true })).toBeVisible()
    expect(fs.existsSync(images) ? fs.readdirSync(images) : []).toEqual([])
    await page.getByRole('button', { name: 'Attach valid sketch', exact: true }).click()
    await expect(page.getByRole('button', { name: 'Sketch attached', exact: true })).toBeVisible()
    await expect(editor).toHaveText(text)
    await expect(page.locator('[data-slot="aui_thread-viewport"]').getByText(text, { exact: true })).toHaveCount(0)
    await expect(page.locator('[data-slot="composer-attachments"]').getByText(/^test-sketch_.*\.png$/)).toBeVisible()
    const files = fs.readdirSync(images)
    expect(files).toHaveLength(1)
    expect(fs.readFileSync(path.join(images, files[0]!)).equals(Buffer.from(PNG, 'base64'))).toBe(true)
  } finally {
    await fixture.cleanup()
  }
})
