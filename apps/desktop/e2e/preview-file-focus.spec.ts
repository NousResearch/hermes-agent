import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { _electron, expect, test } from '@playwright/test'
import { createServer } from 'vite'

import { resolveElectronBinary } from './electron-binary'

// Real preview components and native clipboard shortcuts, without a gateway,
// provider credentials, or access to the developer's Hermes profile.
test('file previews keep native selection, scrolling and clipboard inside the focused surface', async () => {
  const root = path.resolve(import.meta.dirname, '..')
  const userData = await mkdtemp(path.join(tmpdir(), 'hermes-preview-focus-'))

  const server = await createServer({
    configFile: path.join(root, 'vite.config.ts'),
    root,
    logLevel: 'warn',
    server: { host: '127.0.0.1', port: 0 }
  })

  let app: Awaited<ReturnType<typeof _electron.launch>> | undefined

  try {
    await server.listen()
    const address = server.httpServer!.address()

    if (!address || typeof address === 'string') {
      throw new Error('Vite did not bind a TCP port')
    }

    const url = `http://127.0.0.1:${address.port}/e2e/fixtures/preview-file-focus/index.html`
    app = await _electron.launch({
      executablePath: resolveElectronBinary([root, path.resolve(root, '../..')]),
      args: [path.join(root, 'e2e/fixtures/preview-file-focus/main.cjs'), url, `--user-data-dir=${userData}`]
    })
    const page = await app.firstWindow()
    const composer = page.getByRole('textbox', { name: 'Composer' })
    const source = page.locator('.preview-source-code pre').first()
    const mod = process.platform === 'darwin' ? 'Meta' : 'Control'
    const text = Array.from({ length: 1600 }, (_, n) => `source line ${n + 1}`).join('\n') + '\nrevision 0'

    const previousClipboard = await app.evaluateHandle(async ({ clipboard, ClipboardItem }) =>
      Promise.all(
        (await clipboard.read()).map(
          async item =>
            new ClipboardItem(
              Object.fromEntries(await Promise.all(item.types.map(async type => [type, await item.getType(type)])))
            )
        )
      )
    )

    try {
      await source.waitFor()
      await composer.click()
      await source.click({ position: { x: 50, y: 8 } })
      await page.keyboard.press(`${mod}+a`)
      await page.keyboard.press(`${mod}+c`)
      await expect.poll(() => app!.evaluate(({ clipboard }) => clipboard.readText())).toBe(text)
      expect(await page.evaluate(() => document.activeElement?.closest('[data-preview-text]') !== null)).toBe(true)

      await page.keyboard.press('ArrowDown')
      await expect.poll(() => page.locator('[data-preview-text]').evaluate(el => el.scrollTop)).toBeGreaterThan(0)
      await composer.click()
      await page.evaluate(() => window.dispatchEvent(new Event('preview-test-reload')))
      await expect(composer).toBeFocused()
      await page.keyboard.press(`${mod}+a`)
      await page.keyboard.press(`${mod}+c`)
      await expect.poll(() => app!.evaluate(({ clipboard }) => clipboard.readText())).toBe('chat draft')

      await page.getByRole('button', { name: 'Show Markdown' }).click()
      await page.getByText('Rendered file body').click()
      await page.keyboard.press(`${mod}+a`)
      await page.keyboard.press(`${mod}+c`)
      const markdown = await app.evaluate(({ clipboard }) => clipboard.readText())
      expect(markdown).toContain('Markdown file')
      expect(markdown).toContain('Rendered file body')
      expect(markdown).not.toContain('Chat transcript')
      expect(markdown).not.toContain('chat draft')

      await page.getByRole('button', { name: 'Show source' }).click()
      await source.click({ position: { x: 50, y: 8 } })
      await page.keyboard.press(`${mod}+a`)
      await page.keyboard.press(`${mod}+c`)
      await expect
        .poll(() => app!.evaluate(({ clipboard }) => clipboard.readText()))
        .toBe(text.replace('revision 0', 'revision 1'))
    } finally {
      await app.evaluate(({ clipboard }, previous) => clipboard.write(previous), previousClipboard)
      await previousClipboard.dispose()
    }
  } finally {
    await app?.close()
    await server.close()
    await rm(userData, { recursive: true, force: true })
  }
})
