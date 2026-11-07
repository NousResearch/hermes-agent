import { mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { build } from 'esbuild'

import { resolveElectronBinary } from './electron-binary'
import { _electron, expect, test } from './test'

/** Real tab components and renderer CSS in a frameless native window, without a backend. */
async function openTabs() {
  const temporary = mkdtempSync(path.join(tmpdir(), 'hermes-tab-layout-'))
  const main = path.join(temporary, 'main.cjs')
  writeFileSync(
    main,
    `const { app, BrowserWindow } = require('electron');
app.setPath('userData', ${JSON.stringify(path.join(temporary, 'user-data'))});
app.whenReady().then(() => {
  const window = new BrowserWindow({ width: 900, height: 400, frame: false,
    webPreferences: { sandbox: true, contextIsolation: true } });
  window.loadURL('about:blank');
});`
  )
  const assets = path.resolve('dist/assets')
  const stylesheet = readdirSync(assets).find(file => /^index-.*\.css$/.test(file))!

  const css = readFileSync(path.join(assets, stylesheet), 'utf8').replace(
    /url\(\.\/(codicon-[^)]+\.ttf)(?:\?[^)]*)?\)/g,
    (_match, file: string) => `url(data:font/ttf;base64,${readFileSync(path.join(assets, file)).toString('base64')})`
  )

  const errors: string[] = []
  const require = createRequire(path.resolve('package.json'))

  const bundle = await build({
    entryPoints: [path.resolve('src/test/pane-tab-hover-mask.tsx')],
    alias: {
      react: path.dirname(require.resolve('react/package.json')),
      'react-dom': path.dirname(require.resolve('react-dom/package.json'))
    },
    bundle: true,
    format: 'iife',
    write: false,
    jsx: 'automatic',
    loader: { '.css': 'empty' },
    plugins: [
      {
        name: 'tab-fixture-services',
        setup(build) {
          // Translation/keybind seams keep unrelated stores and gateway startup out of this fixture.
          build.onResolve({ filter: /^@\/i18n$|^@\/lib\/keybinds\/use-keybind-hint$/ }, args => ({
            path: args.path,
            namespace: 'tab-fixture'
          }))
          build.onLoad({ filter: /.*/, namespace: 'tab-fixture' }, () => ({
            contents: `export const translateNow = key => key === 'common.close' ? 'Close' : key;
export const useI18n = () => ({ t: { keybinds: { actions: {} } } });
export const useKeybindHint = () => null;`
          }))
        }
      }
    ]
  })

  const app = await _electron.launch({
    executablePath: resolveElectronBinary([path.resolve('.'), path.resolve('../..')]),
    args: ['--no-sandbox', main]
  })

  const page = await app.firstWindow()
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', route => route.abort())
  await page.setContent(`<style>${css}</style>
<style>body{margin:0;padding:16px}#root{height:28px}output{display:none}</style><div id="root"></div>`)
  await page.addScriptTag({ content: bundle.outputFiles[0].text })
  await page.getByTestId('wrapped').waitFor()
  await page.evaluate(() => document.fonts.ready)

  return { app, page, errors }
}

test('the hover mask survives a chat tab’s context-menu wrapper', async () => {
  const { app, page, errors } = await openTabs()

  try {
    for (const size of [16, 20]) {
      await page.evaluate(size => {
        document.documentElement.style.fontSize = `${size}px`
      }, size)

      for (const glass of [false, true]) {
        await page.evaluate(enabled => document.documentElement.toggleAttribute('data-hermes-glass', enabled), glass)

        for (const id of ['bare', 'wrapped', 'long']) {
          const tab = page.getByTestId(id)
          await page.mouse.move(850, 350)

          const idle = await tab.evaluate(element => ({
            width: element.getBoundingClientRect().width,
            mask: getComputedStyle(element.querySelector('.pane-tab-content')!).maskImage
          }))

          await tab.hover()

          const hovered = await tab.evaluate(element => ({
            width: element.getBoundingClientRect().width,
            mask: getComputedStyle(element.querySelector('.pane-tab-content')!).maskImage,
            closeWidth: element.querySelector('button[aria-label="Close"]')!.getBoundingClientRect().width,
            overflow: getComputedStyle(element.querySelector('.truncate')!).textOverflow
          }))

          expect(idle.mask).toBe('none')
          expect(hovered.mask).toContain('linear-gradient')
          expect(hovered.closeWidth).toBe(size * 1.5)
          expect(hovered.width).toBe(idle.width)
          expect(hovered.overflow).toBe('clip')
        }
      }
    }

    for (const id of ['home', 'vertical']) {
      const tab = page.getByTestId(id)
      await tab.hover()
      await expect(tab.getByRole('button', { name: 'Close' })).toHaveCount(0)
      expect(await tab.locator('.pane-tab-content').evaluate(element => getComputedStyle(element).maskImage)).toBe(
        'none'
      )
    }

    expect(errors).toEqual([])
  } finally {
    await app.close()
  }
})

test('wrapped tabs retain right-click actions and close without activating', async () => {
  const { app, page, errors } = await openTabs()

  try {
    const tab = page.getByTestId('wrapped')
    await tab.click({ button: 'right' })
    await expect(page.getByRole('menuitem', { name: 'Rename' })).toBeVisible()
    await page.keyboard.press('Escape')
    await tab.hover()
    await tab.getByRole('button', { name: 'Close' }).click()
    await expect(tab).toHaveCount(0)
    await expect(page.getByTestId('activations')).toHaveText('0')
    expect(errors).toEqual([])
  } finally {
    await app.close()
  }
})
