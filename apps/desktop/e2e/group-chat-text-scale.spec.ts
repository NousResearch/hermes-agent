import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'

import { build } from 'esbuild'
import { createServer } from 'vite'

import { resolveElectronBinary } from './electron-binary'
import { _electron, expect, test } from './test'

// Real components, shared Markdown renderer and stylesheet in Chromium. Only
// the gateway and plugin translations are scripted; no model or backend needed.
test('group text follows Chat Text Size without resizing room controls', async () => {
  const testInfo = test.info()
  const temporary = mkdtempSync(path.join(tmpdir(), 'hermes-group-text-scale-'))
  const main = path.join(temporary, 'main.cjs')
  writeFileSync(main, `const { app, BrowserWindow } = require('electron');
app.setPath('userData', ${JSON.stringify(path.join(temporary, 'user-data'))});
app.whenReady().then(() => {
  const window = new BrowserWindow({ width: 1000, height: 800,
    webPreferences: { sandbox: true, contextIsolation: true } });
  window.loadFile(${JSON.stringify(path.join(temporary, 'index.html'))});
});`)
  writeFileSync(path.join(temporary, 'index.html'), '<div id="root" style="height:700px"></div>')

  const server = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' })
  let css: string

  try {
    css = (await server.transformRequest('/src/styles.css?direct'))!.code
  } finally {
    await server.close()
  }

  const bundle = await build({
    stdin: {
      resolveDir: process.cwd(), loader: 'tsx', contents: `
import { createRoot } from 'react-dom/client'
import { CanonicalGroupWorkspace } from './src/plugins/hermes-bots/canonical-group-workspace'
import { setChatTextScale } from './src/store/chat-text-scale'
window.setChatTextScale = setChatTextScale
createRoot(document.getElementById('root')).render(
  <CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'team', roomId: 'room' }} />
)`
    },
    bundle: true, format: 'iife', write: false, jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"production"', 'import.meta.env': '{}', 'import.meta.hot': 'undefined' },
    alias: { '@': path.resolve('src') },
    loader: { '.woff2': 'dataurl', '.css': 'empty' },
    plugins: [{
      name: 'group-gateway-fixture',
      setup(builder) {
        builder.onResolve({ filter: /^@hermes\/plugin-sdk$/ }, () => ({ path: 'sdk', namespace: 'fixture' }))
        builder.onLoad({ filter: /.*/, namespace: 'fixture' }, () => ({
          resolveDir: process.cwd(), loader: 'tsx', contents: `
export { Button } from './src/components/ui/button'
export { Textarea } from './src/components/ui/textarea'
export { MessageTextContent } from './src/components/assistant-ui/markdown-text'
export { Streamdown } from 'streamdown'
export { atom } from 'nanostores'
export { relativeTime } from './src/lib/time'
import { en } from './src/i18n/en'
import { CANONICAL_GROUP_LOCALES } from './src/plugins/hermes-bots/canonical-group-locales'
export const useI18n = () => ({ t: en })
export const usePluginI18n = () => key => CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '')] ?? key
export const GlyphSpinner = () => null
export const host = { requestProfile: async (_route, method) => {
  if (method === 'groups.state') return { room: { name: 'Typography room', members: [] }, driver_status: {} }
  if (method === 'groups.log') return { events: [
    { seq: 1, kind: 'message.user', payload: { text: 'User message' } },
    { seq: 2, kind: 'message.member', payload: { text: 'Member reply\\n\\n## Heading\\n\\n- List item\\n\\n| Column |\\n| --- |\\n| Table cell |\\n\\n~~~text\\ncode sample\\n~~~' } },
    { seq: 3, kind: 'turn.failed', payload: { error: 'Provider timed out' } }
  ] }
  throw new Error('Unexpected RPC: ' + method)
} }`
        }))
      }
    }]
  })

  const app = await _electron.launch({
    executablePath: resolveElectronBinary([path.resolve('.'), path.resolve('../..')]),
    args: ['--no-sandbox', main]
  })

  try {
    const page = await app.firstWindow()
    await page.addStyleTag({ content: css })
    await page.addScriptTag({ content: bundle.outputFiles[0].text })
    await expect(page.getByText('Member reply', { exact: true })).toBeVisible()
    await page.getByRole('textbox').fill('Unsent draft')
    const samples = []

    for (const scale of [100, 175, 90, 110]) {
      await page.evaluate(value => (window as unknown as { setChatTextScale: (n: number) => void }).setChatTextScale(value), scale)

      const sample = await page.evaluate(() => {
        const font = (selector: string) => Number.parseFloat(getComputedStyle(document.querySelector(selector)!).fontSize)
        const box = document.querySelector('textarea')!

        return {
          user: font('[role="log"] .aui-md p'),
          member: font('[role="log"] > div:nth-child(2) .aui-md p'),
          heading: font('[role="log"] h2'),
          list: font('[role="log"] li'),
          table: font('[role="log"] td'),
          code: font('[role="log"] pre code'),
          notice: font('[role="log"] > div:last-of-type > p'),
          composer: font('textarea'),
          composerLineHeight: Number.parseFloat(getComputedStyle(box).lineHeight),
          header: font('header h2'),
          button: font('button[type="submit"]'),
          width: box.getBoundingClientRect().width
        }
      })

      samples.push({ scale, ...sample })
      await expect(page.getByRole('textbox')).toHaveValue('Unsent draft')

      if (scale === 100 || scale === 175) {
        await testInfo.attach(`group-text-${scale}.png`, { body: await page.screenshot(), contentType: 'image/png' })
      }
    }

    await testInfo.attach('font-sizes.json', { body: JSON.stringify(samples, null, 2), contentType: 'application/json' })

    for (const sample of samples.slice(1)) {
      for (const key of ['user', 'member', 'heading', 'list', 'table', 'code', 'notice', 'composer', 'composerLineHeight'] as const) {
        expect.soft(sample[key], `${key} at ${sample.scale}%`).toBeCloseTo(samples[0][key] * sample.scale / 100, 2)
      }

      expect(sample.header).toBe(samples[0].header)
      expect(sample.button).toBe(samples[0].button)
      expect(sample.width).toBeCloseTo(samples[0].width, 1)
    }
  } finally {
    await app.close()
    rmSync(temporary, { recursive: true, force: true })
  }
})
