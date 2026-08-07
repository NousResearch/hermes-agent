import fs from 'node:fs'
import path from 'node:path'

import { type NoProviderFixture, setupNoProvider } from './fixtures'
import { expect, test } from './test'

const LONG_TOKEN = 'x'.repeat(500)

function rendererCss(): string {
  const assetsDir = path.resolve(import.meta.dirname, '..', 'dist', 'assets')
  const stylesheet = fs.readdirSync(assetsDir).find(file => /^index-.*\.css$/.test(file))

  if (!stylesheet) {
    throw new Error(`Desktop renderer stylesheet not found in ${assetsDir}`)
  }

  return fs.readFileSync(path.join(assetsDir, stylesheet), 'utf8')
}

let fixture: NoProviderFixture | null = null

test.beforeAll(async () => {
  fixture = await setupNoProvider()
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('constrains prose and tables while code keeps its overlay scroller', async () => {
  const page = fixture!.page

  await page.setContent(`
    <style>${rendererCss()}</style>
    <main
      class="aui-md prose w-full min-w-0 max-w-none overflow-hidden"
      data-testid="markdown"
      style="width: 320px"
    >
      <p class="wrap-anywhere">${LONG_TOKEN}</p>
      <div class="aui-md-table my-2 max-w-full overflow-hidden">
        <table class="m-0 w-full table-fixed border-collapse">
          <thead>
            <tr><th class="wrap-anywhere">Header with a long value</th><th>Other</th></tr>
          </thead>
          <tbody>
            <tr><td class="wrap-anywhere">${LONG_TOKEN}</td><td>value</td></tr>
          </tbody>
        </table>
      </div>
      <div data-slot="code-card">
        <div class="[&_pre]:overflow-x-auto [&_pre]:scrollbar-overlay" data-slot="code-card-body">
          <div class="scrollbar-overlay overflow-y-auto overflow-x-auto">
            <pre class="m-0 overflow-hidden" data-testid="code-scroller">
              <code class="block whitespace-pre">${LONG_TOKEN}</code>
            </pre>
          </div>
        </div>
      </div>
    </main>
  `)

  const metrics = await page.getByTestId('markdown').evaluate(async element => {
    const root = element as HTMLElement

    await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))

    const prose = root.querySelector('p') as HTMLElement
    const tableWrapper = root.querySelector('.aui-md-table') as HTMLElement
    const codeScroller = root.querySelector('[data-testid="code-scroller"]') as HTMLElement

    return {
      codeClientWidth: codeScroller.clientWidth,
      codeOverflowX: getComputedStyle(codeScroller).overflowX,
      codeScrollWidth: codeScroller.scrollWidth,
      proseClientWidth: prose.clientWidth,
      proseScrollWidth: prose.scrollWidth,
      rootClientWidth: root.clientWidth,
      rootScrollWidth: root.scrollWidth,
      tableClientWidth: tableWrapper.clientWidth,
      tableOverflowX: getComputedStyle(tableWrapper).overflowX,
      tableScrollWidth: tableWrapper.scrollWidth
    }
  })

  expect(metrics.rootClientWidth).toBe(320)
  expect(metrics.rootScrollWidth).toBe(metrics.rootClientWidth)
  expect(metrics.proseScrollWidth).toBe(metrics.proseClientWidth)
  expect(metrics.tableOverflowX).toBe('hidden')
  expect(metrics.tableScrollWidth).toBe(metrics.tableClientWidth)
  expect(metrics.codeOverflowX).toBe('auto')
  expect(metrics.codeScrollWidth).toBeGreaterThan(metrics.codeClientWidth)
})
