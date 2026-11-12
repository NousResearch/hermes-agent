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

test('constrains prose while tables and code scroll inside their own cards', async () => {
  const page = fixture!.page

  await page.setContent(`
    <style>${rendererCss()}</style>
    <main
      class="aui-md prose w-full min-w-0 max-w-none overflow-hidden"
      data-testid="markdown"
      style="width: 320px"
    >
      <p class="wrap-anywhere">${LONG_TOKEN}</p>
      <div class="aui-md-table my-2 max-w-full overflow-x-auto">
        <table class="m-0 w-full min-w-[18rem] border-collapse">
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
  expect(metrics.tableOverflowX).toBe('auto')
  expect(metrics.tableScrollWidth).toBeGreaterThan(metrics.tableClientWidth)
  expect(metrics.codeOverflowX).toBe('auto')
  expect(metrics.codeScrollWidth).toBeGreaterThan(metrics.codeClientWidth)
})

// #145797: the message body's `wrap-anywhere` used to reach table cells. Auto
// layout then counted every letter as a break opportunity and shrank the short
// columns until labels broke mid-word. Cells wrap between words now, so each
// column is at least as wide as its longest word and an over-wide table scrolls
// inside its card instead.
test('keeps table words whole and scrolls an over-wide table inside its card', async () => {
  const page = fixture!.page
  const cell = 'min-w-12 px-2.5 py-1.5 align-top leading-snug wrap-break-word'

  const row = [
    'Revisión independiente inicial del plan',
    '<code>a070870466f38f8f</code>',
    'TASK-3943 77e0f97e',
    '<code>idle</code>',
    'Revisión de planificación del cambio de método de merge',
    'No; aceptación y revisión humana pendientes en el padre'
  ]

  await page.setContent(`
    <style>${rendererCss()}</style>
    <main class="wrap-anywhere min-w-0 max-w-full overflow-hidden" data-testid="message" style="width: 384px">
      <div class="aui-md prose w-full min-w-0 max-w-none overflow-hidden">
        <div class="aui-md-table my-2 max-w-full overflow-x-auto">
          <table class="m-0 w-full min-w-[18rem] border-collapse">
            <thead>
              <tr>${row.map((_, index) => `<th>C${index}</th>`).join('')}</tr>
            </thead>
            <tbody>
              <tr>${row.map(value => `<td class="${cell}">${value}</td>`).join('')}</tr>
            </tbody>
          </table>
        </div>
      </div>
    </main>
  `)

  const metrics = await page.getByTestId('message').evaluate(async element => {
    const root = element as HTMLElement

    await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))

    const tableWrapper = root.querySelector('.aui-md-table') as HTMLElement
    const [label, id] = Array.from(root.querySelectorAll('td'))

    // The widest unbreakable run each cell holds, rendered in the cell's font.
    const textWidth = (cellElement: Element, text: string) => {
      const probe = document.createElement('span')
      probe.style.whiteSpace = 'nowrap'
      probe.textContent = text
      cellElement.appendChild(probe)
      const width = probe.getBoundingClientRect().width
      probe.remove()

      return width
    }

    const contentWidth = (cellElement: Element) => {
      const style = getComputedStyle(cellElement)

      return cellElement.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
    }

    const idCode = id.querySelector('code') as HTMLElement

    return {
      // An inline box yields one client rect per line it spans.
      idCodeLines: idCode.getClientRects().length,
      labelContentWidth: contentWidth(label),
      longestLabelWord: textWidth(label, 'independiente'),
      rootClientWidth: root.clientWidth,
      rootScrollWidth: root.scrollWidth,
      tableClientWidth: tableWrapper.clientWidth,
      tableScrollWidth: tableWrapper.scrollWidth
    }
  })

  expect(metrics.labelContentWidth).toBeGreaterThanOrEqual(Math.floor(metrics.longestLabelWord))
  expect(metrics.idCodeLines).toBe(1)
  expect(metrics.tableScrollWidth).toBeGreaterThan(metrics.tableClientWidth)
  expect(metrics.rootScrollWidth).toBe(metrics.rootClientWidth)
})
