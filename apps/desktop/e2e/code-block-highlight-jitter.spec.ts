/**
 * Diagnostic regression for #88151: opening a persisted transcript containing
 * fenced code blocks must not move the viewport when LazyShiki replaces its
 * plain Suspense fallback with the highlighted DOM.
 *
 * This uses the real Electron renderer and a real persisted SessionDB row. The
 * mock provider only supplies deterministic Markdown; no external model calls.
 */

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig,
} from './fixtures'
import { startMockServer } from './mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

const SESSION_TITLE = 'E2E Code Block Highlight Jitter'
const SURFACE = '[data-composer-target]:not([data-pane-hidden] [data-composer-target])'
const INLINE_BLOCK_COUNT = 6
const LONG_BLOCK_LINES = 240
const LINES_PER_SHORT_BLOCK = 14

const shortBlocks = Array.from({ length: INLINE_BLOCK_COUNT }, (_, block) => {
  const lines = Array.from(
    { length: LINES_PER_SHORT_BLOCK },
    (_, line) => `const value_${block}_${line} = ${block * 100 + line}`,
  )

  return [`Block ${block + 1}`, '```typescript', ...lines, '```'].join('\n')
})

const longBlock = [
  'Long block (exercises the >200-line chunked fallback)',
  '```typescript',
  ...Array.from({ length: LONG_BLOCK_LINES }, (_, line) => `const long_value_${line} = ${line}`),
  '```',
].join('\n')

const CODE_REPLY = [...shortBlocks, longBlock].join('\n\n')

interface LayoutSample {
  reason: string
  at: number
  blockCount: number
  fallbackCount: number
  shikiCount: number
  heights: number[]
  typography: Array<{
    codeFontSize: string
    codeHeight: number
    codeLineHeight: string
    codeDisplay: string
    innerPreHeight: number | null
    outerFontSize: string
    outerLineHeight: string
    outerPaddingBlock: string
    node: string
  }>
  totalHeight: number
  scrollTop: number
  scrollHeight: number
}

interface LayoutProbe {
  samples: LayoutSample[]
  observer: MutationObserver
}

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  const mock = await startMockServer({ replyText: CODE_REPLY })
  const sandbox = createSandbox('code-jitter')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  const builder = await RealSessionBuilder.start(sandbox.hermesHome)

  try {
    await builder.createSession({
      title: SESSION_TITLE,
      turns: ['Render a deterministic fenced-code transcript for layout measurement.'],
    })
  } finally {
    await builder.close()
  }

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))
  fixture = {
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
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

// Playwright requires the fixture argument to use object destructuring.
// eslint-disable-next-line no-empty-pattern
test('LazyShiki replacement keeps code block height stable', async ({}, testInfo) => {
  const page = fixture!.page

  const sessionRow = page
    .locator('[data-slot="sidebar"] button')
    .filter({ hasText: SESSION_TITLE })
    .first()

  await sessionRow.waitFor({ state: 'visible', timeout: 60_000 })

  await page.evaluate((surfaceSelector: string) => {
    const state: LayoutProbe = {
      samples: [],
      observer: null as unknown as MutationObserver,
    }

    let lastSignature = ''

    const capture = (reason: string) => {
      const surfaces = [...document.querySelectorAll<HTMLElement>(surfaceSelector)]
      const surface = surfaces.at(-1)
      const viewport = surface?.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')
      const blocks = [...(surface?.querySelectorAll<HTMLElement>('.aui-shiki') ?? [])]

      if (!viewport || blocks.length === 0) {
        return
      }

      const heights = blocks.map(block => Number(block.getBoundingClientRect().height.toFixed(2)))

      const typography = blocks.map(block => {
        const node = block.querySelector<HTMLElement>('code') ?? block
        const innerPre = block.querySelector<HTMLElement>('pre')
        const outerStyle = getComputedStyle(block)
        const codeStyle = getComputedStyle(node)

        return {
          codeFontSize: codeStyle.fontSize,
          codeHeight: Number(node.getBoundingClientRect().height.toFixed(2)),
          codeLineHeight: codeStyle.lineHeight,
          codeDisplay: codeStyle.display,
          innerPreHeight: innerPre ? Number(innerPre.getBoundingClientRect().height.toFixed(2)) : null,
          outerFontSize: outerStyle.fontSize,
          outerLineHeight: outerStyle.lineHeight,
          outerPaddingBlock: `${outerStyle.paddingTop} ${outerStyle.paddingBottom}`,
          node: node.tagName.toLowerCase(),
        }
      })

      const shikiCount = blocks.filter(block => Boolean(block.querySelector('.shiki'))).length
      const fallbackCount = blocks.length - shikiCount

      const sample: LayoutSample = {
        reason,
        at: performance.now(),
        blockCount: blocks.length,
        fallbackCount,
        shikiCount,
        heights,
        typography,
        totalHeight: Number(heights.reduce((sum, height) => sum + height, 0).toFixed(2)),
        scrollTop: Number(viewport.scrollTop.toFixed(2)),
        scrollHeight: viewport.scrollHeight,
      }

      const signature = JSON.stringify([
        sample.blockCount,
        sample.fallbackCount,
        sample.shikiCount,
        sample.heights,
        sample.scrollTop,
        sample.scrollHeight,
      ])

      if (signature !== lastSignature) {
        state.samples.push(sample)
        lastSignature = signature
      }
    }

    state.observer = new MutationObserver(() => {
      capture('mutation')
      requestAnimationFrame(() => capture('animation-frame'))
    })
    state.observer.observe(document.body, { childList: true, subtree: true })
    ;(window as unknown as { __CODE_BLOCK_LAYOUT_PROBE__?: LayoutProbe }).__CODE_BLOCK_LAYOUT_PROBE__ = state
  }, SURFACE)

  await sessionRow.click()
  await page.waitForFunction(
    ([surfaceSelector, expected]: [string, number]) => {
      const surfaces = [...document.querySelectorAll<HTMLElement>(surfaceSelector)]
      const surface = surfaces.at(-1)

      return (surface?.querySelectorAll('.aui-shiki .shiki').length ?? 0) === expected
    },
    [SURFACE, INLINE_BLOCK_COUNT] as [string, number],
    { timeout: 60_000 },
  )
  await page.waitForTimeout(1_000)

  const samples = await page.evaluate(() => {
    const probe = (window as unknown as { __CODE_BLOCK_LAYOUT_PROBE__?: LayoutProbe }).__CODE_BLOCK_LAYOUT_PROBE__
    probe?.observer.disconnect()

    return probe?.samples ?? []
  })

  await testInfo.attach('code-block-layout-samples', {
    body: Buffer.from(JSON.stringify(samples, null, 2)),
    contentType: 'application/json',
  })

  const completeSamples = samples.filter(sample => sample.blockCount === INLINE_BLOCK_COUNT)

  const fallback = completeSamples
    .filter(sample => sample.fallbackCount > 0)
    .sort((a, b) => b.fallbackCount - a.fallbackCount || a.at - b.at)[0]

  const highlighted = [...completeSamples].reverse().find(sample => sample.shikiCount === INLINE_BLOCK_COUNT)

  expect(fallback, `Expected to capture the PlainCode fallback. Samples: ${JSON.stringify(samples)}`).toBeTruthy()
  expect(highlighted, `Expected to capture the final Shiki DOM. Samples: ${JSON.stringify(samples)}`).toBeTruthy()
  await expect(page.getByRole('button', { name: /typescript Code Open/ })).toBeVisible()

  const perBlockHeightDeltas = highlighted!.heights.map((height, index) =>
    Number(Math.abs(height - fallback!.heights[index]).toFixed(2)),
  )

  const heightDelta = Number(Math.abs(highlighted!.totalHeight - fallback!.totalHeight).toFixed(2))
  const scrollDelta = Number(Math.abs(highlighted!.scrollTop - fallback!.scrollTop).toFixed(2))
  console.log(
    `#88151 layout evidence: height ${fallback!.totalHeight} -> ${highlighted!.totalHeight} (Δ${heightDelta}px), ` +
      `scrollTop ${fallback!.scrollTop} -> ${highlighted!.scrollTop} (Δ${scrollDelta}px)`,
  )

  expect(
    Math.max(...perBlockHeightDeltas),
    `Individual code block heights changed across fallback -> Shiki: ${JSON.stringify({ perBlockHeightDeltas, fallback, highlighted })}`,
  ).toBeLessThanOrEqual(1)
  expect(heightDelta, `Code block height changed across fallback -> Shiki: ${JSON.stringify({ fallback, highlighted })}`).toBeLessThanOrEqual(1)
})
