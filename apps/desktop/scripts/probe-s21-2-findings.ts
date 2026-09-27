/**
 * Resolve the S21.2 independent-review findings with measurements:
 * 1. search affordance contrast (rest + focused)
 * 2. primary button contrast (dark + light)
 * 3. 160-of-160 counter vs 3 visible rows — is the list internally scrollable?
 * Writes a JSON receipt next to the screenshots.
 */
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { chromium } from '@playwright/test'
import { createServer, type ViteDevServer } from 'vite'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const FLEET_SIZE = 160
const ROLLOUT_TITLE = 'عمليات النشر المُدارة'
const FLEET_TITLE = 'أهداف النشر المُدار'
const RECEIPT = path.resolve(DESKTOP_ROOT, '../../.hermes/campaigns/managed-ssh-fleet/118029-live/reviews/s21-2-finding-measurements.json')

function fleetRows() {
  return Array.from({ length: FLEET_SIZE }, (_, index) => {
    const connectionId = '00000000-0000-4000-8000-' + (index + 1).toString(16).padStart(12, '0')
    const installId = (index + 1).toString(16).padStart(32, '0')

    return {
      installId,
      connectionId,
      aliasConnectionIds: index === 0
        ? Array.from({ length: 12 }, (_, alias) => '00000000-0000-4000-8000-' + (alias + 200).toString(16).padStart(12, '0'))
        : [],
      codeRoot: '/disposable/hermes/' + index,
      repositoryId: 'github.com/NousResearch/hermes-agent',
      headSha: 'a'.repeat(40),
      requiredScopeIds: ['default'],
      source: {
        connectionId,
        verifiedHostKeyFingerprint: index < 2 ? 'SHA256:shared-fixture-machine' : 'SHA256:fixture-machine-' + index
      }
    }
  })
}

async function relativeLuminance(rgb: [number, number, number]): Promise<number> {
  const [r, g, b] = rgb.map(channel => {
    const value = channel / 255

    return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
  }) as [number, number, number]

  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function parseRgb(value: string): [number, number, number] {
  const match = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(value)

  return match ? [Number(match[1]), Number(match[2]), Number(match[3])] : [0, 0, 0]
}

function contrast(a: string, b: string) {
  return (async () => {
    const l1 = await relativeLuminance(parseRgb(a))
    const l2 = await relativeLuminance(parseRgb(b))

    return Number(((Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05)).toFixed(2))
  })()
}

async function main() {
  let server: ViteDevServer | undefined

  try {
    const scratch = await fs.promises.mkdtemp(path.join(os.tmpdir(), 's21-2-probe-'))
    Object.assign(globalThis, { __dirname: DESKTOP_ROOT })
    server = await createServer({
      root: DESKTOP_ROOT,
      configFile: path.join(DESKTOP_ROOT, 'vite.config.ts'),
      configLoader: 'runner',
      cacheDir: path.join(scratch, 'node_modules/.vite'),
      server: { host: '127.0.0.1', port: 0, strictPort: false },
      optimizeDeps: { entries: ['scripts/fixtures/managed-rollout-ui.html'] }
    })
    await server.listen()
    const url = server.resolvedUrls!.local[0] + 'scripts/fixtures/managed-rollout-ui.html'
    const browser = await chromium.launch()
    const page = await browser.newPage({ viewport: { width: 1220, height: 800 } })

    await page.addInitScript(rows => { (window as any).__managedRolloutRows = rows }, fleetRows())
    const findings: Record<string, unknown> = { fleetSize: FLEET_SIZE, url }

    for (const scheme of ['dark', 'light'] as const) {
      await page.emulateMedia({ colorScheme: scheme })
      await page.goto(url)
      await page.setViewportSize({ width: 1220, height: 800 })
      await page.getByRole('heading', { name: ROLLOUT_TITLE }).waitFor({ timeout: 30_000 })

      const region = page.getByRole('region', { name: ROLLOUT_TITLE })

      // 3. Counter vs visible rows: is the fleet region internally scrollable?
      const fleet = region.getByRole('region', { name: FLEET_TITLE })

      const scroll = await fleet.evaluate(element => ({
        clientHeight: element.clientHeight,
        scrollHeight: element.scrollHeight,
        overflowY: getComputedStyle(element).overflowY,
        rows: element.querySelectorAll('button[aria-pressed]').length
      }))

      // can we actually reach the last row?
      await fleet.evaluate(element => { element.scrollTop = element.scrollHeight })

      const lastVisible = await fleet.evaluate(element => {
        const buttons = element.querySelectorAll('button[aria-pressed]')
        const last = buttons[buttons.length - 1] as HTMLElement | undefined

        if (!last) {return false}
        const fleetRect = element.getBoundingClientRect()
        const lastRect = last.getBoundingClientRect()

        return lastRect.top < fleetRect.bottom && lastRect.bottom > fleetRect.top
      })

      // 1. search affordance: rest vs focused (wait for the opacity transition to settle)
      const searchInput = region.getByRole('textbox').first()

      const searchRest = await searchInput.evaluate(element => ({
        opacity: getComputedStyle(element.parentElement!).opacity,
        placeholderColor: getComputedStyle(element, '::placeholder').color,
        textColor: getComputedStyle(element).color,
        bg: getComputedStyle(document.body).backgroundColor
      }))

      await searchInput.focus()
      await page.waitForTimeout(600)

      const searchFocused = await searchInput.evaluate(element => ({
        opacity: getComputedStyle(element.parentElement!).opacity,
        placeholderColor: getComputedStyle(element, '::placeholder').color
      }))

      await searchInput.blur()
      await page.waitForTimeout(600)

      const searchBlurred = await searchInput.evaluate(element => ({
        opacity: getComputedStyle(element.parentElement!).opacity
      }))

      // 2. action buttons: measure both the disabled and enabled states
      const buttons = region.locator('button').filter({ hasText: /إعداد|مراجعة/ })

      const measureButtons = async () => {
        const data: unknown[] = []
        const count = await buttons.count()

        for (let index = 0; index < count; index += 1) {
          const button = buttons.nth(index)

          const style = await button.evaluate(element => {
            const computed = getComputedStyle(element)

            return { text: (element.textContent || '').trim().slice(0, 40), color: computed.color, bg: computed.backgroundColor, opacity: computed.opacity, disabled: (element as HTMLButtonElement).disabled }
          })

          const parentBg = await button.evaluate(element => {
            let node: HTMLElement | null = element.parentElement

            while (node) {
              const bg = getComputedStyle(node).backgroundColor

              if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {return bg}
              node = node.parentElement
            }

            return getComputedStyle(document.body).backgroundColor
          })

          const resolved = style.bg && style.bg !== 'rgba(0, 0, 0, 0)' ? style.bg : parentBg

          data.push({ ...style, resolvedBackground: resolved, contrast: await contrast(style.color, resolved) })
        }

        return data
      }

      const buttonsDisabled = await measureButtons()

      // Select the first row's checkbox-equivalent (a target row toggle) to enable the actions.
      const firstRow = fleet.locator('button[aria-pressed]').first()
      await firstRow.click()
      await page.waitForTimeout(300)
      const buttonsEnabled = await measureButtons()

      findings[scheme] = {
        scroll,
        lastVisibleAfterScroll: lastVisible,
        searchRest,
        searchFocused,
        searchBlurred,
        buttonsDisabled,
        buttonsEnabled
      }
    }

    fs.mkdirSync(path.dirname(RECEIPT), { recursive: true })
    fs.writeFileSync(RECEIPT, JSON.stringify(findings, null, 2))
    console.log(JSON.stringify(findings, null, 2))
    await browser.close()
  } finally {
    if (server) {await server.close()}
  }
}

main().catch(error => {
  console.error(error)
  process.exit(1)
})
