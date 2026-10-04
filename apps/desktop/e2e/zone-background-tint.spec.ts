import { type Locator } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

let fixture: MockBackendFixture
const rendererErrors: string[] = []

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  fixture.page.on('pageerror', error => rendererErrors.push(error.message))
  await waitForAppReady(fixture, 120_000)
})

test.afterEach(() => {
  expect(rendererErrors).toEqual([])
})

test.afterAll(async () => {
  await fixture?.cleanup()
})

function chatZone(): Locator {
  return fixture.page.locator('[data-slot="composer-bounds"]').first().locator('xpath=ancestor::*[@data-tree-group][1]')
}

async function enterLayoutEdit(): Promise<void> {
  const { page } = fixture
  await page.getByRole('button', { name: 'Layout editor', exact: true }).click()
  // The draggable preset card starts over the central zone's action button.
  const heading = page.getByRole('heading', { name: 'Layouts', exact: true })
  const rect = await heading.boundingBox()

  if (!rect) {
    throw new Error('Layout picker did not appear')
  }

  await page.mouse.move(rect.x + 20, rect.y + 10)
  await page.mouse.down()
  await page.mouse.move(40, 540, { steps: 5 })
  await page.mouse.up()
}

async function chooseTint(zone: Locator, label: string): Promise<void> {
  await zone.getByRole('button', { name: 'Zone actions', exact: true }).click()
  await fixture.page.getByRole('menuitem', { name: 'Background tint', exact: true }).hover()
  await fixture.page.getByRole('menuitemradio', { name: label, exact: true }).click()
}

test('a headerless zone can select, persist, and reset its own tint', async () => {
  const { page } = fixture

  const id = await page
    .getByRole('tab', { name: 'Sessions', exact: true })
    .locator('xpath=ancestor::*[@data-tree-group][1]')
    .getAttribute('data-tree-group')

  const zone = page.locator(`[data-tree-group=${JSON.stringify(id)}]`)

  const otherTints = await page
    .locator('[data-tree-group]')
    .evaluateAll(
      (groups, chatId) =>
        groups
          .filter(group => group.getAttribute('data-tree-group') !== chatId)
          .map(group => [group.getAttribute('data-tree-group'), group.getAttribute('data-zone-background-tint')]),
      id
    )

  await enterLayoutEdit()
  await zone.getByRole('button', { name: 'Zone actions', exact: true }).click()
  await page.getByRole('menuitem', { name: /^Hide tabs/ }).click()
  await expect(zone).not.toHaveAttribute('data-zone-header')
  await chooseTint(zone, 'Cyan')
  await expect(zone).toHaveAttribute('data-zone-background-tint', 'cyan')
  expect(
    await page
      .locator('[data-tree-group]')
      .evaluateAll(
        (groups, chatId) =>
          groups
            .filter(group => group.getAttribute('data-tree-group') !== chatId)
            .map(group => [group.getAttribute('data-tree-group'), group.getAttribute('data-zone-background-tint')]),
        id
      )
  ).toEqual(otherTints)

  await page.getByRole('button', { name: 'Done', exact: true }).click()
  await page.reload()
  await waitForAppReady(fixture)
  await expect(zone).toHaveAttribute('data-zone-background-tint', 'cyan')

  await enterLayoutEdit()
  await chooseTint(zone, 'Default background')
  await expect(zone).not.toHaveAttribute('data-zone-background-tint')
  await page.getByRole('button', { name: 'Done', exact: true }).click()
  await page.reload()
  await waitForAppReady(fixture)
  await expect(zone).not.toHaveAttribute('data-zone-background-tint')
})

test('glass paints one tint regardless of content depth and preserves opaque masks', async () => {
  const { page } = fixture
  const zone = chatZone()

  await enterLayoutEdit()
  await chooseTint(zone, 'Cyan')
  await page.getByRole('button', { name: 'Done', exact: true }).click()

  for (const appearance of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme: appearance })
    await expect(page.locator('html')).toHaveAttribute('data-hermes-mode', appearance)

    // Exercise the CSS contract independently of native material availability,
    // so the same regression runs on Linux as well as macOS and Windows.
    const colors = await zone.evaluate(element => {
      const root = document.documentElement
      const hadGlass = root.hasAttribute('data-hermes-glass')
      root.setAttribute('data-hermes-glass', '')

      const color = (node: Element) => getComputedStyle(node).backgroundColor
      const composer = element.querySelector('[data-slot="composer-bounds"]')!
      const layers = document.createElement('div')
      element.append(layers)
      const fieldColors: string[] = []
      let parent = layers

      for (let depth = 0; depth < 4; depth++) {
        const field = document.createElement('div')
        field.style.backgroundColor = 'var(--ui-chat-surface-background)'
        parent.append(field)
        fieldColors.push(color(field))
        parent = field
      }

      const mask = document.createElement('div')
      mask.setAttribute('data-glass-opaque', '')
      mask.style.backgroundColor = 'var(--ui-chat-surface-background)'
      parent.append(mask)

      const glass = { zone: color(element), composer: color(composer), fields: fieldColors, mask: color(mask) }
      root.removeAttribute('data-hermes-glass')
      const opaque = { zone: color(element), composer: color(composer) }
      layers.remove()
      root.toggleAttribute('data-hermes-glass', hadGlass)

      return { glass, opaque }
    })

    expect(colors.glass.zone).toMatch(/\/ 0\.1\)$/)
    expect(colors.glass.composer).toBe('rgba(0, 0, 0, 0)')
    expect(colors.glass.fields.every(color => color === 'rgba(0, 0, 0, 0)')).toBe(true)
    expect(colors.glass.mask).not.toBe('rgba(0, 0, 0, 0)')
    expect(colors.glass.mask).not.toMatch(/\//)
    expect(colors.opaque.zone).toBe(colors.opaque.composer)
    expect(colors.opaque.zone).not.toMatch(/\//)
    await page.screenshot({ path: test.info().outputPath(`zone-tint-${appearance}.png`) })
  }
})
