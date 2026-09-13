/**
 * TEMPORARY repro spec — model options submenu (Thinking / Effort / Fast).
 *
 * Reproduces the upstream reports: #97505 (submenu closes during diagonal
 * pointer travel), #86966 (hover-only, no affordance). Goal: determine
 * whether the submenu (a) never appears on hover, (b) appears but is
 * fragile to pointer travel, or (c) works fine in a clean Electron run
 * (which would point at a WSLg-specific interaction problem).
 */
import { expect, test, type Page } from '@playwright/test'

import { setupMockBackend } from './fixtures'

async function openModelMenu(page: Page) {
  page.on('console', m => {
    if (m.type() === 'error') console.log('[console.error]', m.text())
  })
  page.on('pageerror', e => console.log('[pageerror]', e.message))
  // Composer pill: aria-label "Model · mock: mock-model" (provider id is the lowercase key)
  const pill = page.getByRole('button', { name: /Model · [Mm]ock: mock-model/ })
  // Cold-boot race: the app sometimes lands on the empty sessions view
  // ("No sessions yet") before the mock session exists — open one so the
  // composer (and its model pill) mounts.
  try {
    await expect(pill).toBeVisible({ timeout: 8_000 })
  } catch {
    await page.getByRole('button', { name: /New session/i }).first().click()
  }
  await expect(pill).toBeVisible({ timeout: 30_000 })
  await pill.click()
  // Menu content should list the mock model row; the catalog may still be
  // loading on a cold boot (rows render disabled), so retry open/close.
  const row = page.getByRole('menuitem', { name: /Mock Model/i }).first()
  for (let attempt = 0; attempt < 5; attempt++) {
    if (await row.isVisible().catch(() => false)) break
    await page.keyboard.press('Escape').catch(() => undefined)
    await page.waitForTimeout(2_000)
    await pill.click()
  }
  await expect(page.getByRole('menu')).toBeVisible({ timeout: 10_000 })
}

async function hoverModelRow(page: Page) {
  // Row visible text is the display name ("Mock Model") + effort meta ("Med"),
  // NOT the raw model id — match on the display name.
  const row = page.getByRole('menuitem', { name: /Mock Model/i })
  await expect(row).toBeVisible({ timeout: 10_000 })
  // Playwright's locator.hover() hit-target check mis-fires here (the menu
  // content is reported as intercepting), so move the real mouse instead —
  // this still dispatches genuine pointerenter events that Radix listens to.
  const box = await row.boundingBox()
  if (!box) throw new Error('row has no bounding box')
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  await page.mouse.move(box.x + box.width / 2 + 1, box.y + box.height / 2)
  return row
}

test('hover opens the per-model options submenu', async () => {
  const { page, cleanup } = await setupMockBackend()
  test.setTimeout(90_000)

  try {
    await openModelMenu(page)
    const row = await hoverModelRow(page)

    // The submenu portal should mount: a nested menu containing the effort
    // radios (Minimal … Ultra) and/or the Thinking switch.
    const optionsLabel = page.getByText('Options', { exact: true })
    const effortRadio = page.getByRole('menuitemradio', { name: 'Medium' })

    // Give Radix its open delay plus margin.
    await expect
      .poll(
        async () => {
          const labelVisible = await optionsLabel.isVisible().catch(() => false)
          const radioVisible = await effortRadio.isVisible().catch(() => false)
          return labelVisible || radioVisible
        },
        { timeout: 5_000, intervals: [200] },
      )
      .toBe(true)

    // And keep hovering: it must STAY open while we rest on the trigger.
    await page.waitForTimeout(500)
    await expect(effortRadio).toBeVisible()
    await expect(row).toHaveAttribute('data-state', /open/)

    // Select an effort directly via the submenu — stepped real-mouse travel
    // from the trigger to the radio, logging geometry + open state at each
    // step so we can see exactly where the submenu dies.
    const rowBox = await row.boundingBox()
    const radioBox = await effortRadio.boundingBox()
    if (!rowBox || !radioBox) throw new Error('missing geometry')
    console.log('[geom] trigger', JSON.stringify(rowBox), 'radio', JSON.stringify(radioBox))
    const steps = 20
    for (let i = 0; i <= steps; i++) {
      const t = i / steps
      const x = rowBox.x + rowBox.width / 2 + t * (radioBox.x + radioBox.width / 2 - (rowBox.x + rowBox.width / 2))
      const y = rowBox.y + rowBox.height / 2 + t * (radioBox.y + radioBox.height / 2 - (rowBox.y + rowBox.height / 2))
      await page.mouse.move(x, y)
      const openNow = await row.getAttribute('data-state').catch(() => 'gone')
      if (i % 5 === 0 || openNow !== 'open') console.log(`[travel] step ${i} xy=(${x | 0},${y | 0}) row data-state=${openNow}`)
    }
    // Click whatever the pointer is on now via a real press.
    const atPoint = await page.evaluate(({ x, y }) => {
      const el = document.elementFromPoint(x, y)
      return el ? `${el.tagName}[role=${el.getAttribute('role') ?? ''}] ${el.textContent?.trim().slice(0, 30)}` : 'none'
    }, { x: radioBox.x + radioBox.width / 2, y: radioBox.y + radioBox.height / 2 })
    console.log('[travel] element at radio point:', atPoint)
    await page.mouse.click(radioBox.x + radioBox.width / 2, radioBox.y + radioBox.height / 2)
    // NOTE: the menu intentionally STAYS open after picking an effort (Radix
    // radio items don't dismiss the menu). Assert the selection registered —
    // the radio is checked and the pill shows the effort pin dot.
    await expect(effortRadio).toHaveAttribute('aria-checked', 'true')
  } finally {
    await cleanup()
  }
})

test('clicking the row caret opens the submenu without committing the model', async () => {
  const { page, cleanup } = await setupMockBackend()
  test.setTimeout(90_000)

  try {
    await openModelMenu(page)

    // The caret is the always-visible affordance; clicking it must open the
    // options submenu (a hover-free pointer path for environments that drop
    // hover events) and must NOT select the model / close the menu.
    const caret = page.getByRole('menuitem', { name: /Mock Model/i }).locator('[data-row-caret]')
    await expect(caret).toBeVisible({ timeout: 10_000 })
    // NOTE: locator.click() mis-fires here (the menu content is reported as
    // intercepting — same hit-test drift as hoverModelRow), so press with the
    // real mouse at the caret's center.
    const caretBox = await caret.boundingBox()
    if (!caretBox) throw new Error('caret has no bounding box')
    // A standalone move first: it starts Radix's 100ms open timer the same way
    // a real user's approach does, then the click lands on an already-opening
    // trigger instead of racing it.
    await page.mouse.move(caretBox.x + caretBox.width / 2, caretBox.y + caretBox.height / 2)
    await page.mouse.click(caretBox.x + caretBox.width / 2, caretBox.y + caretBox.height / 2)

    const effortRadio = page.getByRole('menuitemradio', { name: 'Medium' })
    await expect(effortRadio).toBeVisible({ timeout: 5_000 })

    // The catalog is still open and the submenu mounted (two role=menu nodes
    // now exist — parent + portaled submenu).
    await expect(page.getByRole('menuitemradio', { name: 'Medium' })).toBeVisible()
    await expect(page.getByRole('menuitem', { name: /Mock Model/i })).toBeVisible()
  } finally {
    await cleanup()
  }
})

test('submenu survives diagonal pointer travel toward a lower effort option', async () => {
  const { page, cleanup } = await setupMockBackend()
  test.setTimeout(90_000)

  try {
    await openModelMenu(page)
    const row = page.getByRole('menuitem', { name: /Mock Model/i })
    await expect(row).toBeVisible()
    const box0 = await row.boundingBox()
    if (!box0) throw new Error('row has no bounding box')
    await page.mouse.move(box0.x + box0.width / 2, box0.y + box0.height / 2)
    await page.mouse.move(box0.x + box0.width / 2 + 1, box0.y + box0.height / 2)

    const effortRadio = page.getByRole('menuitemradio', { name: 'Medium' })
    await expect(effortRadio).toBeVisible({ timeout: 5_000 })

    // Diagonal move: from the trigger toward a LOWER option in the portaled
    // submenu, deliberately crossing the sibling-row band (issue #97505).
    const from = await row.boundingBox()
    const to = await page.getByRole('menuitemradio', { name: 'Minimal' }).boundingBox()
    if (!from || !to) throw new Error('missing geometry')

    // Start at the trigger's right edge, move in a straight diagonal line to
    // just above the Minimal option's right edge, sampling in small steps.
    const steps = 30
    for (let i = 0; i <= steps; i++) {
      const t = i / steps
      const x = from.x + from.width - 6 + t * ((to.x + to.width - 6) - (from.x + from.width - 6))
      const y = from.y + from.height / 2 + t * ((to.y + to.height / 2) - (from.y + from.height / 2))
      await page.mouse.move(x, y)
      await page.waitForTimeout(16)
    }

    // The diagonal path may legitimately pass over sibling rows; the submenu
    // should still be reachable — Minimal should be clickable at the end.
    const minimal = page.getByRole('menuitemradio', { name: 'Minimal' })
    const survived = await minimal.isVisible().catch(() => false)
    if (survived) {
      await minimal.click()
    }

    // Soft assertion: log the outcome, don't hard-fail — we're gathering
    // evidence for the upstream report first.
    expect({ diagonalSurvived: survived }).toEqual({ diagonalSurvived: true })
  } finally {
    await cleanup()
  }
})