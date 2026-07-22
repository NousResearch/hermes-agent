import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { expect, test } from './test'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend({ language: 'ar' })
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('titlebar edge controls follow the physical RTL pane positions before and after flipping', async () => {
  const page = fixture!.page
  const leftEdge = page.getByRole('button', { name: /الشريط الجانبي الأيسر/ })
  const rightEdge = page.getByRole('button', { name: /الشريط الجانبي الأيمن/ })
  const flip = page.getByRole('button', { name: 'تبديل جانبي الشريطين' })
  const sessionSearch = page.getByRole('textbox', { name: 'البحث في الجلسات' })

  await expect(page.locator('html')).toHaveAttribute('dir', 'rtl')

  if ((await leftEdge.getAttribute('aria-label'))?.startsWith('إخفاء')) {
    await leftEdge.click()
  }

  if ((await rightEdge.getAttribute('aria-label'))?.startsWith('إخفاء')) {
    await rightEdge.click()
  }

  await expect(leftEdge).toHaveAttribute('aria-label', 'إظهار الشريط الجانبي الأيسر')
  await expect(rightEdge).toHaveAttribute('aria-label', 'إظهار الشريط الجانبي الأيمن')

  await rightEdge.click()
  await expect(rightEdge).toHaveAttribute('aria-label', 'إخفاء الشريط الجانبي الأيمن')
  await expect(sessionSearch).toBeVisible()

  const initialSearchBox = await sessionSearch.boundingBox()
  expect(initialSearchBox).not.toBeNull()
  expect(initialSearchBox!.x).toBeGreaterThan((await page.evaluate(() => window.innerWidth)) / 2)

  await flip.click()
  await expect(leftEdge).toHaveAttribute('aria-label', 'إخفاء الشريط الجانبي الأيسر')
  await expect(rightEdge).toHaveAttribute('aria-label', 'إظهار الشريط الجانبي الأيمن')

  const flippedSearchBox = await sessionSearch.boundingBox()
  expect(flippedSearchBox).not.toBeNull()
  expect(flippedSearchBox!.x + flippedSearchBox!.width).toBeLessThan(
    (await page.evaluate(() => window.innerWidth)) / 2,
  )

  await leftEdge.click()
  await expect(sessionSearch).toBeHidden()
  await expect(leftEdge).toHaveAttribute('aria-label', 'إظهار الشريط الجانبي الأيسر')
})

test('RTL titlebar controls remain reachable at the narrow responsive breakpoint', async () => {
  const page = fixture!.page

  await fixture!.app.evaluate(({ BrowserWindow }) => {
    BrowserWindow.getAllWindows()[0]?.setSize(720, 700)
  })
  await expect.poll(() => page.evaluate(() => window.innerWidth)).toBeLessThanOrEqual(720)

  const leftEdge = page.getByRole('button', { name: /الشريط الجانبي الأيسر/ })
  const rightEdge = page.getByRole('button', { name: /الشريط الجانبي الأيمن/ })
  const flip = page.getByRole('button', { name: 'تبديل جانبي الشريطين' })

  await expect(leftEdge).toBeVisible()
  await expect(rightEdge).toBeVisible()
  await expect(flip).toBeVisible()

  const [leftBox, rightBox] = await Promise.all([leftEdge.boundingBox(), rightEdge.boundingBox()])
  expect(leftBox).not.toBeNull()
  expect(rightBox).not.toBeNull()
  expect(leftBox!.x + leftBox!.width).toBeLessThan(rightBox!.x)
})
