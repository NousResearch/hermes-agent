import { expect, test } from '@playwright/test'

test('Workstation resource page renders without page errors', async ({ page }) => {
  const pageErrors: string[] = []
  page.on('pageerror', error => pageErrors.push(error.message))

  await page.goto('/workstation')
  await expect(page.getByRole('heading', { name: 'Workstation', level: 2 })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Browser tasks' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Recent events' })).toBeVisible()
  await expect(page.getByText('No BrowserTask is currently registered.')).toBeVisible()

  expect(pageErrors).toEqual([])
})
