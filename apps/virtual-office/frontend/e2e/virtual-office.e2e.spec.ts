import { expect, test } from '@playwright/test';

test('virtual office operator workflow', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Main Office Dashboard' })).toBeVisible();
  await expect(page.getByText('Codex workdir:')).toContainText('D:\\Codex');

  await page.getByRole('link', { name: 'Settings' }).click();
  await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
  await page.getByRole('button', { name: 'Save Settings' }).click();
  await expect(page.getByText('Saved at')).toBeVisible();

  await page.getByRole('link', { name: 'Task Board' }).click();
  await expect(page.getByRole('heading', { name: 'Task Board' })).toBeVisible();

  const stamp = Date.now();
  await page.getByLabel('Title').fill(`E2E Task ${stamp}`);
  await page.getByLabel('Goal').fill('Reply with exactly E2E_OK and nothing else.');
  await page.getByLabel('Context').fill('Browser harness verification.');
  await page.getByRole('button', { name: 'Create + Run' }).click();

  await expect(page.getByRole('heading', { name: 'Task Detail' })).toBeVisible();
  await expect(page.getByText(/^E2E_OK$/).first()).toBeVisible();

  await page.getByRole('link', { name: 'Handoff Logs' }).click();
  await expect(page.getByRole('heading', { name: 'Handoff Logs' })).toBeVisible();
  await expect(page.locator('pre').first()).toContainText('E2E_OK');

  await page.getByRole('link', { name: 'Console Logs' }).click();
  await expect(page.getByRole('heading', { name: 'Console Logs' })).toBeVisible();
  await page.getByRole('button', { name: /INFO/i }).first().click();
  await expect(page.getByRole('heading', { name: 'Log Detail' })).toBeVisible();
  await expect(page.locator('aside')).toContainText('task_id');

  await page.getByRole('link', { name: 'Trade Room' }).click();
  await expect(page.getByRole('heading', { name: 'Trade Room' })).toBeVisible();
  await page.getByPlaceholder('Ask Codex to review market context, summarize risk, or suggest next steps.').fill('Reply with exactly TRADE_E2E_OK and nothing else.');
  await page.getByRole('button', { name: 'Run Codex Analysis' }).click();
  await expect(page.getByText(/^TRADE_E2E_OK$/)).toBeVisible();
});
