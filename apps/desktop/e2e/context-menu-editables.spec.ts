/**
 * Context-menu edit verbs on real editables — the regressions jsdom cannot
 * catch, exercised against the real renderer (real radix focus trap, real
 * React unmount timing, real selection).
 *
 * The class under test: "Select all" from the app context menu must act on
 * the FIELD the menu was opened on, never on the surrounding transcript.
 * The first fix (focus-restore before dispatch) passed unit tests and still
 * failed live because the radix trap steals focus back; the second fix runs
 * selection renderer-side after the trap unmounts. These tests pin the
 * observable outcome, not the mechanism.
 *
 * Menu items are addressed by accessible-name PREFIX (`/^Copy/`): the name
 * includes the shortcut suffix ("Copy Ctrl+V" / "Copy ⌘V"), which is also
 * host-dependent.
 */

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('select all from the composer context menu selects the draft, not the chat', async () => {
  const page = fixture!.page
  const composer = page.locator('[data-slot="composer-rich-input"]').first()

  // Put a message into the transcript so there is chat text a document-wide
  // select-all WOULD grab — the bug this test exists to catch. Wait for the
  // mock reply to COMPLETE: while the turn is busy the composer is in its
  // steer shape and a typed draft does not land in it.
  await composer.click()
  await composer.pressSequentially('transcript anchor message')
  await page.keyboard.press('Enter')
  await page.waitForFunction(() => (document.body.textContent ?? '').includes('mock inference server'), undefined, {
    timeout: 60_000
  })

  // Draft text in the composer, then right-click it.
  await composer.click()
  await composer.pressSequentially('draft under selection')
  await composer.click({ button: 'right' })

  const selectAll = page.getByRole('menuitem', { name: /^Select all/ })

  await selectAll.waitFor({ state: 'visible', timeout: 10_000 })
  await selectAll.click()

  // The selection must live inside the composer and cover exactly the draft.
  await expect
    .poll(
      () =>
        page.evaluate(() => {
          const selection = window.getSelection()
          const editable = document.querySelector('[data-slot="composer-rich-input"]')

          if (!selection || selection.rangeCount === 0 || !editable) {
            return { inside: false, text: '' }
          }

          return {
            inside: editable.contains(selection.getRangeAt(0).commonAncestorContainer),
            text: selection.toString()
          }
        }),
      { timeout: 10_000 }
    )
    .toEqual({ inside: true, text: 'draft under selection' })

  // Clear the draft so later tests start clean.
  await page.keyboard.press('Delete')
})

test('cut, copy, and select all gray out in an empty composer', async () => {
  const page = fixture!.page
  const composer = page.locator('[data-slot="composer-rich-input"]').first()

  await composer.click()
  await composer.click({ button: 'right' })

  const selectAll = page.getByRole('menuitem', { name: /^Select all/ })

  await selectAll.waitFor({ state: 'visible', timeout: 10_000 })

  await expect(selectAll).toHaveAttribute('data-disabled', /.*/)
  await expect(page.getByRole('menuitem', { name: /^Cut/ })).toHaveAttribute('data-disabled', /.*/)
  await expect(page.getByRole('menuitem', { name: /^Copy/ })).toHaveAttribute('data-disabled', /.*/)

  await page.keyboard.press('Escape')
})

test('paste enables when the clipboard holds text', async () => {
  const page = fixture!.page
  const composer = page.locator('[data-slot="composer-rich-input"]').first()

  // The empty-clipboard branch stays in the unit suite: the e2e app shares
  // the SYSTEM clipboard, and writeText('') does not reliably clear it.
  await page.evaluate(() =>
    (
      window as unknown as { hermesDesktop?: { writeClipboard?: (text: string) => Promise<boolean> } }
    ).hermesDesktop?.writeClipboard?.('clipboard payload')
  )
  await composer.click()
  await composer.click({ button: 'right' })

  const paste = page.getByRole('menuitem', { name: /^Paste/ })

  await paste.waitFor({ state: 'visible', timeout: 10_000 })

  // The clipboard probe is an async IPC — the item enables when it lands.
  await expect.poll(() => paste.getAttribute('data-disabled'), { timeout: 10_000 }).toBeNull()

  await page.keyboard.press('Escape')
})
