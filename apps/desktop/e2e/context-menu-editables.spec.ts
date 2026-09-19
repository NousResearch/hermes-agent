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

import { type ElectronApplication } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

type NativeSpellcheckParams = {
  dictionarySuggestions: string[]
  isEditable: boolean
  misspelledWord: string
  spellcheckEnabled: boolean
}

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

async function installNativeContextMenuCapture(app: ElectronApplication) {
  await app.evaluate(({ BrowserWindow }) => {
    const window = BrowserWindow.getAllWindows()[0]

    if (!window) {
      throw new Error('No BrowserWindow available for native spellcheck test')
    }

    const captured: NativeSpellcheckParams[] = []

    const listener = (
      _event: unknown,
      params: {
        dictionarySuggestions: string[]
        isEditable: boolean
        misspelledWord: string
        spellcheckEnabled: boolean
      }
    ) => {
      captured.push({
        dictionarySuggestions: params.dictionarySuggestions,
        isEditable: params.isEditable,
        misspelledWord: params.misspelledWord,
        spellcheckEnabled: params.spellcheckEnabled
      })
    }

    window.webContents.on('context-menu', listener)
    ;(
      window as unknown as {
        __nativeSpellcheckTest?: { captured: NativeSpellcheckParams[]; listener: typeof listener }
      }
    ).__nativeSpellcheckTest = {
      captured,
      listener
    }
  })
}

async function readNativeContextMenuCapture(app: ElectronApplication) {
  return (await app.evaluate(({ BrowserWindow }) => {
    const window = BrowserWindow.getAllWindows()[0]

    const testState = (
      window as unknown as { __nativeSpellcheckTest?: { captured: NativeSpellcheckParams[] } }
    ).__nativeSpellcheckTest

    return testState?.captured ?? []
  })) as NativeSpellcheckParams[]
}

async function removeNativeContextMenuCapture(app: ElectronApplication) {
  await app.evaluate(({ BrowserWindow }) => {
    const window = BrowserWindow.getAllWindows()[0]

    if (!window) {
      return
    }

    const testState = (
      window as unknown as {
        __nativeSpellcheckTest?: { listener: (...args: never[]) => void }
      }
    ).__nativeSpellcheckTest

    if (!testState) {
      return
    }

    window.webContents.removeListener('context-menu', testState.listener as never)
    delete (window as unknown as { __nativeSpellcheckTest?: unknown }).__nativeSpellcheckTest
  })
}

test('composer enables native spellcheck without autocorrect', async () => {
  const composer = fixture!.page.locator('[data-slot="composer-rich-input"]').first()

  await expect(composer).toHaveAttribute('spellcheck', 'true')
  await expect(composer).toHaveAttribute('autocorrect', 'off')
  await expect(composer).toHaveAttribute('autocapitalize', 'off')
})

function isEnglishAppLocale(locale: string) {
  return /^en([_-]|$)/i.test(locale)
}

test('synthetic noneditable spellcheck emit does not send spellcheck IPC', async () => {
  // Main-process gate only: webContents.emit is not a native right-click.
  // Assert the channel was not sent (sync in installContextMenuBridge).
  // A UI not.toBeVisible would false-pass before a late renderer attach.
  // The native recieve → receive dictionary proof is macOS + English only;
  // this test is the cross-platform contract Linux CI can actually run.
  const { app } = fixture!

  const sentSpellcheck = await app.evaluate(({ BrowserWindow }) => {
    const window = BrowserWindow.getAllWindows()[0]

    if (!window) {
      throw new Error('No BrowserWindow available for synthetic spellcheck guard')
    }

    const sent: string[] = []
    const originalSend = window.webContents.send

    window.webContents.send = ((channel: string, ...args: unknown[]) => {
      sent.push(channel)

      return originalSend.call(window.webContents, channel, ...args)
    }) as typeof window.webContents.send

    try {
      window.webContents.emit('context-menu', {}, {
        dictionarySuggestions: ['receive'],
        isEditable: false,
        misspelledWord: 'recieve',
        spellcheckEnabled: true,
        x: 0,
        y: 0
      })

      return sent.includes('hermes:context-menu-spellcheck')
    } finally {
      window.webContents.send = originalSend
    }
  })

  expect(sentSpellcheck).toBe(false)
})

test('native right-click on recieve offers receive and replaces the word', async () => {
  // Not Linux CI parity. Chromium's misspelledWord / dictionarySuggestions
  // for recieve → receive is a macOS + English app-locale fact. Linux CI
  // proves the cross-platform unit/type contracts and the IPC gate above.
  const { app, page } = fixture!
  const composer = page.locator('[data-slot="composer-rich-input"]').first()

  test.skip(
    process.platform !== 'darwin',
    'Native recieve→receive dictionary proof is macOS-only; not Linux CI parity'
  )

  const appLocale = await app.evaluate(({ app: electronApp }) => electronApp.getLocale())

  test.skip(
    !isEnglishAppLocale(appLocale),
    `Native recieve→receive dictionary proof needs an English app locale (got ${appLocale})`
  )

  let capturing = false

  try {
    await installNativeContextMenuCapture(app)
    capturing = true

    await composer.fill('')
    await composer.click()
    await composer.pressSequentially('recieve', { delay: 80 })
    await expect(composer).toHaveText('recieve')

    const wordPoint = await composer.evaluate(element => {
      const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT)
      const text = walker.nextNode()

      if (!text) {
        throw new Error('Composer has no text node for native spellcheck test')
      }

      const range = document.createRange()
      range.selectNodeContents(text)
      const rect = range.getBoundingClientRect()

      return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }
    })

    await expect
      .poll(
        async () => {
          await page.keyboard.press('Escape')
          await page.mouse.click(wordPoint.x, wordPoint.y, { button: 'right' })
          const captured = await readNativeContextMenuCapture(app)

          return captured.some(
            params =>
              params.isEditable &&
              params.misspelledWord === 'recieve' &&
              params.dictionarySuggestions.includes('receive')
          )
        },
        { timeout: 15_000 }
      )
      .toBe(true)

    const menu = page.getByRole('menu')
    await menu.waitFor({ state: 'visible', timeout: 10_000 })
    await expect(menu.getByRole('menuitem', { name: 'receive' })).toBeVisible()
    await menu.getByRole('menuitem', { name: 'receive' }).click()

    await expect(composer).toHaveText('receive')
  } finally {
    try {
      await page.keyboard.press('Escape')
      await composer.fill('')
    } finally {
      if (capturing) {
        await removeNativeContextMenuCapture(app)
      }
    }
  }
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
