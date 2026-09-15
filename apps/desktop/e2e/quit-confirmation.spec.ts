import * as fs from 'node:fs'
import * as path from 'node:path'

import type { App, MessageBoxOptions, MessageBoxReturnValue } from 'electron'

import { buildAppEnv, createSandbox, launchDesktop, type Sandbox } from './fixtures'
import { allowErrorBanners, type ElectronApplication, expect, type Page, test } from './test'

type QuitMode = 'never' | 'while-working' | 'always'

interface QuitBridge {
  settings: {
    getDefaultProjectDir: () => Promise<{ defaultLabel: string; dir: null | string; resolvedCwd: string }>
    getQuitConfirmation: () => Promise<QuitMode>
    setQuitConfirmation: (mode: QuitMode) => Promise<QuitMode>
  }
  openWindow: () => Promise<unknown>
  setActiveWork: (work: { count: number; titles: string[] }) => void
}

interface QuitProbe {
  closeEvents: Array<{ kind: 'close' | 'closed'; id: number; prevented?: boolean }>
  dialogs: Array<{ parentId: number | null; options: MessageBoxOptions }>
  resolve: ((result: MessageBoxReturnValue) => void) | null
  willQuit: number
}

interface ObservedApp extends App {
  quitProbe: QuitProbe
}

async function stopApp(app: ElectronApplication): Promise<void> {
  if (app.process().exitCode !== null || app.process().signalCode !== null) {
    return
  }

  const closed = Promise.all([app.waitForEvent('close'), app.context().waitForEvent('close')])
  await app
    .evaluate(({ app: nativeApp }) => {
      setImmediate(() => nativeApp.exit(0))
    })
    .catch(() => undefined)
  await closed
}

async function launch(sandbox: Sandbox): Promise<{ app: ElectronApplication; page: Page }> {
  const launched = await launchDesktop(
    buildAppEnv(sandbox, {
      HERMES_DESKTOP_BOOT_FAKE_ERROR: 'Quit lifecycle fixture: local backend intentionally not started',
      HERMES_DESKTOP_DEV_SERVER: '',
      HERMES_DESKTOP_SKIP_QUIT_CONFIRM: '0',
      HERMES_SKIP_INTRO: '1'
    })
  )

  try {
    // Let the renderer publish its initial idle report before the fixture
    // reports work; otherwise startup can overwrite the scenario mid-quit.
    await launched.page.waitForLoadState('domcontentloaded')
    await launched.page.waitForFunction(() =>
      Boolean((window as unknown as { hermesDesktop?: QuitBridge }).hermesDesktop?.settings)
    )

    // Only the native dialog's answer is controlled; production main/preload,
    // persistence, app.quit(), and BrowserWindow.close() stay on their real paths.
    await launched.app.evaluate(
      ({ app, dialog }, receiptPath) => {
        const { writeFileSync } = process.getBuiltinModule('node:fs')

        const probe: QuitProbe = { closeEvents: [], dialogs: [], resolve: null, willQuit: 0 }

        ;(app as ObservedApp).quitProbe = probe
        dialog.showMessageBox = ((...args: unknown[]) => {
          const hasParent = args.length === 2
          probe.dialogs.push({
            parentId: hasParent ? (args[0] as { id: number }).id : null,
            options: args[hasParent ? 1 : 0] as MessageBoxOptions
          })

          return new Promise<MessageBoxReturnValue>(resolve => {
            probe.resolve = resolve
          })
        }) as typeof dialog.showMessageBox
        app.on('will-quit', () => {
          probe.willQuit += 1
          writeFileSync(receiptPath, JSON.stringify({ dialogs: probe.dialogs, willQuit: probe.willQuit }))
        })
      },
      path.join(sandbox.root, 'quit-receipt.json')
    )

    return launched
  } catch (error) {
    await stopApp(launched.app)
    throw error
  }
}

async function preference(page: Page, mode?: QuitMode): Promise<QuitMode> {
  return page.evaluate(async requested => {
    const desktop = (window as unknown as { hermesDesktop: QuitBridge }).hermesDesktop

    if (requested !== undefined) {
      await desktop.settings.setQuitConfirmation(requested)
    }

    return desktop.settings.getQuitConfirmation()
  }, mode)
}

async function publishWork(page: Page, count: number): Promise<void> {
  await page.evaluate(async activeCount => {
    const desktop = (window as unknown as { hermesDesktop: QuitBridge }).hermesDesktop
    desktop.setActiveWork({ count: activeCount, titles: activeCount ? ['Unfinished quit-regression work'] : [] })
    // Flush the report without reading or initializing the quit preference.
    await desktop.settings.getDefaultProjectDir()
  }, count)
}

async function requestQuit(app: ElectronApplication): Promise<void> {
  await app.evaluate(({ app: nativeApp }) => {
    setImmediate(() => nativeApp.quit())
  })
}

async function respond(app: ElectronApplication, response: number): Promise<void> {
  await app.evaluate(({ app: nativeApp }, answer) => {
    const probe = (nativeApp as ObservedApp).quitProbe
    const resolve = probe.resolve
    probe.resolve = null

    if (!resolve) {
      throw new Error('No quit confirmation is waiting for an answer')
    }

    setImmediate(() => resolve({ response: answer, checkboxChecked: false }))
  }, response)
}

async function dialogCount(app: ElectronApplication): Promise<number> {
  return app.evaluate(({ app: nativeApp }) => (nativeApp as ObservedApp).quitProbe.dialogs.length)
}

async function cancelRepeatedQuit(app: ElectronApplication, page: Page, closeWindow: boolean): Promise<void> {
  const window = await app.browserWindow(page)
  const before = await dialogCount(app)

  const request = closeWindow ? () => window.evaluate(nativeWindow => nativeWindow.close()) : () => requestQuit(app)

  await request()
  await expect.poll(() => dialogCount(app)).toBe(before + 1)
  expect(
    await app.evaluate(({ app: nativeApp }) => (nativeApp as ObservedApp).quitProbe.dialogs.at(-1)?.options)
  ).toMatchObject({ cancelId: 0, defaultId: 0 })
  await requestQuit(app)
  await request()

  expect(await dialogCount(app)).toBe(before + 1)
  expect(await window.evaluate(nativeWindow => nativeWindow.isDestroyed())).toBe(false)
  expect(await app.evaluate(({ app: nativeApp }) => (nativeApp as ObservedApp).quitProbe.willQuit)).toBe(0)

  await respond(app, 0)
  expect(await window.evaluate(nativeWindow => nativeWindow.isDestroyed())).toBe(false)
  expect(await preference(page)).toBeDefined()
  expect(page.isClosed()).toBe(false)
}

async function expectExit(app: ElectronApplication, sandbox: Sandbox, acceptPrompt: boolean): Promise<void> {
  const expectedDialogs = await dialogCount(app)
  const closed = Promise.all([app.waitForEvent('close'), app.context().waitForEvent('close')])

  if (acceptPrompt) {
    await respond(app, 1)
  } else {
    await requestQuit(app)
  }

  await closed

  const receipt = JSON.parse(fs.readFileSync(path.join(sandbox.root, 'quit-receipt.json'), 'utf8')) as QuitProbe
  await test.info().attach('native-quit-receipt', { body: JSON.stringify(receipt), contentType: 'application/json' })
  expect(receipt.willQuit).toBe(1)
  expect(receipt.dialogs).toHaveLength(expectedDialogs)
}

async function closePeerWindows(app: ElectronApplication, page: Page): Promise<void> {
  const opened = app.waitForEvent('window')
  await page.evaluate(() => (window as unknown as { hermesDesktop: QuitBridge }).hermesDesktop.openWindow())
  const peer = await opened
  await peer.waitForLoadState('domcontentloaded')
  const first = await app.browserWindow(page)
  const second = await app.browserWindow(peer)
  const ids = await Promise.all([first.evaluate(win => win.id), second.evaluate(win => win.id)])
  const before = await dialogCount(app)

  await app.evaluate(({ app: nativeApp, BrowserWindow }, windowIds) => {
    const probe = (nativeApp as ObservedApp).quitProbe
    const windows = windowIds.map(id => BrowserWindow.fromId(id)!)

    for (const win of windows) {
      const id = win.id
      win.on('close', event => probe.closeEvents.push({ kind: 'close', id, prevented: event.defaultPrevented }))
      win.on('closed', () => probe.closeEvents.push({ kind: 'closed', id }))
    }

    windows[0].focus()

    // Same native event-loop callback: the first window can still be focused
    // and closing when the second window needs a surviving dialog parent.
    for (const win of windows) {
      win.close()
    }
  }, ids)

  const expectedClosed = process.platform === 'darwin' ? 2 : 1
  await expect
    .poll(() =>
      app.evaluate(
        ({ app: nativeApp }) =>
          (nativeApp as ObservedApp).quitProbe.closeEvents.filter(event => event.kind === 'closed').length
      )
    )
    .toBe(expectedClosed)
  const events = await app.evaluate(({ app: nativeApp }) => (nativeApp as ObservedApp).quitProbe.closeEvents)
  await test.info().attach('native-window-close-order', {
    body: JSON.stringify({ platform: process.platform, events }),
    contentType: 'application/json'
  })

  if (process.platform === 'darwin') {
    expect(await dialogCount(app)).toBe(before)
  } else {
    expect(await dialogCount(app)).toBe(before + 1)
    expect(await second.evaluate(win => win.isDestroyed())).toBe(false)
    expect(
      await app.evaluate(({ app: nativeApp }) => (nativeApp as ObservedApp).quitProbe.dialogs.at(-1)?.parentId)
    ).toBe(ids[1])
    await requestQuit(app)
    expect(await dialogCount(app)).toBe(before + 1)
    await respond(app, 0)
    expect(await preference(peer)).toBe('always')
  }
}

test('quit preference survives restart and native quit waits for one answer before destroying windows', async () => {
  test.setTimeout(180_000)
  allowErrorBanners()
  const sandbox = createSandbox('quit-confirmation')
  const preferencePath = path.join(sandbox.userDataDir, 'quit-confirmation.json')
  fs.writeFileSync(preferencePath, '{truncated preference')
  let running: ElectronApplication | null = null

  try {
    let launched = await launch(sandbox)
    running = launched.app
    expect(await preference(launched.page)).toBe('while-working')

    await test.step('invalid values and failed writes preserve the last authoritative preference', async () => {
      await expect(preference(launched.page, 'sometimes' as QuitMode)).rejects.toThrow(
        'Invalid quit confirmation preference'
      )
      expect(await preference(launched.page)).toBe('while-working')
      const blocker = `${preferencePath}.tmp`
      fs.mkdirSync(blocker)

      try {
        await expect(preference(launched.page, 'never')).rejects.toThrow()
        expect(await preference(launched.page)).toBe('while-working')
      } finally {
        fs.rmdirSync(blocker)
      }
    })

    await test.step('the default protects active work and cancellation leaves the window usable', async () => {
      await publishWork(launched.page, 1)
      await cancelRepeatedQuit(launched.app, launched.page, process.platform !== 'darwin')
    })

    await test.step('Always protects idle quits and respects native window-close semantics', async () => {
      expect(await preference(launched.page, 'always')).toBe('always')
      await publishWork(launched.page, 0)
      await cancelRepeatedQuit(launched.app, launched.page, process.platform !== 'darwin')
      await closePeerWindows(launched.app, launched.page)

      if (process.platform === 'darwin') {
        const before = await dialogCount(launched.app)
        await requestQuit(launched.app)
        await expect.poll(() => dialogCount(launched.app)).toBe(before + 1)
        expect(
          await launched.app.evaluate(
            ({ app: nativeApp }) => (nativeApp as ObservedApp).quitProbe.dialogs.at(-1)?.parentId
          )
        ).toBeNull()
        await requestQuit(launched.app)
        expect(await dialogCount(launched.app)).toBe(before + 1)
        await respond(launched.app, 0)
      }

      const before = await dialogCount(launched.app)
      await requestQuit(launched.app)
      await expect.poll(() => dialogCount(launched.app)).toBe(before + 1)
      await expectExit(launched.app, sandbox, true)
      running = null
    })

    for (const scenario of [
      { title: 'saved Always protects idle quit before Settings reads', persisted: 'always', mode: 'never', work: 1 },
      { title: 'saved Never allows active quit before Settings reads', persisted: 'never', mode: null, work: 1 },
      { title: 'while-working allows idle quit after restart', persisted: 'never', mode: 'while-working', work: 0 }
    ] as const) {
      await test.step(scenario.title, async () => {
        launched = await launch(sandbox)
        running = launched.app

        if (scenario.persisted === 'always') {
          await publishWork(launched.page, 0)
          await cancelRepeatedQuit(launched.app, launched.page, false)
          expect(await preference(launched.page)).toBe(scenario.persisted)
        }

        if (scenario.mode !== null) {
          expect(await preference(launched.page, scenario.mode)).toBe(scenario.mode)
        }

        await publishWork(launched.page, scenario.work)
        await expectExit(launched.app, sandbox, false)
        running = null
      })
    }
  } finally {
    // A failed assertion must not leave the deliberately unanswered dialog
    // or its isolated app alive. app.exit is cleanup only, never the assertion.
    if (running) {
      await stopApp(running)
    }

    sandbox.cleanup()
  }
})
