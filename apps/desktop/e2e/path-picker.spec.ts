import { expect, test } from '@playwright/test'

import { setupNoProvider } from './fixtures'

test('native picker validates defaults before applying the Downloads fallback', async () => {
  const fixture = await setupNoProvider()

  try {
    const downloads = await fixture.app.evaluate(({ app, dialog }) => {
      dialog.showOpenDialog = async (...args: unknown[]) => ({
        canceled: false,
        filePaths: [JSON.stringify(args.at(-1))]
      })

      return app.getPath('downloads')
    })

    await fixture.page.waitForFunction(() => Boolean((window as any).hermesDesktop?.selectPaths))

    const dialogs = await fixture.page.evaluate(async explicitPath => {
      const options = [
        { fallbackToDownloads: true },
        { defaultPath: { invalid: true }, fallbackToDownloads: true },
        { defaultPath: 42, fallbackToDownloads: true },
        { defaultPath: '   ', fallbackToDownloads: true },
        { defaultPath: explicitPath, fallbackToDownloads: true },
        { defaultPath: { invalid: true } },
        {}
      ]

      const results = []

      for (const option of options) {
        const [serialized] = await (window as any).hermesDesktop.selectPaths(option)

        results.push(JSON.parse(serialized))
      }

      return results
    }, fixture.sandbox.root)

    expect(dialogs.map(options => options.defaultPath)).toEqual([
      downloads,
      downloads,
      downloads,
      downloads,
      fixture.sandbox.root,
      undefined,
      undefined
    ])
    expect(dialogs.every(options => options.properties.includes('openFile'))).toBe(true)
  } finally {
    await fixture.cleanup()
  }
})
