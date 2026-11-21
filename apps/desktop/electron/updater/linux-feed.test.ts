import { GenericProvider } from 'electron-updater/out/providers/GenericProvider'
import { afterEach, describe, expect, it, vi } from 'vitest'

import feedContract from '../../update-feed.cjs'

afterEach((): void => {
  vi.unstubAllEnvs()
})

describe('Linux feed contract', () => {
  it.each(['x64', 'arm64'])('names the %s channel file electron-updater requests', async arch => {
    vi.stubEnv('TEST_UPDATER_ARCH', arch)
    const requested: string[] = []

    const executor = {
      request: async (options: { protocol: string; hostname: string; path: string }): Promise<string> => {
        requested.push(`${options.protocol}//${options.hostname}${options.path}`)

        return 'version: 0.99.1\nfiles:\n  - url: Hermes.AppImage\n    sha512: x\npath: Hermes.AppImage\nsha512: x\n'
      }
    }

    for (const channel of ['stable', 'canary']) {
      const feed = feedContract.linuxFeed(channel, arch)
      type Options = ConstructorParameters<typeof GenericProvider>

      const provider = new GenericProvider(
        { provider: 'generic', url: `https://updates.example/${feed.directory}/` },
        { channel: feed.channel } as Options[1],
        { platform: 'linux', isUseMultipleRangeRequest: true, executor: executor as unknown as Options[2]['executor'] }
      )

      await provider.getLatestVersion()
    }

    const suffix = arch === 'x64' ? '' : `-${arch}`
    expect(requested).toEqual([
      `https://updates.example/releases/linux/stable/stable-linux${suffix}.yml`,
      `https://updates.example/releases/linux/canary/canary-linux${suffix}.yml`
    ])
    expect(feedContract.linuxFeed('canary', arch).fileName).toBe(`canary-linux${suffix}.yml`)
  })
})
