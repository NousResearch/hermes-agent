import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { expect, it } from 'vitest'
import feedContract from '../apps/desktop/update-feed.cjs'

const root = fileURLToPath(new URL('../', import.meta.url))

it.each([
  ['bundled', 'v0.28.0', 'stable', false],
  ['bundled', 'v0.29.0-canary.20260906000000', 'canary', false],
  ['light', 'v0.28.0', 'stable', true],
  ['light', 'v0.29.0-canary.20260906000000', 'canary', true]
])('packaging and runtime agree for %s %s', (variant, tag, channel, light) => {
  const result = execFileSync(process.execPath, ['-e', "const c=require('./apps/desktop/electron-builder.config.cjs'); console.log(JSON.stringify({publish:c.mac.publish,notarize:c.mac.notarize,targets:c.mac.target}))"], {
    cwd: root,
    encoding: 'utf8',
    env: { ...process.env, HERMES_DESKTOP_VARIANT: variant, HERMES_PAYLOAD_TAG: tag, CLOUDFLARE_R2_PUBLIC_URL: 'https://updates.example' }
  })
  const config = JSON.parse(result)
  const feed = feedContract.darwinFeed(channel, light)
  expect(config.publish).toEqual([{ provider: 'generic', url: `https://updates.example/${feed.directory}/`, channel: feed.channel }])
  expect(config.targets).toContain('zip')
  expect(config.notarize).toBe(false)
})
