import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { expect, it } from 'vitest'
import feedContract from '../apps/desktop/update-feed.cjs'

const root = fileURLToPath(new URL('../', import.meta.url))

// electron-builder's resolution: the platform's publish, or the top-level one when the platform sets none.
const linuxPublish = (env) => JSON.parse(execFileSync(process.execPath, ['-e',
  "const c=require('./apps/desktop/electron-builder.config.cjs'); console.log(JSON.stringify(c.linux.publish === undefined ? c.publish : c.linux.publish))"], {
  cwd: root,
  encoding: 'utf8',
  env: { ...process.env, CLOUDFLARE_R2_PUBLIC_URL: 'https://updates.example', ...env }
}))

it.each([
  ['v0.28.0', 'stable'],
  ['v0.29.0+canary.20260906T000000Z', 'canary']
])('the AppImage of %s bakes the %s feed the client reads', (tag, channel) => {
  const baked = linuxPublish({ HERMES_DESKTOP_VARIANT: 'bundled', HERMES_PAYLOAD_TAG: tag })
  expect(baked).toEqual([{ provider: 'generic', url: `https://updates.example/releases/linux/${channel}/`, channel }])
  const feed = feedContract.linuxFeed(channel, 'arm64')
  expect(baked[0]).toEqual({ provider: 'generic', url: `https://updates.example/${feed.directory}/`, channel: feed.channel })
})

it('bakes no feed where nothing publishes one', () => {
  expect(linuxPublish({ HERMES_DESKTOP_VARIANT: 'light', HERMES_PAYLOAD_TAG: 'v0.28.0' })).toBeNull()
  expect(linuxPublish({ HERMES_DESKTOP_VARIANT: 'bundled', HERMES_PAYLOAD_TAG: 'v0.28.0', CLOUDFLARE_R2_PUBLIC_URL: '' }))
    .toBeNull()
  expect(linuxPublish({ HERMES_DESKTOP_VARIANT: 'bundled', HERMES_BUILD_COMMIT: 'a'.repeat(40) })).toBeNull()
})
