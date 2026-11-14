// Seed an isolated HERMES_HOME with just enough config (NOT sessions) so the
// spawned instance reaches an empty chat view instead of the onboarding wizard.
//
// The isolated home gets its own gateway, and a gateway starts every messaging
// platform that has activation evidence: a config block (top-level `<name>:`,
// `platforms.<name>`, `gateway.platforms.<name>`), credentials in `.env`, or a
// credential-pool record in `auth.json`. Seeded verbatim, the throwaway home
// would open a second Telegram/Discord/iMessage connection: a sidecar on a
// fixed port reaps the user's live one and inbound messages go to the wrong
// process. So every platform the gateway knows is pinned `enabled: false` under
// top-level `platforms:` — an explicit disable outranks .env and auth.json
// evidence (gateway/config_env.py `_enable_from_env`, `_enable_plugin_platform`).
// Model and auth config stay as-is.

import { copyFileSync, existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

import { parse, stringify } from 'yaml'

// gateway/config.py `Platform` members that are messaging adapters (`local` is the CLI itself).
const CORE_PLATFORMS = [
  'api_server',
  'bluebubbles',
  'dingtalk',
  'discord',
  'email',
  'feishu',
  'matrix',
  'mattermost',
  'msgraph_webhook',
  'qqbot',
  'relay',
  'signal',
  'slack',
  'sms',
  'telegram',
  'webhook',
  'wecom',
  'wecom_callback',
  'weixin',
  'whatsapp',
  'whatsapp_cloud'
]

/** Bundled plugin platforms, named by directory like `Platform._scan_bundled_plugin_platforms`. */
function bundledPluginPlatforms(repoRoot) {
  const dir = join(repoRoot, 'plugins', 'platforms')

  if (!existsSync(dir)) {
    return []
  }

  return readdirSync(dir, { withFileTypes: true })
    .filter(entry => entry.isDirectory() && existsSync(join(dir, entry.name, '__init__.py')))
    .map(entry => entry.name.toLowerCase())
}

/** `configText` with every platform in `platformNames` pinned off under top-level `platforms:`. */
export function withPlatformsDisabled(configText, platformNames) {
  const config = parse(configText) ?? {}
  const platforms = config.platforms && typeof config.platforms === 'object' ? config.platforms : {}
  for (const name of platformNames) {
    const block = platforms[name]
    platforms[name] = { ...(block && typeof block === 'object' ? block : {}), enabled: false }
  }

  config.platforms = platforms

  return stringify(config)
}

export function seedConfigFrom(sourceHome, targetHome, repoRoot) {
  if (!existsSync(sourceHome)) {
    return
  }

  const platformNames = [...new Set([...CORE_PLATFORMS, ...bundledPluginPlatforms(repoRoot)])]
  const config = join(sourceHome, 'config.yaml')
  const configText = existsSync(config) ? readFileSync(config, 'utf8') : ''

  writeFileSync(join(targetHome, 'config.yaml'), withPlatformsDisabled(configText, platformNames))

  for (const name of ['.env', 'auth.json']) {
    const from = join(sourceHome, name)

    if (existsSync(from)) {
      copyFileSync(from, join(targetHome, name))
    }
  }
}
