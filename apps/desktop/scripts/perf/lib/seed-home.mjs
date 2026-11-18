// Seed an isolated HERMES_HOME with just enough config (NOT sessions) so the
// spawned instance reaches an empty chat view instead of the onboarding wizard.
//
// The isolated home gets its own gateway, and a gateway starts every messaging
// platform that has activation evidence: a config block (top-level `<name>:`,
// `platforms.<name>`, `gateway.platforms.<name>`), credentials in `.env`, or a
// credential-pool record in `auth.json`. Seeded verbatim, the throwaway home
// would open a second Telegram/Discord/iMessage connection: a sidecar on a
// fixed port reaps the user's live one and inbound messages go to the wrong
// process. Every known platform is pinned off in `platforms:` and in any
// shorthand block that could override it. Explicit disables also outrank .env
// and auth.json evidence (gateway/config_env.py).
// Model and auth config stay as-is.

import { copyFileSync, existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

import { isAlias, isMap, parseDocument, visit } from 'yaml'

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

// Materialize an aliased mapping before editing it, leaving the anchor's other
// consumers alone. Scalars retain their source spelling and quoting.
function mapping(doc, parent, key) {
  let node = parent.get(key)
  if (isAlias(node)) {
    node = node.resolve(doc).clone()
    visit(node, {
      Node: (_key, child) => {
        if ('anchor' in child) delete child.anchor
      }
    })
    parent.set(key, node)
  }

  return isMap(node) ? node : null
}

/** Pin adapters off without re-serializing unrelated settings through JS values. */
export function withPlatformsDisabled(configText, platformNames) {
  const doc = parseDocument(configText, { version: '1.1', intAsBigInt: true })
  if (doc.errors.length) throw doc.errors[0]
  if (doc.getIn([]) == null) doc.contents = doc.createNode({})

  let platforms = mapping(doc, doc, 'platforms')
  if (!platforms) {
    platforms = doc.createNode({})
    doc.set('platforms', platforms)
  }
  const gateway = mapping(doc, doc, 'gateway')

  for (const name of platformNames) {
    let block = mapping(doc, platforms, name)
    if (!block) {
      block = doc.createNode({})
      platforms.set(name, block)
    }
    block.set('enabled', false)

    // merge_platform_sections applies gateway.<name> last; the shared-key
    // bridge then applies an explicit top-level <name>.enabled.
    for (const parent of [doc, gateway]) {
      const shorthand = parent && mapping(doc, parent, name)
      // toJS sees keys inherited through a YAML merge, as the Python loader does.
      if (shorthand && Object.hasOwn(shorthand.toJS(doc), 'enabled')) shorthand.set('enabled', false)
    }
  }

  // Unwrapped: readDesktopLaunchConfig reads desktop.electron_flags line by line.
  return doc.toString({ lineWidth: 0 })
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
