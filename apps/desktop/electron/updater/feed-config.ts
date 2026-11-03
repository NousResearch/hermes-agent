import { readFileSync } from 'node:fs'

import { load } from 'js-yaml'

import { channelPublicBase } from './channel-protocol'

/**
 * The feed base a bundled install downloads its update descriptor from:
 * config.yaml's `updates.desktop_feed_base_url`, else '' (the OS-registered
 * App Installer source / no feed). Config is the one override: a second env
 * source would let a stray variable silently redirect updates. A configured
 * value is held to the same rule as every other update origin
 * (channelPublicBase: HTTPS or loopback, no credentials/traversal) — the
 * descriptor is fetched and opened unverified, so a plain-http base would
 * hand a network attacker the update source. Throws on a bad value rather
 * than silently falling back, so the misconfiguration is named.
 */
export function resolveFeedBaseUrl(configured: string): string {
  const value: string = configured.trim()

  if (!value) {
    return ''
  }

  channelPublicBase(value)

  return value
}

export function readUpdatesFeedBaseFromConfig(configPath: string): string {
  try {
    // YAML permits scalar and sequence roots; validate the nested string at this I/O boundary.
    const config: unknown = load(readFileSync(configPath, 'utf8'))

    if (!config || typeof config !== 'object' || !('updates' in config)) {
      return ''
    }

    const updates: unknown = config.updates

    if (!updates || typeof updates !== 'object' || !('desktop_feed_base_url' in updates)) {
      return ''
    }

    const value: unknown = updates.desktop_feed_base_url

    return typeof value === 'string' ? value.trim() : ''
  } catch {
    // An absent/unreadable override leaves the registered feed eligible.
    return ''
  }
}
