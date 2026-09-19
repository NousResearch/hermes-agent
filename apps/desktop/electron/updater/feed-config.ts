import { readFileSync } from 'node:fs'

import { load } from 'js-yaml'

export function readUpdatesFeedBaseFromConfig(configPath: string): string {
  try {
    // YAML permits scalar and sequence roots; validate the nested string at this I/O boundary.
    const config: unknown = load(readFileSync(configPath, 'utf8'))

    if (!config || typeof config !== 'object' || !('updates' in config)) {return ''}

    const updates: unknown = config.updates

    if (!updates || typeof updates !== 'object' || !('desktop_feed_base_url' in updates)) {return ''}

    const value: unknown = updates.desktop_feed_base_url

    return typeof value === 'string' ? value.trim() : ''
  } catch {
    // An absent/unreadable override leaves environment and registered feeds eligible.
    return ''
  }
}
