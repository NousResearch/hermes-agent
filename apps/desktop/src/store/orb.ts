import { atom, computed } from 'nanostores'

import { defaultOrbConfig, type OrbParams, parseOrbConfigUrl } from '@/components/orb/orb-url'
import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'

/**
 * Orb thinking-indicator state. A desktop-local presentation preference:
 * whether the WebGPU orb replaces the standard thinking spinner, and the
 * optional custom configurator URL (the BYO orb) parsed into render params.
 * The URL is validated on read; a bad URL falls back to the built-in orb.
 */

const ORB_ENABLED_KEY = 'hermes.desktop.orb.enabled'
const ORB_CONFIG_URL_KEY = 'hermes.desktop.orb.configUrl'

export const $orbEnabled = atom(storedBoolean(ORB_ENABLED_KEY, false))
export const $orbConfigUrl = atom(storedString(ORB_CONFIG_URL_KEY) ?? '')

export function setOrbEnabled(enabled: boolean): void {
  $orbEnabled.set(enabled)
  persistBoolean(ORB_ENABLED_KEY, enabled)
}

export function setOrbConfigUrl(url: string): void {
  const trimmed = url.trim()
  $orbConfigUrl.set(trimmed)
  persistString(ORB_CONFIG_URL_KEY, trimmed === '' ? null : trimmed)
}

/** Render params: the custom URL when it parses, otherwise the built-in default. */
export const $orbParams = computed($orbConfigUrl, (url): OrbParams => {
  if (!url) {
    return defaultOrbConfig()
  }

  const result = parseOrbConfigUrl(url)

  return 'params' in result ? result.params : defaultOrbConfig()
})

/** Parse error key for the settings UI (null when the URL parses or is empty). */
export const $orbUrlError = computed($orbConfigUrl, (url): null | string => {
  if (!url) {
    return null
  }

  const result = parseOrbConfigUrl(url)

  return 'error' in result ? result.error : null
})
