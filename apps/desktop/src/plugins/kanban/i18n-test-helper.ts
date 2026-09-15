/**
 * Resolve a kanban message key against the plugin's own `en` bundle.
 *
 * Tests that render kanban components stub `usePluginI18n` with this instead
 * of registering the bundle for real: registration normally happens in
 * `ctx.i18n.register`, and the registry lives behind an app-internal module a
 * plugin (or its tests) may not import. Mirrors hermes-bots/i18n-test-helper.ts.
 */

import { en } from './i18n'

export function translateKanban(key: string, ...args: unknown[]): string {
  const value = key
    .split('.')
    .reduce<unknown>(
      (node, part) => (node && typeof node === 'object' ? (node as Record<string, unknown>)[part] : undefined),
      en
    )

  if (typeof value === 'function') {
    return String((value as (...params: unknown[]) => string)(...args))
  }

  return typeof value === 'string' ? value : key
}
