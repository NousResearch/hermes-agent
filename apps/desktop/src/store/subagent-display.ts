/**
 * How delegated subagent runs show up and open in Desktop / the Webapp. Both
 * settings live in config.yaml (per profile), so every window and every
 * browser on the same profile agree and `hermes config set` works too.
 *
 * - `sessions.show_subagents`: `false` (none) | `"active"` | `true` (all).
 *   The backend filters the sidebar listing; this module only maps the value
 *   to the three choices the settings page offers. Mirrors
 *   `hermes_cli.session_listing.resolve_subagent_sidebar_mode`.
 * - `desktop.subagent_open`: `window` (default) pops the child into its own
 *   watch window (a new browser tab in the Webapp); `tab` opens it as an
 *   in-app session tab beside the parent chat, which stays open.
 */
import { atom } from 'nanostores'

export type SubagentSidebarMode = 'active' | 'all' | 'none'

export type SubagentOpenMode = 'tab' | 'window'

const TRUTHY_WORDS = new Set(['1', 'on', 'true', 'y', 'yes'])

export function subagentSidebarMode(value: unknown): SubagentSidebarMode {
  if (typeof value === 'string') {
    const word = value.trim().toLowerCase()

    return word === 'active' ? 'active' : TRUTHY_WORDS.has(word) ? 'all' : 'none'
  }

  return value === true || (typeof value === 'number' && value !== 0) ? 'all' : 'none'
}

/** The `sessions.show_subagents` value a mode writes: booleans stay booleans. */
export function showSubagentsConfigValue(mode: SubagentSidebarMode): 'active' | boolean {
  return mode === 'active' ? 'active' : mode === 'all'
}

export function normalizeSubagentOpenMode(value: unknown): SubagentOpenMode {
  return typeof value === 'string' && value.trim().toLowerCase() === 'tab' ? 'tab' : 'window'
}

export const $subagentOpenMode = atom<SubagentOpenMode>('window')

export function setSubagentOpenModeFromConfig(value: unknown): void {
  $subagentOpenMode.set(normalizeSubagentOpenMode(value))
}
