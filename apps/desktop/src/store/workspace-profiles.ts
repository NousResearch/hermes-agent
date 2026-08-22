import type { Codec } from '@/lib/persisted'

import { normalizeProfileKey } from './profile'

// ── Workspace ↔ profile bindings (#64221) ────────────────────────────────────
// Pure contract for the renderer-owned preference that binds named profiles to
// a sidebar project ("workspace"): inside a bound workspace the profile rail
// promotes those profiles above an always-present Shared section. This module
// holds only shapes + pure policy (normalization, stale-read safety); the
// persisted atom and its mutations live beside the other workspace state in
// `store/projects.ts`. Nothing here touches profile activation or routing — a
// binding changes which squares are visible, never which backend answers.
//
// Scope (AGENTS.md): one desktop-global key. Bindings are few and keyed by the
// durable project id (`p_<hex>` / auto-project path id), so per-workspace
// entries disambiguate themselves; this is cosmetic state like profile colors,
// not connection-scoped truth.

export interface WorkspaceProfileBindings {
  [workspaceId: string]: string[]
}

/** localStorage key; declares its desktop-global scope in the key itself. */
export const WORKSPACE_PROFILE_BINDINGS_KEY = 'hermes.desktop.workspaceProfileBindings'

/**
 * Canonical form: trimmed non-empty workspace ids → deduped canonical profile
 * keys in first-seen order. Anything else (a non-object payload, a non-array
 * member, non-string members) is dropped, so stale/corrupt persisted entries
 * are inert at read time instead of crashing or activating filtering. An entry
 * whose bound set normalizes to empty is dropped entirely: "has bindings" must
 * stay falsy so the rail falls back to today's unfiltered render.
 */
export function sanitizeWorkspaceProfileBindings(value: unknown): WorkspaceProfileBindings {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const clean: WorkspaceProfileBindings = {}

  for (const [rawId, rawNames] of Object.entries(value)) {
    const workspaceId = rawId.trim()

    if (!workspaceId || !Array.isArray(rawNames)) {
      continue
    }

    const bound = [
      ...new Set(
        rawNames
          .filter((name): name is string => typeof name === 'string' && name.trim().length > 0)
          .map(name => normalizeProfileKey(name))
      )
    ]

    if (bound.length > 0) {
      clean[workspaceId] = bound
    }
  }

  return clean
}

/**
 * THE resolver for "which profiles are bound to this workspace": canonical
 * keys in stored order, or null when filtering is off (no active workspace, an
 * unknown/synthetic id, or no bindings). The rail reads through this and only
 * this, so it can never disagree with what the picker wrote.
 */
export function workspaceBoundProfiles(
  bindings: WorkspaceProfileBindings,
  workspaceId: null | string | undefined
): null | string[] {
  const bound = workspaceId ? bindings[workspaceId.trim()] : undefined

  return bound && bound.length > 0 ? [...bound] : null
}

/**
 * Codec for persistentAtom: decode parses then sanitizes the untrusted stored
 * shape (a malformed payload throws into persistentAtom's fallback path); encode
 * re-sanitizes on write and removes the key once no workspace holds bindings.
 */
export const workspaceProfileBindingsCodec: Codec<WorkspaceProfileBindings> = {
  decode: raw => sanitizeWorkspaceProfileBindings(JSON.parse(raw) as unknown),
  encode: value => {
    const clean = sanitizeWorkspaceProfileBindings(value)

    return Object.keys(clean).length === 0 ? null : JSON.stringify(clean)
  }
}
