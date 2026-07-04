import type { Translations } from './types'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Deep-merge untrusted user overrides on top of a bundled catalog, preserving
 * the catalog's shape and value types.
 *
 * The override source is a plain JSON object the user hand-writes (or that a
 * tool produces) under `<hermes-home>/locale-overrides/desktop/<lang>.json`. It
 * can only carry strings, so the merge is deliberately conservative:
 *
 * - a **string** leaf in the base is replaced only by a **string** override;
 * - nested objects are merged recursively;
 * - **function** leaves (e.g. ``open => `... ${open} ...` ``) are left intact —
 *   a JSON string can't safely replace a callable, and doing so would crash the
 *   call sites that invoke it;
 * - keys present in the override but **not** in the base are ignored, so a typo
 *   or stale key can never inject unknown entries.
 *
 * The result therefore always has the exact same key set and value types as
 * `base`; only string leaves can change. Anything malformed is a no-op.
 */
export function applyLocaleOverrides(base: Translations, overrides: unknown): Translations {
  if (!isRecord(overrides)) {
    return base
  }

  return mergeRecord(base as unknown as Record<string, unknown>, overrides) as unknown as Translations
}

function mergeRecord(
  base: Record<string, unknown>,
  overrides: Record<string, unknown>
): Record<string, unknown> {
  let changed = false
  const result: Record<string, unknown> = { ...base }

  for (const [key, overrideValue] of Object.entries(overrides)) {
    if (!(key in base)) {
      continue
    }

    const baseValue = base[key]

    if (typeof baseValue === 'string') {
      if (typeof overrideValue === 'string' && overrideValue !== baseValue) {
        result[key] = overrideValue
        changed = true
      }
      continue
    }

    if (isRecord(baseValue) && isRecord(overrideValue)) {
      const mergedChild = mergeRecord(baseValue, overrideValue)
      if (mergedChild !== baseValue) {
        result[key] = mergedChild
        changed = true
      }
    }
    // functions, arrays, and type mismatches: keep the base value untouched.
  }

  // Return the original reference when nothing changed so callers can cheaply
  // detect a no-op merge (used to avoid pointless re-renders).
  return changed ? result : base
}
