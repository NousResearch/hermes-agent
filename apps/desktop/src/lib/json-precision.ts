/** JSON number precision guard shared by tool-result hydration and display. */

// Strings match whole so digits inside them are skipped; only scan text JSON.parse accepted.
const JSON_STRING_OR_NUMBER_RE = /"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g

/** JSON.parse rounds integers past 2^53 (64-bit chat/user ids) and overflows huge
 *  numbers to Infinity, so re-indenting such text would show a different value. */
export function hasLossyJsonNumber(json: string): boolean {
  for (const [token] of json.matchAll(JSON_STRING_OR_NUMBER_RE)) {
    if (token.startsWith('"')) {
      continue
    }

    const value = Number(token)

    if (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value))) {
      return true
    }
  }

  return false
}
