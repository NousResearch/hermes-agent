/**
 * How a hand-edited Bot Mode flag is read — a dependency-free leaf so every surface can
 * share one answer, including the relay publisher (which runs without the roster's atoms).
 */

/**
 * Mirrors the gateway's `_boolish`: YAML hands back `true`, the int `1`, or a quoted string,
 * and anything unrecognised is NOT set — a typo must never quietly remove a bot from the mesh.
 *
 * Plain JS truthiness is wrong in both directions here: it makes `private: "no"` private, while
 * a strict `=== true` publishes `private: 1` to the relay.
 */
export function isPrivateFlag(value: unknown): boolean {
  if (typeof value === 'boolean') {
    return value
  }

  if (typeof value === 'number') {
    return value === 1
  }

  return typeof value === 'string' && ['1', 'true', 'yes', 'on'].includes(value.trim().toLowerCase())
}
