/**
 * A hand-edited Bot Mode flag, read the way the gateway's `_boolish` reads it (`true`, the int
 * `1`, or a quoted word); anything unrecognised is not set, so a typo never removes a bot from
 * the mesh. A dependency-free leaf, since the relay publisher runs without the roster's atoms.
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
