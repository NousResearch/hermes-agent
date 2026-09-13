// windows-file-version.mjs — the Windows VERSIONINFO quad for a release tag.
//
// Windows VERSIONINFO holds four 16-bit fields, so a canary's semver
// string cannot go in it. resedit splits whatever it is handed on "." and
// clamps each token to [0, 65535], so `0.28.0-canary.20260819171926`
// becomes 0.28.0.65535: "0-canary" parses as NaN and falls to the min,
// and the timestamp saturates at the max. Every canary of a minor then
// shows the same meaningless quad on the Details tab.
//
// The timestamp is the only part worth showing there — the semver line is
// identical across a whole canary series — so pack it as
// yyyy.mmdd.hhmm.ss. Every field stays under 65536, and the quad compares
// in timestamp order, which is the order Windows sorts versions in.
//
// Display metadata only: electron-updater keys on the semver version in
// extraMetadata, never on this quad.

/**
 * The `yyyy.mmdd.hhmm.ss` quad for a canary tag, or null for a stable tag
 * (whose version is already four legal fields or fewer, so app-builder-lib's
 * own handling is correct).
 *
 * @param {string} releaseTag A release tag, with the leading `v`.
 * @returns {string | null}
 */
export function windowsFileVersion(releaseTag) {
  const stamp = /-canary\.(20\d{6}(?:\d{6})?)$/.exec(String(releaseTag))?.[1]
  if (!stamp) {
    return null
  }

  // The legacy date-only shape (YYYYMMDD) is midnight of that day.
  const parts = /^(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?$/.exec(stamp)
  if (!parts) {
    return null
  }

  const [, year, month, day, hour, minute, second] = parts
  const field = (a, b) => Number(`${a ?? '00'}${b ?? '00'}`)

  return `${Number(year)}.${field(month, day)}.${field(hour, minute)}.${Number(second ?? 0)}`
}
