'use strict'

// Reads user-authored desktop locale overrides from disk, OUTSIDE the app
// bundle, so they survive updates that replace the compiled catalog.
//
//   <hermes-home>/locale-overrides/desktop/<lang>.json
//
// The renderer can't touch the filesystem, so the main process reads the file
// and hands the parsed JSON to the renderer, which merges it (type-safely) on
// top of the bundled translations. Anything missing or malformed yields null —
// a bad override file must never break the UI; it just means no override.

const fs = require('node:fs')
const path = require('node:path')

const OVERRIDES_SUBDIR = path.join('locale-overrides', 'desktop')

// Locale codes are short ASCII tokens (e.g. "ja", "zh-hant"). Constraining the
// charset keeps the value from escaping the overrides directory via traversal
// or absolute-path tricks before it ever reaches path.join.
const LANG_RE = /^[A-Za-z]{2,8}(-[A-Za-z]{2,8})*$/

function localeOverridesFile(hermesHome, lang) {
  if (typeof hermesHome !== 'string' || hermesHome.trim() === '') return null
  if (typeof lang !== 'string' || !LANG_RE.test(lang)) return null

  const baseDir = path.resolve(hermesHome, OVERRIDES_SUBDIR)
  const file = path.resolve(baseDir, `${lang}.json`)

  // Belt-and-suspenders: the resolved file must stay directly under baseDir.
  if (path.dirname(file) !== baseDir) return null

  return file
}

/**
 * Resolve and read the override file for `lang`. Returns the parsed plain
 * object, or null when the file is absent, unreadable, not valid JSON, or not a
 * JSON object. Never throws.
 */
function readLocaleOverridesForIpc(hermesHome, lang) {
  const file = localeOverridesFile(hermesHome, lang)
  if (!file) return null

  let raw
  try {
    raw = fs.readFileSync(file, 'utf8')
  } catch {
    // Missing file is the common case (no override authored) — not an error.
    return null
  }

  let parsed
  try {
    parsed = JSON.parse(raw)
  } catch {
    return null
  }

  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return null
  }

  return parsed
}

module.exports = { readLocaleOverridesForIpc, localeOverridesFile }
