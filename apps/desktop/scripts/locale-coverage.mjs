#!/usr/bin/env node
// Reports how much of en.ts each locale actually translates.
//
// `defineLocale()` merges every locale onto the English base, so a locale
// object always has 100% of the keys at runtime and coverage looks perfect
// even when nothing was translated. What matters is how many leaves still
// hold the English value. That is what this reports.
//
// A leaf counts as untranslated when its value is byte-identical to English,
// which also catches words that are simply spelled the same in both languages
// (Repository, System, Email, brand names). Those can never be "translated",
// so read the number as a floor rather than an exact figure. It is still the
// right signal for drift: it only moves when a locale falls behind en.ts.
//
// Keys a locale omits entirely are counted separately as `missing`, because
// treating an absent key as translated would inflate the number. Locales built
// with defineLocale() never have any. zh.ts is a plain object literal instead,
// so it does: those keys fall through to whatever the call site uses as a
// default, which for keybinds.actions is the raw action id.
//
// Run it through tsx, since it imports the TypeScript catalog directly:
//
//   npm run locale:coverage                  # table
//   npm run locale:coverage -- --json        # machine readable
//   npm run locale:coverage -- --list ja     # the untranslated keys for one locale

function leaves(node, prefix = '', out = new Map()) {
  if (node === null || typeof node !== 'object') return out

  for (const [key, value] of Object.entries(node)) {
    const path = `${prefix}${key}`

    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      leaves(value, `${path}.`, out)
    } else {
      out.set(path, typeof value === 'function' ? String(value) : JSON.stringify(value))
    }
  }

  return out
}

// `.href` rather than `.pathname`: pathname yields `/D:/repo/...` on Windows and
// keeps percent-encoding, so a checkout on Windows or a path with spaces fails.
const catalog = await import(new URL('../src/i18n/catalog.ts', import.meta.url).href)
const translations = catalog.TRANSLATIONS
const english = leaves(translations.en)

// Derived from the catalog, so a new locale is measured without touching this file.
const locales = Object.keys(translations).filter(id => id !== 'en')

const report = locales.map(id => {
  const own = leaves(translations[id])
  const missing = []
  const untranslated = []

  for (const [path, value] of english) {
    if (!own.has(path)) {
      missing.push(path)
    } else if (own.get(path) === value) {
      untranslated.push(path)
    }
  }

  const translated = english.size - untranslated.length - missing.length

  return {
    locale: id,
    keys: english.size,
    untranslated: untranslated.length,
    missing: missing.length,
    coverage: Number(((translated / english.size) * 100).toFixed(1)),
    untranslatedPaths: untranslated,
    missingPaths: missing
  }
})

const listIndex = process.argv.indexOf('--list')

if (listIndex !== -1) {
  const wanted = process.argv[listIndex + 1]

  if (!wanted || wanted.startsWith('--')) {
    console.error(`--list needs a locale id, one of: ${locales.join(', ')}`)
    process.exit(1)
  }

  const row = report.find(r => r.locale === wanted)

  if (!row) {
    console.error(`unknown locale: ${wanted}. Known: ${locales.join(', ')}`)
    process.exit(1)
  }

  // stdout stays one key per line so it can be piped; the note goes to stderr.
  if (row.missingPaths.length > 0) {
    console.error(`note: ${row.missingPaths.length} key(s) absent from ${row.locale} entirely:`)
    console.error(row.missingPaths.map(path => `  ${path}`).join('\n'))
  }

  console.log(row.untranslatedPaths.join('\n'))
} else if (process.argv.includes('--json')) {
  const summary = report.map(row => ({
    locale: row.locale,
    keys: row.keys,
    untranslated: row.untranslated,
    missing: row.missing,
    coverage: row.coverage
  }))

  console.log(JSON.stringify(summary, null, 2))
} else {
  // Width from the data, so a longer locale id cannot break the columns.
  const width = Math.max(6, ...locales.map(id => id.length))

  console.log(`en.ts leaf keys: ${english.size}\n`)
  console.log(`${'locale'.padEnd(width)}   untranslated   missing   coverage`)

  for (const row of report) {
    console.log(
      `${row.locale.padEnd(width)}   ${String(row.untranslated).padStart(12)}   ${String(row.missing).padStart(7)}   ${String(row.coverage).padStart(7)}%`
    )
  }
}
