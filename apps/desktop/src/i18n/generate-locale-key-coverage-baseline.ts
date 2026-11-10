// One-off maintenance script - NOT part of the test suite (filename does
// not match *.test.ts). Regenerates locale-key-coverage.baseline.json
// after a translation PR closes some missing-key gap, or after en.ts
// grows and a locale needs its baseline raised to match.
//
// Run from apps/desktop/:
//   npx tsx src/i18n/generate-locale-key-coverage-baseline.mts
//
// Then review the diff - a shrinking baseline is a translation
// improvement (good), a growing one should correspond to genuinely new
// en.ts keys the PR is also adding UI for.
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'

import { missingKeyPaths } from './locale-key-coverage.util'

const LOCALES = ['ar', 'ru', 'ja', 'zh', 'zh-hant']
const OUT_PATH = join(import.meta.dirname, 'locale-key-coverage.baseline.json')

const baseline: Record<string, string[]> = {}

for (const locale of LOCALES) {
  baseline[locale] = missingKeyPaths(locale)
}

writeFileSync(OUT_PATH, `${JSON.stringify(baseline, null, 2)}\n`)

for (const [locale, missing] of Object.entries(baseline)) {
  console.log(`${locale}: ${missing.length} missing`)
}
