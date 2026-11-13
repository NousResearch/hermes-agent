#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const source = new URL('../locales/registry.json', import.meta.url)
const target = new URL('../apps/shared/src/locale-registry.generated.ts', import.meta.url)
const registry = JSON.parse(readFileSync(source, 'utf8'))
const output = '// Generated from locales/registry.json by node scripts/gen_locale_registry.mjs. Do not edit.\n'
  + `const registry = ${JSON.stringify(registry, null, 2)} as const\n\nexport default registry\n`

if (process.argv.includes('--check')) {
  if (readFileSync(target, 'utf8') !== output) {
    console.error('Locale registry is stale; run node scripts/gen_locale_registry.mjs')
    process.exitCode = 1
  }
} else {
  writeFileSync(target, output)
  console.log(fileURLToPath(target))
}
