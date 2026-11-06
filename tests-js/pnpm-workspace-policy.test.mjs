import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { load } from 'js-yaml'
import { expect, test } from 'vitest'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const read = name => load(readFileSync(resolve(repo, name), 'utf8'))
const excludes = read('pnpm-workspace.yaml').minimumReleaseAgeExclude ?? []

// A bare name or a `@scope/*` glob lifts the release-age gate for every future version of the
// package, and minimumReleaseAgeExcludePrune never drops a glob: an exemption names the exact
// versions (`name@1.2.3`, or `name@1.2.3 || 1.2.4`) the lockfile needs.
const semver = String.raw`\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?`
const exact = new RegExp(String.raw`^((?:@[^/@*\s]+/)?[^/@*\s]+)@(${semver}(?:\s*\|\|\s*${semver})*)$`)

test('every release-age exclude names exact versions', () => {
  expect(excludes.filter(entry => !exact.test(entry))).toEqual([])
})

// Every install prunes an exclude whose version the lockfile no longer holds, user installs
// included: a stale committed entry would rewrite pnpm-workspace.yaml in a user's checkout.
test('every release-age exclude is a version the lockfile holds', () => {
  const locked = new Set(Object.keys(read('pnpm-lock.yaml').packages))
  const stale = excludes.flatMap(entry => {
    const [, name, versions] = exact.exec(entry) ?? []
    return name ? versions.split('||').map(v => `${name}@${v.trim()}`).filter(spec => !locked.has(spec)) : []
  })
  expect(stale).toEqual([])
})
