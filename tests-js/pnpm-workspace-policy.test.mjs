import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { load } from 'js-yaml'
import { expect, test } from 'vitest'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const read = name => load(readFileSync(resolve(repo, name), 'utf8'))
const workspace = read('pnpm-workspace.yaml')
const excludes = workspace.minimumReleaseAgeExclude ?? []

// A bare name or a `@scope/*` glob lifts the release-age gate for every future version of the
// package, and a prune never drops a glob: an exemption names the exact
// versions (`name@1.2.3`, or `name@1.2.3 || 1.2.4`) the lockfile needs.
const semver = String.raw`\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?`
const exact = new RegExp(String.raw`^((?:@[^/@*\s]+/)?[^/@*\s]+)@(${semver}(?:\s*\|\|\s*${semver})*)$`)

test('every release-age exclude names exact versions', () => {
  expect(excludes.filter(entry => !exact.test(entry))).toEqual([])
})

// `hermes pm update --pnpm` drops an exclude once the lock no longer holds its version; one that
// lands any other way exempts nothing and stays until the next refresh. CI runs this after its
// install, which leaves the file as committed (js-tests.yml checks that).
test('every release-age exclude is a version the lockfile holds', () => {
  const locked = new Set(Object.keys(read('pnpm-lock.yaml').packages))
  const stale = excludes.flatMap(entry => {
    const [, name, versions] = exact.exec(entry) ?? []
    return name ? versions.split('||').map(v => `${name}@${v.trim()}`).filter(spec => !locked.has(spec)) : []
  })
  expect(stale).toEqual([])
})

// pnpm's prune runs on every install that settles the lockfile, frozen and --trust-lockfile
// included: with it on, a stale entry makes every user build (scripts/build/node-deps.mjs)
// rewrite this tracked file, and CI's install hides the entry from the test above.
test('installs never rewrite pnpm-workspace.yaml to prune excludes', () => {
  expect(workspace.minimumReleaseAgeExcludePrune ?? false).toBe(false)
})
