import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

// Static-analysis guard: every pet.* RPC sent from the repo's TypeScript goes
// through the desktop owner wrapper, except the callers named below.
//
// This is a source-text scan, not a behavior test — same category as an ESLint
// rule, expressed as a vitest so it runs with the rest of the suite (precedent:
// components/ui/__tests__/no-native-title.test.ts). The behavior half lives in
// pet-gallery-owner-switch.test.ts: every gallery action stamps the visible
// owner's profile, which is what makes a call through the wrapper correct.
//
// A pet.* string literal is WRAPPED when it is the method argument of one of
// WRAPPERS. Any other pet.* literal (an unscoped request, an event name) must be
// listed for its exact file AND method in ALLOWED, with the reason. So a new
// unscoped caller fails, a new unscoped method inside an allowlisted file
// fails, and a stale entry fails too: dropping an exception is a one-line edit.

const WRAPPERS = ['requestPet', 'petRpc', 'requestPetForFormerOwner']

const GENERATE_CARD =
  'Generate flow stays on the launch profile: the backend registers pet.generate and pet.hatch ' +
  'scoped=False (tui_gateway/methods_session.py:1732, :1789) and ignores the profile param. Own card.'

const ALLOWED: Record<string, { methods: string[]; reason: string }> = {
  'apps/desktop/src/store/pet-generate.ts': {
    methods: [
      'pet.cancel',
      'pet.generate',
      'pet.generate.progress',
      'pet.generate.status',
      'pet.hatch',
      'pet.hatch.progress',
      'pet.remove',
      'pet.rename',
      'pet.select'
    ],
    reason: GENERATE_CARD
  },
  'apps/desktop/src/plugins/hermes-bots/pet.tsx': {
    methods: ['pet.gallery', 'pet.thumb'],
    reason:
      'Bots avatar picker lists the launch profile’s pets. Whose pets it should list is a GM product call. Own card.'
  },
  'apps/desktop/src/app/session/hooks/use-message-stream/gateway-event/lifecycle.ts': {
    methods: ['pet.changed'],
    reason: 'Event name on an incoming gateway event, not an RPC.'
  },
  'ui-tui/src/components/petPicker.tsx': {
    methods: ['pet.gallery', 'pet.select'],
    reason: 'Terminal UI: one profile per process, so the desktop owner claim does not apply.'
  },
  'ui-tui/src/lib/petPolling.ts': {
    methods: ['pet.cells', 'pet.info.meta'],
    reason: 'Terminal UI: one profile per process, so the desktop owner claim does not apply.'
  }
}

// Declares every gateway method; sends none.
const SKIPPED_FILES = new Set(['apps/shared/src/gateway-contract.generated.ts'])

const SKIPPED_DIRS = new Set(['node_modules', 'dist', 'build', 'out', 'release', 'coverage', '__tests__', 'e2e'])

const REPO_ROOT = resolve(__dirname, '../../../..')

function collectSources(dir: string): string[] {
  const results: string[] = []

  for (const entry of readdirSync(dir)) {
    if (SKIPPED_DIRS.has(entry) || entry.startsWith('.')) {
      continue
    }

    const fullPath = join(dir, entry)

    if (statSync(fullPath).isDirectory()) {
      results.push(...collectSources(fullPath))
    } else if (/\.(c|m)?tsx?$/.test(entry) && !/\.(test|spec)\.tsx?$/.test(entry) && !entry.endsWith('.d.ts')) {
      results.push(fullPath)
    }
  }

  return results
}

/** Blank out comments so prose mentions of `pet.info` never count. */
const stripComments = (source: string) =>
  source.replace(/\/\*[\s\S]*?\*\//g, block => block.replace(/[^\n]/g, ' ')).replace(/(^|[^:])\/\/.*$/gm, '$1')

const WRAPPED_BEFORE = new RegExp(`\\b(${WRAPPERS.join('|')})\\s*(<[^()]*?>)?\\s*\\(\\s*[\\w.]+\\s*,\\s*$`)

/** Every pet.* literal in `source` that is not a wrapper's method argument. */
function unwrappedPetMethods(source: string): string[] {
  const code = stripComments(source)
  const found = new Set<string>()

  for (const match of code.matchAll(/(['"`])(pet\.[a-z_.]+)\1/g)) {
    const before = code.slice(Math.max(0, match.index - 200), match.index)

    if (!WRAPPED_BEFORE.test(before)) {
      found.add(match[2])
    }
  }

  return [...found].sort()
}

// Every top-level dir holding tracked TypeScript. A missing one throws: the
// scan must never quietly shrink.
const SCAN_ROOTS = ['apps', 'evals', 'optional-skills', 'tests', 'tests-js', 'ui-tui', 'web', 'website']

function scanRepo(): Record<string, string[]> {
  const result: Record<string, string[]> = {}

  for (const file of SCAN_ROOTS.flatMap(dir => collectSources(join(REPO_ROOT, dir)))) {
    const rel = relative(REPO_ROOT, file).replace(/\\/g, '/')
    const methods = SKIPPED_FILES.has(rel) ? [] : unwrappedPetMethods(readFileSync(file, 'utf8'))

    if (methods.length) {
      result[rel] = methods
    }
  }

  return result
}

describe('pet RPC owner wrapper', () => {
  it('classifies wrapped calls, unscoped calls and comments', () => {
    expect(unwrappedPetMethods("await petRpc<{ ok: boolean; x?: string }>(request, 'pet.thumb', {})")).toEqual([])
    expect(unwrappedPetMethods("requestPet<PetInfo>(\n  requestGateway,\n  'pet.info',\n  {}\n)")).toEqual([])
    expect(unwrappedPetMethods("host.request('pet.gallery', {})")).toEqual(['pet.gallery'])
    expect(unwrappedPetMethods("// sends `pet.info` via the wrapper\n/* 'pet.scale' */")).toEqual([])
  })

  it('has no unscoped pet.* caller outside the named exceptions', () => {
    const found = scanRepo()
    const unexpected: string[] = []

    for (const [file, methods] of Object.entries(found)) {
      const allowed = ALLOWED[file]?.methods ?? []

      for (const method of methods) {
        if (!allowed.includes(method)) {
          unexpected.push(`${file}: ${method}`)
        }
      }
    }

    expect(unexpected, 'route these through requestPet (store/pet.ts) or add a reasoned ALLOWED entry').toEqual([])
  })

  it('has no stale exception', () => {
    const found = scanRepo()
    const stale: string[] = []

    for (const [file, { methods }] of Object.entries(ALLOWED)) {
      for (const method of methods) {
        if (!found[file]?.includes(method)) {
          stale.push(`${file}: ${method}`)
        }
      }
    }

    expect(stale, 'remove exceptions the code no longer needs').toEqual([])
  })
})
