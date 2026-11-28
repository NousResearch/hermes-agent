import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { afterAll, describe, expect, it } from 'vitest'

import { insideHermesProfileHome, sandboxBaseDir } from './sandbox-root'

const made: string[] = []

// NOT os.tmpdir(): this suite itself may run from a Hermes agent session whose TMPDIR is a profile
// home — the very shape under test. The fixtures need a base the helper accepts.
const NEUTRAL = process.platform === 'win32' ? os.tmpdir() : '/tmp'

function tempRoot(label: string): string {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(NEUTRAL, `sandbox-root-${label}-`)))

  made.push(root)

  return root
}

afterAll(() => {
  for (const root of made) {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

describe('insideHermesProfileHome', () => {
  it('spots a sandbox under <hermes home>/profiles/<bot>', () => {
    const root = tempRoot('profile')

    fs.mkdirSync(path.join(root, '.hermes', 'profiles', 'magpie', 'cache', 'hermes-e2e-x'), { recursive: true })
    expect(insideHermesProfileHome(path.join(root, '.hermes', 'profiles', 'magpie', 'cache', 'hermes-e2e-x'))).toBe(true)
    expect(insideHermesProfileHome(path.join(root, '.hermes', 'profiles', 'magpie'))).toBe(true)
  })

  it('accepts a plain temp root', () => {
    const root = tempRoot('plain')

    expect(insideHermesProfileHome(root)).toBe(false)
  })

  it('does not read the default home itself as a sandbox base', () => {
    const root = tempRoot('default-home')

    fs.mkdirSync(path.join(root, '.hermes', 'profiles'), { recursive: true })
    // `~/.hermes` is the default home (its basename ends the walk), not a named profile.
    expect(insideHermesProfileHome(path.join(root, '.hermes'))).toBe(false)
  })

  it('needs a real Hermes home above `profiles`, not just the segment', () => {
    const root = tempRoot('segment')

    fs.mkdirSync(path.join(root, 'work', 'profiles', 'notes'), { recursive: true })
    expect(insideHermesProfileHome(path.join(root, 'work', 'profiles', 'notes'))).toBe(false)
  })
})

describe('sandboxBaseDir', () => {
  it('keeps a safe candidate as is', () => {
    const root = tempRoot('keep')

    expect(sandboxBaseDir(root)).toBe(root)
  })

  it('leaves a profile home for a neutral temp root', () => {
    const root = tempRoot('escape')

    fs.mkdirSync(path.join(root, '.hermes', 'profiles', 'magpie', 'cache'), { recursive: true })
    expect(sandboxBaseDir(path.join(root, '.hermes', 'profiles', 'magpie', 'cache'))).toBe(NEUTRAL)
  })
})
