import * as path from 'node:path'

import { describe, expect, it } from 'vitest'

import { electronBinaryName, electronDistCandidates, pathLookupCommand } from './electron-binary'

// Platform is a parameter everywhere below rather than read from
// process.platform, so the Windows rules are pinned on the Linux CI runner too.
// Reading the real platform would leave every Windows-only rule untested.

describe('electronBinaryName', () => {
  it('asks for electron.exe on Windows', () => {
    expect(electronBinaryName('win32')).toBe('electron.exe')
  })

  it('asks for a bare electron everywhere else', () => {
    expect(electronBinaryName('linux')).toBe('electron')
    expect(electronBinaryName('darwin')).toBe('electron')
  })
})

describe('electronDistCandidates', () => {
  const desktop = path.join('repo', 'apps', 'desktop')
  const repo = 'repo'

  it('probes the workspace-local install before the hoisted one', () => {
    // npm only hoists `electron` to the repo root when nothing conflicts, so
    // apps/desktop/node_modules is an ordinary outcome of `npm install`, not a
    // broken tree. Probing only the repo root is what makes the suite refuse to
    // start with "run npm install" on a tree that has electron installed.
    expect(electronDistCandidates([desktop, repo], 'linux')).toEqual([
      path.join(desktop, 'node_modules', 'electron', 'dist', 'electron'),
      path.join(repo, 'node_modules', 'electron', 'dist', 'electron'),
    ])
  })

  it('carries the platform binary name into every candidate', () => {
    // A bare `electron` file never exists in a Windows dist, so a probe built
    // from a hardcoded name cannot match there no matter which root it walks.
    for (const candidate of electronDistCandidates([desktop, repo], 'win32')) {
      expect(path.basename(candidate)).toBe('electron.exe')
    }
  })
})

describe('pathLookupCommand', () => {
  it('uses where on Windows and which elsewhere', () => {
    // `which` is not a command on Windows; spawning it unconditionally made the
    // PATH fallback fail for a reason unrelated to whether electron is on PATH.
    expect(pathLookupCommand('win32')).toBe('where')
    expect(pathLookupCommand('linux')).toBe('which')
    expect(pathLookupCommand('darwin')).toBe('which')
  })
})
