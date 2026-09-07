'use strict'

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, it } from 'vitest'

import {
  buildRestartManagerScript,
  listRestartManagerHoldersForResources,
  parseRestartManagerOutput,
  RESTART_MANAGER_PER_FILE_LIMIT
} from './windows-restart-manager'

describe('windows restart manager listing', () => {
  // Asserts Windows path rendering inside the generated script; the module is Windows-only.
  it.skipIf(process.platform !== 'win32')('hands resources to PowerShell through a flagged list file, never inline, and removes the file afterwards', async () => {
    const definite = Array.from({ length: 200 }, (_, index) => `C:\\install\\venv\\Scripts\\tool${index}'s.exe`)
    const shared = Array.from({ length: 100 }, (_, index) => `C:\\install\\venv\\Lib\\site-packages\\pkg${index}\\mod.pyd`)
    let capturedScript = ''
    let listContents = ''
    let listPathSeen = ''

    const holders = await listRestartManagerHoldersForResources(definite, {
      platform: 'win32',
      listDir: os.tmpdir(),
      shared,
      attributionRoot: 'C:\\install',
      run: async script => {
        capturedScript = script
        const match = script.match(/ReadAllLines\('([^']+)'/)
        listPathSeen = match?.[1] ?? ''
        listContents = fs.readFileSync(listPathSeen, 'utf8')

        return {
          stdout: JSON.stringify([{ pid: 4242, createdAt: 1700000000, name: 'Python', resource: definite[0] }]),
          stderr: '',
          code: 0
        }
      }
    })

    const lines = listContents.trim().split('\n')

    assert.ok(listPathSeen.startsWith(os.tmpdir()), 'list file lives in the temp dir')
    assert.equal(lines.length, 300)
    assert.equal(lines.filter(line => line.startsWith('D\t')).length, 200)
    assert.equal(lines.filter(line => line.startsWith('A\t')).length, 100)
    assert.ok(!capturedScript.includes('pkg17\\'), 'resource paths are not inlined into the command')
    assert.ok(capturedScript.length < 16_000, 'script stays well under the Windows command-line limit')
    assert.match(capturedScript, /\$attributionRootClaim = 'C:\\install'/)
    assert.equal(fs.existsSync(listPathSeen), false, 'list file is removed after the run')
    assert.deepEqual(holders, [
      {
        pid: 4242,
        createdAt: 1700000000,
        name: 'Python',
        cmdline: 'Python',
        source: 'restart-manager',
        resource: definite[0],
        role: 'other'
      }
    ])
  })

  it('queries small definite lists per file, large ones batched, and attributes ambiguous holders by mapped module', () => {
    const script = buildRestartManagerScript(path.join(os.tmpdir(), 'list.txt'), { attributionRoot: 'C:\\install' })

    assert.match(script, new RegExp(`\\$definite\\.Count -le ${RESTART_MANAGER_PER_FILE_LIMIT}`))
    assert.match(script, /\$batches \+= ,@\(\[string\[\]\]\$definite\)/)
    assert.match(script, /RmRegisterResources\(handle, \(uint\)files\.Length, files/)
    assert.match(script, /\[HermesRm\]::Query\(\[string\[\]\]\$ambiguous/)
    assert.match(script, /\$proc\.Modules/)
    assert.match(script, /\[HermesRm\]::IsSameOrUnderRoot\(\$moduleFinal, \$root\)/)
    // A unary-comma return hands the whole row array to the attribution loop
    // as one "row", whose .pid is then an array and every holder is dropped.
    assert.doesNotMatch(script, /return ,\$rows/)
    assert.match(script, /foreach \(\$row in @\(Convert-RmRows \$raw \(\[string\]\$ambiguous\[0\]\)\)\)/)
  })

  it('never attributes ambiguous files without an attribution root', async () => {
    let listContents = ''

    await listRestartManagerHoldersForResources([], {
      platform: 'win32',
      shared: ['C:\\install\\venv\\Lib\\site-packages\\x.pyd'],
      run: async script => {
        listContents = fs.readFileSync(script.match(/ReadAllLines\('([^']+)'/)?.[1] ?? '', 'utf8')

        return { stdout: '[]', stderr: '', code: 0 }
      }
    })

    assert.equal(listContents, '', 'nothing to query: shared files without a root are skipped entirely')
  })

  it('never loads a cached assembly from a predictable path', () => {
    const script = buildRestartManagerScript(path.join(os.tmpdir(), 'list.txt'), { attributionRoot: 'C:\\install' })

    // The shim is compiled every run. A deterministic
    // `%TEMP%\hermes-restart-manager\HermesRm-<hash>.dll` loaded with
    // `Add-Type -Path` was unverified executable code at a path any same-user
    // process could pre-create, and a hash recorded beside it would be written
    // by that same principal. The live suite proves the compile path works.
    assert.doesNotMatch(script, /Add-Type -Path/)
    assert.doesNotMatch(script, /HermesRm-[0-9a-f]+\.dll/)
    assert.doesNotMatch(script, /-OutputAssembly/)
    assert.match(script, /^Add-Type -TypeDefinition \$rmSource$/m)
  })

  it('keeps an ambiguous holder that could not be attributed', () => {
    const script = buildRestartManagerScript(path.join(os.tmpdir(), 'list.txt'), { attributionRoot: 'C:\\install' })

    // Classification behaviour is proved live against real processes
    // (windows-restart-manager.windows-live.test.ts). This pins only the
    // wiring that is invisible from outside the script: 'foreign' is the ONLY
    // outcome that drops a row.
    assert.match(script, /if \(\$resolved\.status -eq 'foreign'\) \{ continue \}/)
    assert.match(script, /\$row\.attribution = 'unattributed'/)
  })

  it('reports an unattributed holder without an ownership proof', () => {
    const holders = parseRestartManagerOutput(
      JSON.stringify([
        { pid: 20, createdAt: 7, name: 'python', resource: 'C:\\a.pyd', attribution: 'unattributed' },
        { pid: 21, createdAt: 8, name: 'python', resource: 'C:\\b.pyd', attribution: 'proven' }
      ]),
      ['C:\\fallback.pyd']
    )

    // `resource` is the claim the terminate boundary re-proves before it kills
    // anything. An unattributed holder blocks the update but authorizes nothing.
    assert.equal(holders[0]?.resource, undefined)
    assert.deepEqual(holders[0]?.resources, ['C:\\a.pyd'])
    assert.equal(holders[1]?.resource, 'C:\\b.pyd')
  })

  it('collapses duplicate holders reported across per-file sessions and drops malformed rows', () => {
    const rows = [
      { pid: 10, createdAt: 5, name: 'Python', resource: 'C:\\a.pyd' },
      { pid: 10, createdAt: 5, name: 'Python', resource: 'C:\\b.pyd' },
      { pid: 0, createdAt: 5, name: 'bad' },
      { pid: 11, createdAt: 0, name: 'bad' },
      { pid: 12, createdAt: 9 }
    ]

    const holders = parseRestartManagerOutput(JSON.stringify(rows), ['C:\\fallback.pyd'])

    assert.deepEqual(
      holders.map(holder => [holder.pid, holder.resource, holder.name]),
      [
        [10, 'C:\\a.pyd', 'Python'],
        [12, 'C:\\fallback.pyd', 'unknown']
      ]
    )
  })

  it('is a no-op off Windows and for empty lists', async () => {
    assert.deepEqual(await listRestartManagerHoldersForResources(['C:\\x.pyd'], { platform: 'linux' }), [])
    assert.deepEqual(await listRestartManagerHoldersForResources([], { platform: 'win32' }), [])
  })
})
