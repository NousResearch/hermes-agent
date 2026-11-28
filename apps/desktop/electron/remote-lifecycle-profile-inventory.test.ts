import assert from 'node:assert/strict'
import { exec as execCallback } from 'node:child_process'
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

import { listRemoteHermesProfiles } from './remote-lifecycle'

const exec = promisify(execCallback)

// The home probe answers with `home`; the profile listing runs for real under /bin/sh.
function shellSsh(home: string) {
  const calls: string[] = []

  return {
    calls,
    async exec(cmd: string) {
      calls.push(cmd)

      return cmd.includes('/profiles') ? (await exec(cmd, { shell: '/bin/sh' })).stdout : `${home}\n`
    }
  }
}

async function profileDir(profiles: string, name: string, marker?: string) {
  await mkdir(path.join(profiles, name), { recursive: true })

  if (marker) {
    await writeFile(path.join(profiles, name, marker), '')
  }
}

test('listRemoteHermesProfiles lists only profiles the CLI would start', async () => {
  const home = await mkdtemp(path.join(os.tmpdir(), 'hermes-inventory-'))
  const profiles = path.join(home, 'profiles')

  try {
    await profileDir(profiles, 'bob', 'config.yaml')
    await profileDir(profiles, 'dixie', '.env')
    await profileDir(profiles, 'goose', 'state.db')
    await profileDir(profiles, 'ops-2', 'SOUL.md')
    await profileDir(profiles, 'bob.rollback-old', 'config.yaml')
    // A leftover with no identity marker: before this filter every such dir got its own backend.
    await profileDir(profiles, 'audit-148296dba80f4d5b8cf79138a84f145b')
    // `hermes profile delete` tombstone beside a dir that still has a marker.
    await profileDir(profiles, 'gone', 'config.yaml')
    await mkdir(path.join(profiles, '.deleted'))
    await writeFile(path.join(profiles, '.deleted', 'gone'), 'deleted')
    // A dangling symlinked marker counts, as in hermes_constants.named_profile_is_live.
    await profileDir(profiles, 'linked')
    await symlink('/nonexistent-auth', path.join(profiles, 'linked', 'auth.json'))
    await symlink('/nonexistent-target', path.join(profiles, 'dangling-dir'))
    await writeFile(path.join(profiles, 'stray-file'), '')

    const ssh = shellSsh(home)

    assert.deepEqual((await listRemoteHermesProfiles(ssh)).sort(), [
      'bob',
      'default',
      'dixie',
      'goose',
      'linked',
      'ops-2'
    ])
    assert.equal(
      ssh.calls.some(cmd => cmd.includes('serve') || cmd.includes('dashboard')),
      false
    )
  } finally {
    await rm(home, { recursive: true, force: true })
  }
})

test('listRemoteHermesProfiles falls back to the implicit default without a profiles dir', async () => {
  const home = await mkdtemp(path.join(os.tmpdir(), 'hermes-inventory-'))

  try {
    assert.deepEqual(await listRemoteHermesProfiles(shellSsh(home)), ['default'])
  } finally {
    await rm(home, { recursive: true, force: true })
  }
})

test('listRemoteHermesProfiles rejects a hostile HERMES_HOME before listing', async () => {
  const ssh = shellSsh('/tmp/x; echo pwned')

  await assert.rejects(
    () => listRemoteHermesProfiles(ssh),
    (err: unknown) => {
      assert.ok(err && typeof err === 'object' && 'kind' in err)
      assert.equal(err.kind, 'unsafe-path')

      return true
    }
  )
  assert.equal(
    ssh.calls.some(cmd => cmd.includes('/profiles')),
    false
  )
})
