import assert from 'node:assert/strict'

import { test } from 'vitest'

import { fetchRosterSourceData } from './roster-source-fetch'

test('roster source starts profile and install-id reads together and preserves both results', async () => {
  let releaseProfiles!: (value: { profiles: Array<{ name: string }> }) => void
  let releaseInstallId!: (value: string | undefined) => void

  const profiles = new Promise<{ profiles: Array<{ name: string }> }>(resolve => {
    releaseProfiles = resolve
  })

  const installId = new Promise<string | undefined>(resolve => {
    releaseInstallId = resolve
  })

  const started: string[] = []

  const pending = fetchRosterSourceData(
    () => {
      started.push('profiles')

      return profiles
    },
    () => {
      started.push('install-id')

      return installId
    }
  )

  assert.deepEqual(started, ['profiles', 'install-id'])
  releaseInstallId('install-1')
  releaseProfiles({ profiles: [{ name: 'default' }] })
  assert.deepEqual(await pending, {
    body: { profiles: [{ name: 'default' }] },
    installId: 'install-1'
  })
})

test('roster source preserves a profile-read failure after starting the install-id read', async () => {
  const profilesError = new Error('profiles unavailable')
  let installIdStarted = false

  await assert.rejects(
    fetchRosterSourceData(
      async () => {
        throw profilesError
      },
      async () => {
        installIdStarted = true

        return undefined
      }
    ),
    profilesError
  )
  assert.equal(installIdStarted, true)
})
