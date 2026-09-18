import assert from 'node:assert/strict'

import { test } from 'vitest'

import { fetchRosterSourceData, rosterProfileMetadata } from './roster-source-fetch'

test('roster profile metadata maps the explicit REST identity fields', () => {
  assert.deepEqual(
    rosterProfileMetadata({
      bot_title: '  Build Bot  ',
      display_name: '  Builder  ',
      has_avatar: true,
      title: 'Legacy title',
      ui_meta: { 'hermes-bots': { color: '#abcdef' } }
    }),
    {
      display_name: 'Builder',
      has_avatar: true,
      title: 'Build Bot',
      ui_meta: { 'hermes-bots': { color: '#abcdef' } }
    }
  )
})

test('roster profile metadata keeps legacy title compatibility and rejects malformed fields', () => {
  assert.deepEqual(
    rosterProfileMetadata({
      bot_title: ' ',
      display_name: 7,
      has_avatar: 'yes',
      title: '  Legacy title  ',
      ui_meta: []
    }),
    { title: 'Legacy title' }
  )
  assert.deepEqual(rosterProfileMetadata(null), {})
})

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
