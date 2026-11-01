import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { createDesktopProfilePreferences } from './desktop-profile'
import { PrimaryProfilePin } from './primary-profile-pin'
import { connectPrimaryRemote } from './primary-remote-startup'

const remote = {
  baseUrl: 'https://offline.invalid',
  wsUrl: 'wss://offline.invalid/api/ws',
  connectionId: 'remote',
  source: 'settings',
  token: 'placeholder'
}

test('only a successful authoritative roster moves the primary, onto a profile that gateway serves', async ({
  onTestFinished
}) => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-profile-restore-'))
  const preferencePath = path.join(home, 'active-profile.json')
  const preferences = createDesktopProfilePreferences(preferencePath)
  onTestFinished(() => fs.rmSync(home, { recursive: true, force: true }))

  const cases = [
    { name: 'deleted', listing: { profiles: [{ name: 'default' }] }, want: 'default' },
    { name: 'still exists', listing: { profiles: [{ name: 'default' }, { name: 'work' }] }, want: 'work' },
    {
      name: 'connection memory served',
      listing: { profiles: [{ name: 'default' }, { name: 'lab' }] },
      remembered: 'lab',
      want: 'lab'
    },
    {
      name: 'connection memory not served',
      listing: { profiles: [{ name: 'default' }] },
      remembered: 'lab',
      want: 'default'
    },
    { name: 'no default', listing: { profiles: [{ name: 'other' }] }, want: 'other' },
    { name: 'empty', listing: { profiles: [] }, want: 'work' },
    { name: 'malformed', listing: {}, want: 'work' },
    { name: 'partial roster', listing: { profiles: [{ name: 'default' }, {}] }, want: 'work' },
    { name: 'failed read', listing: new Error('offline'), want: 'work' },
    { name: 'explicit route or destination override', allow: false, want: 'work' },
    { name: 'destination override added during probe', destinationChanged: true, want: 'work' },
    { name: 'profile alias', source: 'profile', want: 'work' },
    { name: 'SSH remote profile', ssh: { remoteProfile: 'work' }, want: 'work' }
  ]

  for (const row of cases) {
    preferences.remember('work')
    preferences.rememberConnectionProfiles(row.remembered ? { remote: row.remembered, other: 'work' } : {})
    let allow = row.allow !== false
    const pin = new PrimaryProfilePin()
    pin.pin('work')

    const connection = await connectPrimaryRemote(
      { ...remote, source: row.source || remote.source, ssh: row.ssh },
      {
        profile: 'work',
        restoreRememberedProfile: () => allow,
        connectionProfile: preferences.connectionProfile,
        assertCurrentAttempt: () => undefined,
        advanceBootProgress: async () => undefined,
        waitForRemote: async () => undefined,
        readProfiles: async descriptor => {
          assert.equal(descriptor.baseUrl, 'https://offline.invalid')
          assert.equal(descriptor.connectionId, 'remote')

          if (row.destinationChanged) {allow = false}

          if (row.listing instanceof Error) {throw row.listing}

          return row.listing ?? { profiles: [{ name: 'default' }] }
        },
        onProfileRestored: profile => {
          pin.pin(profile)
        },
        updateBootProgress: () => undefined,
        logs: () => [],
        windowState: () => ({})
      }
    )

    assert.equal(pin.booted, row.want, row.name)
    assert.equal(connection.profile, row.want === 'work' ? undefined : row.want, row.name)
    // The legacy profile is this machine's; a remote that lacks it never rewrites it.
    assert.equal(preferences.readActive(), 'work', row.name)
    assert.equal(connection.baseUrl, 'https://offline.invalid')
    assert.equal(connection.connectionId, 'remote')
  }
})

test('connection memory is pruned with its connection and ignores invalid names', ({ onTestFinished }) => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-profile-memory-'))
  const preferences = createDesktopProfilePreferences(path.join(home, 'active-profile.json'))
  onTestFinished(() => fs.rmSync(home, { recursive: true, force: true }))

  preferences.rememberConnectionProfiles({ lab: 'work', bad: '../etc', other: 'default' })
  assert.equal(preferences.connectionProfile('lab'), 'work')
  assert.equal(preferences.connectionProfile('bad'), null)
  preferences.connectionRemoved('lab')
  assert.equal(preferences.connectionProfile('lab'), null)
  assert.equal(preferences.connectionProfile('other'), 'default')
})

test('a superseded or aborted roster probe cannot publish or mutate the primary profile', async () => {
  for (const cancelledBy of ['attempt', 'signal'] as const) {
    const controller = new AbortController()
    let current = true
    const pin = new PrimaryProfilePin()
    pin.pin('work')
    let ready = false
    await assert.rejects(
      connectPrimaryRemote(remote, {
        profile: 'work',
        restoreRememberedProfile: () => true,
        signal: controller.signal,
        assertCurrentAttempt: () => {
          if (!current) {throw new Error('superseded')}
        },
        advanceBootProgress: async () => undefined,
        waitForRemote: async () => undefined,
        readProfiles: async () => {
          if (cancelledBy === 'signal') {controller.abort(new Error('aborted'))}
          else {current = false}

          return { profiles: [{ name: 'default' }] }
        },
        onProfileRestored: profile => {
          pin.pin(profile)
        },
        updateBootProgress: () => {
          ready = true
        },
        logs: () => [],
        windowState: () => ({})
      }),
      /superseded|aborted/
    )
    assert.equal(pin.booted, 'work')
    assert.equal(ready, false)
  }
})
