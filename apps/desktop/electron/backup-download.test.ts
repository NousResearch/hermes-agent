/**
 * Tests for electron/backup-download.ts.
 *
 * Run with: node --test electron/backup-download.test.ts
 *
 * backupDownloadPath is the routing seam behind the "download a backup
 * archive" IPC bridge — it must scope the request to the requesting profile
 * the same way the generic api() JSON bridge does (pathWithGlobalRemoteProfile),
 * or a shared global-remote dashboard would let one profile download another's
 * backup archive.
 */
import assert from 'node:assert/strict'

import { test } from 'vitest'

import { backupDownloadPath } from './backup-download'

test('backupDownloadPath appends profile in global remote mode', () => {
  assert.equal(
    backupDownloadPath('/home/worker/backups/hermes-backup-worker.zip', 'worker', {
      globalRemote: true,
      profileRemoteOverride: false
    }),
    '/api/ops/backup/download?archive=%2Fhome%2Fworker%2Fbackups%2Fhermes-backup-worker.zip&profile=worker'
  )
})

test('backupDownloadPath skips the primary profile, which the remote already serves', () => {
  assert.equal(
    backupDownloadPath('/home/coder/backups/hermes-backup.zip', 'coder', {
      globalRemote: true,
      primaryProfile: 'coder',
      profileRemoteOverride: false
    }),
    '/api/ops/backup/download?archive=%2Fhome%2Fcoder%2Fbackups%2Fhermes-backup.zip'
  )
})

test('backupDownloadPath skips local and per-profile remote override paths', () => {
  assert.equal(
    backupDownloadPath('/home/worker/backups/x.zip', 'worker', {
      globalRemote: false,
      profileRemoteOverride: false
    }),
    '/api/ops/backup/download?archive=%2Fhome%2Fworker%2Fbackups%2Fx.zip'
  )
  assert.equal(
    backupDownloadPath('/home/worker/backups/x.zip', 'worker', {
      globalRemote: true,
      profileRemoteOverride: true
    }),
    '/api/ops/backup/download?archive=%2Fhome%2Fworker%2Fbackups%2Fx.zip'
  )
})

test('backupDownloadPath URL-encodes the archive path', () => {
  assert.equal(
    backupDownloadPath('/home/my profile/backups/a b.zip', null, {}),
    '/api/ops/backup/download?archive=%2Fhome%2Fmy+profile%2Fbackups%2Fa+b.zip'
  )
})
