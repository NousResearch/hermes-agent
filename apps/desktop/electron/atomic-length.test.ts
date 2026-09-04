import { assert, test } from 'vitest'

import { atomicWindowsSpawnCommand } from './windows-remote-lifecycle'

test('atomic spawn command stays under the cmd.exe length budget', () => {
  const command = atomicWindowsSpawnCommand(
    {
      hermesHome: 'C:\\Users\\TestUser\\AppData\\Local\\Temp\\ssh-probe-home',
      python: 'C:\\Users\\TestUser\\AppData\\Local\\hermes\\hermes-agent\\venv\\Scripts\\python.exe'
    },
    {
      ownershipId: '0123456789abcdef0123456789abcdef',
      spawnNonce: '0123456789abcdef',
      profile: 'default',
      hermesPath: 'C:\\Users\\TestUser\\AppData\\Local\\Temp\\ssh-probe-home\\fake-hermes.exe',
      hermesHome: 'C:\\Users\\TestUser\\AppData\\Local\\Temp\\ssh-probe-home',
      tokenFingerprint: 'a'.repeat(32),
      startedAt: '2026-08-27T14:00:00.000Z'
    }
  )

  // cmd.exe rejects commands at 8191 chars — the atomic spawn command carries
  // the full lock record and helper paths, so it is the tightest budget.
  assert.ok(command.length > 0)
  assert.ok(command.length < 8191, `atomic spawn command is too long: ${command.length}`)
})
