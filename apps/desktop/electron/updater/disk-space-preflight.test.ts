import assert from 'node:assert/strict'

import { test } from 'vitest'

import { preflightDiskSpace, UPDATE_SPACE_NEEDED_BYTES } from './disk-space-preflight'

test('passes when the install root has enough free space', async (): Promise<void> => {
  const logs: string[] = []
  let asked: string | null = null

  await preflightDiskSpace({
    root: '/tmp/hermes-install',
    log: (message: string): void => {
      logs.push(message)
    },
    freeBytes: async (root: string): Promise<number> => {
      asked = root
      return UPDATE_SPACE_NEEDED_BYTES + 1
    }
  })

  assert.equal(asked, '/tmp/hermes-install')
  assert.equal(logs.length, 1)
  assert.match(logs[0], /disk space pre-flight/)
})

test('fails closed with an actionable message when the disk is full', async (): Promise<void> => {
  const logs: string[] = []

  await assert.rejects(
    preflightDiskSpace({
      root: '/tmp/hermes-install',
      log: (message: string): void => {
        logs.push(message)
      },
      freeBytes: async (): Promise<number> => 512 * 1024 * 1024,
      neededBytes: 2 * 1024 * 1024 * 1024
    }),
    (error: unknown): boolean => {
      assert.ok(error instanceof Error)
      assert.match(error.message, /Brakuje miejsca na dysku/)
      assert.match(error.message, /2\.0 GiB/)
      assert.match(error.message, /0\.5 GiB/)
      assert.match(error.message, /instalacja nie została zmieniona/)
      assert.match(error.message, /Spróbuj ponownie/)
      return true
    }
  )

  assert.equal(logs.length, 1)
  assert.match(logs[0], /Brakuje miejsca/)
})

test('fails closed when free space cannot be read, without claiming the cause', async (): Promise<void> => {
  await assert.rejects(
    preflightDiskSpace({
      root: '/tmp/hermes-install',
      log: (): void => undefined,
      freeBytes: async (): Promise<number> => {
        throw new Error('statfs EACCES')
      }
    }),
    (error: unknown): boolean => {
      assert.ok(error instanceof Error)
      assert.match(error.message, /Nie udało się sprawdzić wolnej przestrzeni/)
      assert.match(error.message, /zatrzymana, zanim cokolwiek zmieniono/)
      assert.match(error.message, /statfs EACCES/)
      return true
    }
  )
})
