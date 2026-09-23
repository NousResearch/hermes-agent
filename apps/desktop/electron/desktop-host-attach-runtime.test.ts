import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { createDesktopHostAttachRuntime } from './desktop-host-attach-runtime'

test('host spawn reservation is bounded to the selected Hermes home and releases its gate', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-host-attach-'))

  try {
    const runtime = createDesktopHostAttachRuntime({
      HERMES_HOME: root,
      ISOLATED_BACKEND: false,
      rememberLog: () => undefined,
      waitForHermes: async () => undefined,
      invalidatePrimaryConnection: () => undefined,
      scheduleUnexpectedPrimaryRecovery: () => undefined
    })

    const gate = runtime.hostSpawnGateDeps()

    expect(gate.read()).toBeNull()

    const release = gate.take()

    expect(gate.read()).toMatchObject({ ownerAlive: true })

    release()

    expect(gate.read()).toBeNull()
  } finally {
    expect(path.resolve(root).startsWith(path.resolve(os.tmpdir()) + path.sep)).toBe(true)

    fs.rmSync(root, { recursive: true, force: true })
  }
})
