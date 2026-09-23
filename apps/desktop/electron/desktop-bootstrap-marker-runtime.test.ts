import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { createDesktopBootstrapMarkerRuntime } from './desktop-bootstrap-marker-runtime'

test('a completed bootstrap marker records provenance without authorizing an unusable runtime', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-marker-'))

  try {
    const markerPath = path.join(root, 'install', '.hermes-bootstrap-complete')

    const runtime = createDesktopBootstrapMarkerRuntime({
      ACTIVE_HERMES_ROOT: path.dirname(markerPath),
      VENV_ROOT: path.join(root, 'install', 'venv'),
      BOOTSTRAP_COMPLETE_MARKER: markerPath,
      BOOTSTRAP_MARKER_SCHEMA_VERSION: 1,
      app: { getVersion: () => 'test-version' },
      getVenvPython: () => path.join(root, 'missing-python'),
      isHermesSourceRoot: () => false,
      fileExists: () => false,
      writeFileAtomic: (target, data, encoding) => fs.writeFileSync(target, data, encoding)
    })

    const pinnedCommit = '1234567890abcdef1234567890abcdef12345678'

    runtime.writeBootstrapMarker({ pinnedCommit, pinnedBranch: 'main' })

    expect(runtime.readBootstrapMarker()).toMatchObject({ pinnedCommit, pinnedBranch: 'main' })
    expect(await runtime.activeRuntimeState()).toMatchObject({
      hasValidMarker: true,
      shouldUseActiveRuntime: false,
      usabilityReason: 'unusable'
    })
  } finally {
    expect(path.resolve(root).startsWith(path.resolve(os.tmpdir()) + path.sep)).toBe(true)

    fs.rmSync(root, { recursive: true, force: true })
  }
})
