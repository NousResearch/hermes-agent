import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { routedGatewayEndpoint } from './local-gateway'

// B2-17: with a symlinked ~/.hermes (dotfiles link, macOS /var -> /private/var) the raw launch
// HERMES_HOME is not canonical. The sibling profile home is built from the endpoint's own resolved
// profile_id, so mintLocalGatewayTicket's realpath check holds instead of 'Noncanonical gateway profile'.
test.skipIf(process.platform === 'win32')('a sibling route under a symlinked launch home is canonical', () => {
  const base = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-symlink-home-')))
  const real = path.join(base, 'real')
  fs.mkdirSync(path.join(real, 'profiles', 'work'), { recursive: true })
  fs.symlinkSync(real, path.join(base, 'link'))
  const endpoint = {
    profile_id: real,
    instance_id: 'i',
    authority_epoch: 1,
    runtime_protocol: 1,
    api_origin: 'http://127.0.0.1:1',
    capabilities: [],
    supervisor: 'none' as const,
    control_home: null
  }

  try {
    const routed = routedGatewayEndpoint(endpoint as never, 'work', path.join(base, 'link'))
    expect(routed.profile_id).toBe(path.join(real, 'profiles', 'work'))
    expect(fs.realpathSync(routed.profile_id)).toBe(routed.profile_id)
    expect(routedGatewayEndpoint(endpoint as never, 'default', path.join(base, 'link'))).toBe(endpoint)
  } finally {
    fs.rmSync(base, { recursive: true, force: true })
  }
})
