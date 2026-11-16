import { PassThrough } from 'node:stream'

import { Box, renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { $delegationState, resetDelegationState } from '../app/delegationStore.js'
import { AgentsOverlay } from '../components/agentsOverlay.js'
import type { GatewayClient } from '../gatewayClient.js'
import { DEFAULT_THEME } from '../theme.js'

it('does not undo an acknowledged pause when opening status resolves late', async () => {
  resetDelegationState()
  let resolveStatus!: (value: unknown) => void

  const status = new Promise(resolve => {
    resolveStatus = resolve
  })

  const request = vi.fn(async (method: string) => (method === 'delegation.status' ? status : { paused: true }))
  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 16, isTTY: false })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: () => {}, ref: () => {}, unref: () => {} })

  const view = renderSync(
    <Box height={16}>
      <AgentsOverlay gw={{ request } as unknown as GatewayClient} onClose={() => {}} t={DEFAULT_THEME} />
    </Box>,
    {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    }
  )

  try {
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('delegation.status', {}))
    stdin.write('p')
    await vi.waitFor(() => expect($delegationState.get().paused).toBe(true))
    resolveStatus({ paused: false, max_spawn_depth: 4 })
    await status
    await new Promise(resolve => setTimeout(resolve, 50))
    expect($delegationState.get().paused).toBe(true)
  } finally {
    view.unmount()
    view.cleanup()
    resetDelegationState()
  }
})
