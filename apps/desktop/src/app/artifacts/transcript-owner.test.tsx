import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createServer, type Server } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'
import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import { ArtifactsView } from './index'

const list = vi.hoisted(() => vi.fn())
vi.mock('@/hermes', async importOriginal => ({ ...(await importOriginal()), listAllProfileSessions: list }))

interface RoutingHelpers {
  apiRequestRegistryConnectionId(request: HermesApiRequest): null | string
  pathForRegistryBackendRequest(
    path: string,
    profile: null | string | undefined,
    backend: { mode: string; sharedRemote: boolean }
  ): string
}

it('loads owned artifacts across A → B → A and local scopes through real HTTP reads', async () => {
  // Main-process helpers have their own compiler settings; load that side of
  // the adapter at runtime without pulling it into the renderer type program.
  const routingPath = '../../../electron/connection-config.ts'

  const { apiRequestRegistryConnectionId, pathForRegistryBackendRequest } = (await import(
    /* @vite-ignore */ routingPath
  )) as RoutingHelpers

  const root = mkdtempSync(join(tmpdir(), 'hermes-artifacts-owner-'))

  const cases = [
    { connection_id: 'remote-a', profile: 'writer', id: 'session-a', file: 'report-a.txt' },
    { connection_id: 'remote-b', profile: 'writer', id: 'session-b', file: 'report-b.txt' },
    { connection_id: 'remote-a', profile: 'analyst', id: 'session-a-next', file: 'report-a-next.txt' },
    { profile: 'writer', id: 'session-local', file: 'local-report.txt' }
  ]

  for (const fixture of cases) {
    const home = join(root, fixture.connection_id ?? 'local', fixture.profile)
    mkdirSync(home, { recursive: true })
    writeFileSync(
      join(home, `${fixture.id}.json`),
      JSON.stringify({
        session_id: fixture.id,
        messages: [{ role: 'assistant', timestamp: 1000, content: `MEDIA:/srv/${fixture.file}` }]
      })
    )
  }

  const servers: Server[] = []
  const bases = new Map<string, string>()
  const receipts: { connectionId: string; profile: string; status: number }[] = []

  try {
    for (const connectionId of ['remote-a', 'remote-b', 'local']) {
      const server = createServer((request, response) => {
        const url = new URL(request.url!, 'http://localhost')
        const profile = url.searchParams.get('profile') ?? 'default'
        const id = url.pathname.match(/^\/api\/sessions\/([^/]+)\/messages$/)?.[1]
        response.setHeader('Content-Type', 'application/json')

        try {
          const data = readFileSync(join(root, connectionId, profile, `${id}.json`))
          response.end(data)
        } catch {
          response.statusCode = 404
          response.end(JSON.stringify({ detail: 'Session not found' }))
        }
      })

      servers.push(server)
      await new Promise<void>((resolve, reject) => {
        server.once('error', reject)
        server.listen(0, '127.0.0.1', resolve)
      })
      const address = server.address()

      if (!address || typeof address === 'string') {
        throw new Error('Missing loopback address')
      }

      bases.set(connectionId, `http://127.0.0.1:${address.port}`)
    }

    vi.stubGlobal('hermesDesktop', {
      api: async (request: HermesApiRequest) => {
        const connectionId = apiRequestRegistryConnectionId(request) ?? 'local'
        const path = pathForRegistryBackendRequest(request.path, request.profile, { mode: 'local', sharedRemote: true })
        const response = await fetch(`${bases.get(connectionId)}${path}`)
        receipts.push({
          connectionId,
          profile: new URL(response.url).searchParams.get('profile')!,
          status: response.status
        })

        if (!response.ok) {
          throw new Error(`${response.status}: ${await response.text()}`)
        }

        return response.json()
      }
    })
    setApiRequestConnection(null)
    setApiRequestProfile(null)

    for (const [index, fixture] of cases.entries()) {
      list.mockResolvedValueOnce({ sessions: [fixture] })
      render(
        <MemoryRouter>
          <ArtifactsView />
        </MemoryRouter>
      )
      await waitFor(() => expect(receipts).toHaveLength(index + 1), { timeout: 5000 })
      expect(receipts.at(-1)).toEqual({
        connectionId: fixture.connection_id ?? 'local',
        profile: fixture.profile,
        status: 200
      })
      expect(await screen.findByRole('button', { name: fixture.file })).toBeTruthy()
      cleanup()
    }
  } finally {
    cleanup()
    vi.unstubAllGlobals()
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    await Promise.all(
      servers.map(
        server =>
          new Promise<void>(resolve => {
            server.closeAllConnections()
            server.close(() => resolve())
          })
      )
    )
    rmSync(root, { recursive: true, force: true })
  }
})
