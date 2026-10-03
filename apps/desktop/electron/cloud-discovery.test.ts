import { createServer } from 'node:http'
import type { AddressInfo } from 'node:net'

import { expect, test } from 'vitest'

import { httpStatusError } from './api-transport'
import { discoverWithTeamFallback } from './cloud-discovery'

test('a stale team falls back once to current memberships; unrelated failures retain their meaning', async () => {
  const requests: string[] = []
  let status = 403
  let error = 'org_access_denied'
  let fallbackStatus = 409

  const server = createServer((req, res) => {
    requests.push(req.url!)
    const scoped = req.url!.includes('?org=')
    res.writeHead(scoped ? status : fallbackStatus, { 'Content-Type': 'application/json' })

    const body =
      fallbackStatus === 200
        ? { agents: [], org: { id: 'new-team' } }
        : { error: 'org_selection_required', orgs: [{ id: 'new-team' }] }

    res.end(JSON.stringify(scoped ? { error } : body))
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/api/agents`

  const fetchAgents = async (org?: string) => {
    const response = await fetch(`${base}${org ? `?org=${encodeURIComponent(org)}` : ''}`)
    const text = await response.text()

    if (!response.ok) {
      throw httpStatusError(response.status, text)
    }

    return JSON.parse(text)
  }

  try {
    for (const pair of [
      [403, 'org_access_denied'],
      [404, 'org_not_found']
    ] as const) {
      ;[status, error] = pair
      requests.length = 0
      await expect(discoverWithTeamFallback(fetchAgents, 'old-team')).rejects.toMatchObject({ statusCode: 409 })
      expect(requests).toEqual(['/api/agents?org=old-team', '/api/agents'])
    }

    fallbackStatus = 200
    await expect(discoverWithTeamFallback(fetchAgents, 'old-team')).resolves.toMatchObject({
      agents: [],
      org: { id: 'new-team' }
    })

    for (const pair of [
      [401, 'invalid_token'],
      [403, 'forbidden'],
      [503, 'org_access_denied']
    ] as const) {
      ;[status, error] = pair
      requests.length = 0
      await expect(discoverWithTeamFallback(fetchAgents, 'old-team')).rejects.toMatchObject({ statusCode: status })
      expect(requests).toHaveLength(1)
    }
  } finally {
    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
