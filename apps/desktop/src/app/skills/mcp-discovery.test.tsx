import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { connectDiscoveredMcp, discoverMcpServers } from '@/api/mcp-discovery'

import { McpDiscovery } from './mcp-discovery'

vi.mock('@/api/mcp-discovery', () => ({ connectDiscoveredMcp: vi.fn(), discoverMcpServers: vi.fn() }))

const candidate = {
  id: 'source-fingerprint',
  name: 'GitNexus',
  source: 'Hermes: developer',
  transport: 'stdio' as const,
  summary: 'node gitnexus mcp',
  connectable: true
}

afterEach(() => {
  cleanup()
  vi.resetAllMocks()
})

it('discovers only on request, preserves target scope, and only enables after a successful write', async () => {
  const profile = { connectionId: 'remote-a', profile: 'designer' }
  const onConnected = vi.fn(async () => {})
  vi.mocked(discoverMcpServers).mockResolvedValue({
    candidates: [
      candidate,
      {
        ...candidate,
        id: 'private',
        name: 'Private',
        connectable: false,
        reason: 'Inline credentials require manual setup'
      }
    ],
    warnings: ['Scan limited to local listeners']
  })
  vi.mocked(connectDiscoveredMcp)
    .mockRejectedValueOnce(new Error('Source changed; refresh'))
    .mockResolvedValueOnce({ ok: true, name: candidate.name })
  render(<McpDiscovery configuredNames={[]} onConnected={onConnected} profile={profile} />)
  expect(discoverMcpServers).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Find MCP servers' }))
  const toggle = await screen.findByRole('switch', { name: /GitNexus/ })
  expect(discoverMcpServers).toHaveBeenCalledWith(profile)
  expect(screen.getByText('Scan limited to local listeners')).toBeTruthy()
  expect(screen.getByRole('switch', { name: /Private/ }).hasAttribute('disabled')).toBe(true)
  fireEvent.click(toggle)
  expect((await screen.findByRole('alert')).textContent).toContain('Source changed; refresh')
  expect(toggle.getAttribute('aria-checked')).toBe('false')
  expect(onConnected).not.toHaveBeenCalled()
  fireEvent.click(toggle)
  await waitFor(() => expect(onConnected).toHaveBeenCalledOnce())
  expect(connectDiscoveredMcp).toHaveBeenLastCalledWith(candidate.id, profile)
  expect(screen.queryByRole('switch', { name: /GitNexus/ })).toBeNull()
  // A later scan is authoritative (the server may have been removed elsewhere).
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))
  expect(await screen.findByRole('switch', { name: /GitNexus/ })).toBeTruthy()
})

it('does not apply an old profile response to the newly selected profile', async () => {
  let finish!: (result: { ok: boolean; name: string }) => void
  const onOld = vi.fn(async () => {})
  const onNew = vi.fn(async () => {})
  vi.mocked(discoverMcpServers).mockResolvedValue({ candidates: [candidate], warnings: [] })
  vi.mocked(connectDiscoveredMcp).mockImplementation(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  const { rerender } = render(<McpDiscovery configuredNames={[]} onConnected={onOld} profile="old" />)
  fireEvent.click(screen.getByRole('button', { name: 'Find MCP servers' }))
  fireEvent.click(await screen.findByRole('switch', { name: /GitNexus/ }))
  rerender(<McpDiscovery configuredNames={[]} onConnected={onNew} profile="new" />)
  finish({ ok: true, name: candidate.name })
  await waitFor(() => expect(screen.getByText('Enable for new')).toBeTruthy())
  expect(screen.queryByRole('switch')).toBeNull()
  expect(onOld).not.toHaveBeenCalled()
  expect(onNew).not.toHaveBeenCalled()
  expect(connectDiscoveredMcp).toHaveBeenCalledWith(candidate.id, 'old')
})
