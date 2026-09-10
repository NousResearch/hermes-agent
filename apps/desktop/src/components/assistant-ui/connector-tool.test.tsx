import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { createConnectorFlow } from '@/store/connector-flow'

import { ConnectorOffer } from './connector-tool'

const gmail = { connector: 'gmail', enabled: true, connected: false }
const slack = { connector: 'slack', enabled: true, connected: false }
afterEach(cleanup)

it('uses real card controls for partial success, skip and explicit continuation', async () => {
  const request = vi.fn().mockResolvedValue({ available: true, connectors: [gmail, slack] })
  const open = vi.fn().mockResolvedValue(undefined)
  const flow = createConnectorFlow('session-a', [gmail, slack], { request, open, delay: async () => undefined })
  await flow.refresh()
  const onContinue = vi.fn().mockResolvedValue(undefined)
  render(<ConnectorOffer busy={false} flow={flow} onContinue={onContinue} />)
  expect(screen.getAllByRole('button', { name: 'Connect' })).toHaveLength(2)
  request
    .mockResolvedValueOnce({
      results: [{ connector: 'gmail', status: 'initiated', connect_url: 'https://connect.example.test/link' }]
    })
    .mockResolvedValue({ available: true, connectors: [{ ...gmail, connected: true }, slack] })
  fireEvent.click(screen.getAllByRole('button', { name: 'Connect' })[0])
  await waitFor(() => expect(screen.getByText('Connected')).toBeTruthy())
  expect(open).toHaveBeenCalledOnce()
  fireEvent.click(screen.getByRole('button', { name: 'Not now' }))
  expect(screen.getByText('Skipped')).toBeTruthy()
  expect(onContinue).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Continue in chat' }))
  await waitFor(() => expect(onContinue).toHaveBeenCalledOnce())
})

it('keeps cancellation available and exposes retry after a failure', async () => {
  const request = vi.fn().mockResolvedValue({ available: true, connectors: [gmail] })
  const flow = createConnectorFlow('session-a', [gmail], { request, open: vi.fn() })
  await flow.refresh()
  render(<ConnectorOffer busy={false} flow={flow} onContinue={vi.fn()} />)
  request.mockRejectedValueOnce(new Error('offline'))
  fireEvent.click(screen.getByRole('button', { name: 'Connect' }))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy())
  expect(screen.getByText('Could not start authorization. Try again.')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Not now' }))
  expect(screen.getByText('Skipped')).toBeTruthy()
})
