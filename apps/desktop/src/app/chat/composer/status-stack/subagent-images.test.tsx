import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import * as gateway from '@/store/gateway'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

import { SubagentTranscript } from './subagent-transcript'

const IMAGE =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aU1sAAAAASUVORK5CYII='

afterEach(() => {
  cleanup()
  _resetSessionOwnerHintsForTests()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

it('shows child image metadata through the parent gateway but reads using the child session', async () => {
  setSessionOwnerHint('parent', { connectionId: 'remote-owner', profile: 'artist' })
  vi.spyOn(gateway, 'requestGatewayForAgent').mockResolvedValue({
    available: true,
    text: 'Actual child progress',
    truncated: false,
    image_session_id: 'child-stored',
    images: ['/tmp/a.png', '/tmp/b.png'],
    images_truncated: true
  } as never)
  const api = vi.fn().mockResolvedValue({ dataUrl: IMAGE })
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  render(<SubagentTranscript sessionId="parent" subagentId="worker" />)
  await screen.findByRole('img', { name: /Image 1 of 2/ })
  expect(api).toHaveBeenCalledWith(
    expect.objectContaining({
      connectionId: 'remote-owner',
      profile: 'artist',
      path: '/api/fs/read-data-url?path=%2Ftmp%2Fa.png&session_id=child-stored'
    })
  )
  fireEvent.click(screen.getByRole('button', { name: 'Next image' }))
  await screen.findByRole('img', { name: /Image 2 of 2/ })
  expect(screen.getByText('Some image previews were omitted.')).toBeTruthy()
  expect(screen.getByText('Actual child progress')).toBeTruthy()
})

it('discards late image metadata from a previously selected child', async () => {
  setSessionOwnerHint('parent', { connectionId: 'remote-owner', profile: 'artist' })
  let finish!: (value: unknown) => void
  vi.spyOn(gateway, 'requestGatewayForAgent').mockImplementation(async (_c, _p, _m, params) => {
    if ((params as { subagent_id: string }).subagent_id === 'old') {
      return (await new Promise<unknown>(resolve => {
        finish = resolve
      })) as never
    }

    return {
      available: true,
      text: 'New worker',
      truncated: false,
      image_session_id: 'new-child',
      images: ['/tmp/new.png']
    } as never
  })
  const api = vi.fn().mockResolvedValue({ dataUrl: IMAGE })
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  const view = render(<SubagentTranscript sessionId="parent" subagentId="old" />)
  await waitFor(() => expect(finish).toBeTypeOf('function'))
  view.rerender(<SubagentTranscript sessionId="parent" subagentId="new" />)
  await screen.findByRole('img')
  await act(async () =>
    finish({
      available: true,
      text: 'Old private output',
      truncated: false,
      image_session_id: 'old-child',
      images: ['/tmp/old.png']
    })
  )
  expect(screen.queryByText('Old private output')).toBeNull()
  expect(api.mock.calls.every(([request]) => request.path.includes('session_id=new-child'))).toBe(true)
})

it('keeps legacy text readable and never borrows the parent session for incomplete image metadata', async () => {
  setSessionOwnerHint('parent', { connectionId: 'remote-owner', profile: 'artist' })
  vi.spyOn(gateway, 'requestGatewayForAgent').mockResolvedValue({
    available: true,
    text: 'Legacy transcript',
    truncated: false,
    images: ['/tmp/unknown.png']
  } as never)
  const api = vi.fn()
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  render(<SubagentTranscript sessionId="parent" subagentId="worker" />)
  await screen.findByText('Legacy transcript')
  expect(screen.queryByRole('region', { name: 'Image gallery' })).toBeNull()
  expect(api).not.toHaveBeenCalled()
})

it('does not restart slow image reads when only the polled text changes', async () => {
  const clock = vi.spyOn(window, 'setInterval')
  setSessionOwnerHint('parent', { connectionId: 'remote-owner', profile: 'artist' })
  let tick = 0
  vi.spyOn(gateway, 'requestGatewayForAgent').mockImplementation(
    async () =>
      ({
        available: true,
        text: `Progress ${++tick}`,
        truncated: false,
        image_session_id: 'child-stored',
        images: ['/tmp/slow.png']
      }) as never
  )
  let finish!: (value: unknown) => void

  const api = vi.fn(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )

  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  render(<SubagentTranscript sessionId="parent" subagentId="worker" />)
  await waitFor(() => expect(api).toHaveBeenCalledTimes(1))
  await act(async () => {
    const poll = clock.mock.calls.find(([, milliseconds]) => milliseconds === 2000)![0] as () => void
    poll()
    await Promise.resolve()
    poll()
  })
  expect(api).toHaveBeenCalledTimes(1)
  expect(screen.getByText('Progress 2')).toBeTruthy()
  await act(async () => finish({ dataUrl: IMAGE }))
  expect((await screen.findByRole('img')).getAttribute('src')).toBe(IMAGE)
})

it('retires old gateway pixels when ownership changes while a read is pending', async () => {
  const clock = vi.spyOn(window, 'setInterval')
  setSessionOwnerHint('parent', { connectionId: 'old-host', profile: 'artist' })
  vi.spyOn(gateway, 'requestGatewayForAgent').mockResolvedValue({
    available: true,
    text: 'Worker',
    truncated: false,
    image_session_id: 'child-stored',
    images: ['/tmp/same.png']
  } as never)
  let finishOld!: (value: unknown) => void

  const api = vi.fn(request =>
    request.connectionId === 'old-host'
      ? new Promise(resolve => {
          finishOld = resolve
        })
      : Promise.resolve({ dataUrl: IMAGE })
  )

  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  render(<SubagentTranscript sessionId="parent" subagentId="worker" />)
  await waitFor(() => expect(finishOld).toBeTypeOf('function'))
  _resetSessionOwnerHintsForTests()
  setSessionOwnerHint('parent', { connectionId: 'new-host', profile: 'artist' })
  await act(async () => {
    const poll = clock.mock.calls.find(([, milliseconds]) => milliseconds === 2000)![0] as () => void
    poll()
  })
  expect((await screen.findByRole('img')).getAttribute('src')).toBe(IMAGE)
  await act(async () => finishOld({ dataUrl: 'data:image/png;base64,T0xE' }))
  expect(screen.getByRole('img').getAttribute('src')).toBe(IMAGE)
  expect(api).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'new-host' }))
})

it('reports omitted previews even when no image is displayable', async () => {
  setSessionOwnerHint('parent', { connectionId: 'remote-owner', profile: 'artist' })
  vi.spyOn(gateway, 'requestGatewayForAgent').mockResolvedValue({
    available: true,
    text: 'Large output',
    truncated: false,
    image_session_id: 'child',
    images: [],
    images_truncated: true
  } as never)
  render(<SubagentTranscript sessionId="parent" subagentId="worker" />)
  expect(await screen.findByText('Some image previews were omitted.')).toBeTruthy()
  expect(screen.queryByRole('img')).toBeNull()
})

it('refreshes overwritten image paths on a new image revision without resetting selection', async () => {
  const clock = vi.spyOn(window, 'setInterval')
  setSessionOwnerHint('parent', { connectionId: 'remote-owner', profile: 'artist' })
  let revision = 1
  vi.spyOn(gateway, 'requestGatewayForAgent').mockImplementation(
    async () =>
      ({
        available: true,
        text: 'Same log text',
        truncated: false,
        image_session_id: 'child',
        images: ['/tmp/a.png', '/tmp/b.png'],
        image_revision: revision
      }) as never
  )
  const api = vi.fn().mockResolvedValue({ dataUrl: IMAGE })
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  render(<SubagentTranscript sessionId="parent" subagentId="worker" />)
  await screen.findByRole('img', { name: /Image 1 of 2/ })
  fireEvent.click(screen.getByRole('button', { name: 'Next image' }))
  await screen.findByRole('img', { name: /Image 2 of 2/ })
  const before = api.mock.calls.length
  revision = 2
  await act(async () => {
    const poll = clock.mock.calls.find(([, ms]) => ms === 2000)![0] as () => void
    poll()
  })
  await waitFor(() => expect(api.mock.calls.length).toBeGreaterThan(before))
  await screen.findByRole('img', { name: /Image 2 of 2/ })
})
