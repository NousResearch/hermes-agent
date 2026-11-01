import { useStore } from '@nanostores/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return {
    ...(await pluginSdkMock(gateway.host)),
    atom,
    useValue: useStore,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => (key: string) =>
      CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    host: { ...gateway.host, requestProfile: request }
  }
})
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
const originalDesktop = window.hermesDesktop
beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined })
})
afterEach(() => {
  cleanup()
  request.mockReset()
  localStorage.clear()
  window.hermesDesktop = originalDesktop
})

// C21: while a selected file's upload acknowledgement is outstanding, Send cannot dispatch the text
// without it, and the late manifest never lands on the cleared next draft.
it('holds Send until the selected attachment is committed, then sends it with the text', async () => {
  let ack!: () => void
  const uploaded = { attachment_id: 'file-one', kind: 'file', name: 'report.txt', mime: 'text/plain', size: 2 }
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {
      return { room: { name: 'Room' }, driver_status: {} }
    }

    if (method === 'groups.log') {
      return { events: [] }
    }

    if (method === 'groups.attachment.upload') {
      return new Promise(resolve => {
        ack = () => resolve(uploaded)
      })
    }

    return {}
  })
  const view = render(
    <CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'upload-race' }} />
  )
  await screen.findByText('Room')
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.change(view.container.querySelector('input[type=file]')!, {
    target: { files: [new File(['hi'], 'report.txt', { type: 'text/plain' })] }
  })
  await waitFor(() => expect(request.mock.calls.some(c => c[1] === 'groups.attachment.upload')).toBe(true))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'see file' } })
  const sendButton = screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement
  expect(sendButton.disabled).toBe(true)
  fireEvent.submit(sendButton.closest('form')!)
  expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(false)
  ack()
  await screen.findByText('report.txt')
  await waitFor(() => expect(sendButton.disabled).toBe(false))
  fireEvent.click(sendButton)
  await waitFor(() => expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(true))
  const sent = request.mock.calls.filter(c => c[1] === 'groups.send')
  expect(sent).toHaveLength(1)
  expect(sent[0][2].payload).toMatchObject({ text: 'see file', attachments: [uploaded] })
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(''))
  expect(screen.queryByText('report.txt')).toBeNull()
})
