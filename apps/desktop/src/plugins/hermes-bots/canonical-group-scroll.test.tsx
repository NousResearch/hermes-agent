import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

const { request, stick } = vi.hoisted(() => ({
  request: vi.fn(),
  stick: { isAtBottom: false, scrollRef: { current: null as HTMLElement | null }, contentRef: { current: null as HTMLElement | null }, scrollToBottom: vi.fn() }
}))

vi.mock('use-stick-to-bottom', () => ({ useStickToBottom: () => stick }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { en } = await import('@/i18n/en')
  const { ScrollArea } = await import('@/components/ui/scroll-area')
  const { cn } = await import('@/lib/utils')

  return { ScrollArea, cn, atom: (await import('nanostores')).atom,
    host: { requestProfile: request }, useI18n: () => ({ t: en }),
    MessageTextContent: ({ text }: { text: string }) => <div>{text}</div>, Streamdown: undefined,
    relativeTime: () => 'now', CopyButton: () => null,
    Button: ({ size: _size, variant: _variant, ...props }: ComponentProps<'button'> & { size?: string; variant?: string }) => <button {...props} />,
    Codicon: () => <span />, Tip: ({ children }: { children: ReactNode }) => <>{children}</> }
})
vi.mock('./group-chat-parts', () => ({
  GroupMentionInput: ({ onChange, ...props }: Omit<ComponentProps<'textarea'>, 'onChange'> & { onChange: (value: string) => void }) =>
    <textarea {...props} onChange={event => onChange(event.target.value)} />
}))
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, you: 'You', scrollToBottom: 'Scroll to bottom',
    send: 'Send', stop: 'Stop', retry: 'Retry', download: 'Download' }) }
})
import { CanonicalGroupWorkspace } from './canonical-group-workspace'

const binding = { connectionId: 'local', profile: 'team', roomId: 'scroll-room' }
afterEach(() => { cleanup(); request.mockReset(); stick.scrollToBottom.mockClear(); stick.isAtBottom = false; localStorage.clear() })

it('binds sticky scrolling to the actual viewport and exposes a keyboard-accessible jump control', async () => {
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : { events: [] })
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect(stick.scrollRef.current).toBeTruthy())
  expect(stick.scrollRef.current?.dataset.slot).toBe('scroll-area-viewport')
  expect(stick.scrollRef.current?.contains(stick.contentRef.current)).toBe(true)
  expect(view.container.querySelector('[data-slot="scroll-area-scrollbar"]')).toBeTruthy()
  const jump = screen.getByRole('button', { name: 'Scroll to bottom' })
  jump.focus()
  expect(jump.matches(':focus')).toBe(true)
  fireEvent.click(jump)
  expect(stick.scrollToBottom).toHaveBeenCalledOnce()
  stick.isAtBottom = true
  view.rerender(<CanonicalGroupWorkspace binding={binding} />)
  expect(screen.queryByRole('button', { name: 'Scroll to bottom' })).toBeNull()
})

it('resumes following the tail after the user sends from a scrolled-up position', async () => {
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] } : { accepted: true })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'A new question' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(stick.scrollToBottom).toHaveBeenCalledWith({ animation: 'instant' }))
})
