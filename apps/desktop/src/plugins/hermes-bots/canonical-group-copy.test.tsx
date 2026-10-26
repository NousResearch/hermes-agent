import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => {
  const { en } = await import('@/i18n/en')
  const { CopyButton } = await import('@/components/ui/copy-button')
  const { MessageTextContent } = await import('@/components/assistant-ui/markdown-text')
  const { cn } = await import('@/lib/utils')

  return { CopyButton, MessageTextContent, Streamdown: undefined, cn, relativeTime: () => 'now',
    useI18n: () => ({ t: en }),
    Button: (props: ComponentProps<'button'>) => <button {...props} />,
    Codicon: () => <span />, Tip: ({ children }: { children: ReactNode }) => <>{children}</> }
})
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, you: 'You', download: 'Download' }) }
})

import { CanonicalGroupHistory } from './canonical-group-history'

const clipboard = vi.fn().mockResolvedValue(undefined)
const originalClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard')
const binding = { connectionId: 'local', profile: 'team', roomId: 'room-one' }
const code = 'print("hello")\nprint("world")'
const answer = `Here is the **example**:\n\n\`\`\`python\n${code}\n\`\`\``

beforeEach(() => {
  clipboard.mockClear()
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: clipboard } })
})
afterEach(() => {
  cleanup()

  if (originalClipboard) {Object.defineProperty(navigator, 'clipboard', originalClipboard)}
  else {Reflect.deleteProperty(navigator, 'clipboard')}
})

it('copies user text and full Markdown replies without sender labels or timestamps', async () => {
  render(<CanonicalGroupHistory binding={binding} events={[
    { seq: 1, kind: 'message.user', actor: { kind: 'user', id: 'desktop' }, payload: { text: 'Please show an example.' } },
    { seq: 2, kind: 'message.member', actor: { kind: 'member', id: 'helper' }, payload: { text: answer } }
  ]} />)
  expect(screen.getByText('You')).toBeTruthy()
  const buttons = screen.getAllByRole('button', { name: 'Copy' })
  fireEvent.click(buttons[0])
  await waitFor(() => expect(clipboard).toHaveBeenLastCalledWith('Please show an example.'))
  fireEvent.click(buttons[1])
  await waitFor(() => expect(clipboard).toHaveBeenLastCalledWith(answer))
})

it('copies a code block independently through the real shared message renderer', async () => {
  render(<CanonicalGroupHistory binding={binding} events={[
    { seq: 1, kind: 'message.member', actor: { kind: 'member', id: 'helper' }, payload: { text: answer } }
  ]} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Copy code' }))
  await waitFor(() => expect(clipboard).toHaveBeenCalled())
  expect(clipboard.mock.calls.at(-1)?.[0].trimEnd()).toBe(code)
})

it('does not offer an empty copy action for notices or attachment-only messages', () => {
  render(<CanonicalGroupHistory binding={binding} events={[
    { seq: 1, kind: 'turn.failed', payload: { member_id: 'helper', error: 'Timed out' } },
    { seq: 2, kind: 'message.member', actor: { kind: 'member', id: 'helper' }, payload: {} }
  ]} />)
  expect(screen.queryByRole('button', { name: 'Copy' })).toBeNull()
  expect(screen.getByText('Timed out', { exact: false })).toBeTruthy()
})
