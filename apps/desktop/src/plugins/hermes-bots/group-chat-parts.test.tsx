import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'
import type { GroupMember } from './types'

// Group-composer mentions (#89049): the core composer's @-completion area
// doesn't mount inside workspace tiles, so the room's composers wrap the SDK
// input with a member-scoped popover of their own. Everything it inserts has
// to be a string parseGroupChatMentions resolves.

const { answerGroupClarify, host } = vi.hoisted(() => ({
  answerGroupClarify: vi.fn(async () => undefined),
  host: {} as Record<string, unknown>
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')
  const base = await pluginSdkMock(host)

  return {
    ...base,
    Button: (props: React.ComponentProps<'button'>) => <button type="button" {...props} />,
    cn: (...values: unknown[]) => values.filter(Boolean).join(' '),
    Codicon: () => null,
    Input: (props: React.ComponentProps<'input'>) => <input {...props} />,
    RowButton: (props: React.ComponentProps<'button'>) => <button type="button" {...props} />,
    Textarea: (props: React.ComponentProps<'textarea'>) => <textarea {...props} />,
    useI18n: () => ({ t: (_key: string, fallback: string) => fallback }),
    // The plugin bundle normally lands via `ctx.i18n.register` at load, so
    // without this every localized label renders empty.
    usePluginI18n: () => translateBots
  }
})

vi.mock('./group-turns', () => ({ answerGroupClarify }))

const MEMBERS: GroupMember[] = [
  { handle: 'alpha', name: 'alpha', title: '' },
  { handle: 'builder', name: 'builder', title: '' }
]

/** Render the composer the way a room does: value owned by the caller (the
 *  draft atom in production), popover scoped to the seated members. */
async function mount(initial = '') {
  const { GroupMentionInput } = await import('./group-chat-parts')
  const onChange = vi.fn()
  const onSubmitDraft = vi.fn()

  function Harness() {
    const [value, setValue] = useState(initial)

    return (
      <GroupMentionInput
        aria-label="Message Core"
        members={MEMBERS}
        onChange={next => {
          onChange(next)
          setValue(next)
        }}
        onSubmitDraft={onSubmitDraft}
        value={value}
      />
    )
  }

  render(<Harness />)

  return { input: screen.getByLabelText('Message Core') as HTMLTextAreaElement, onChange, onSubmitDraft }
}

/** Type `text`, then park the caret at `caret` (default: end of the text).
 *  The click is how the component re-reads the caret without a keystroke —
 *  jsdom does not preserve a selection across React's controlled re-write. */
function typeInto(input: HTMLTextAreaElement, text: string, caret = text.length) {
  fireEvent.change(input, { target: { value: text } })
  input.setSelectionRange(caret, caret)
  fireEvent.click(input)
}

const options = () => screen.queryAllByRole('button').map(button => button.textContent || '')

beforeEach(() => {
  vi.resetModules()
  Object.assign(host, { notify: vi.fn() })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('the @-token at the caret', () => {
  it('opens the popover on a token that begins a word', async () => {
    const { input } = await mount()

    typeInto(input, 'hey @al')

    expect(options().some(label => label.startsWith('@alpha'))).toBe(true)
  })

  it('offers @everyone and @all as the room-wide broadcast quick picks', async () => {
    const { input } = await mount()

    typeInto(input, '@')

    expect(options().some(label => label.startsWith('@everyone'))).toBe(true)
    expect(options().some(label => label.startsWith('@all'))).toBe(true)
  })

  it('narrows to @everyone as the query grows', async () => {
    const { input } = await mount()

    typeInto(input, 'ping @every')

    expect(options().filter(label => label.startsWith('@everyone'))).toHaveLength(1)
    expect(options().some(label => label.startsWith('@alpha'))).toBe(false)
  })

  it('stays closed mid-word, on plain text, and with the caret before the @', async () => {
    const { input } = await mount()

    typeInto(input, 'email me a@b')

    expect(options()).toHaveLength(0)

    typeInto(input, 'plain text')

    expect(options()).toHaveLength(0)

    typeInto(input, 'hey @al', 3)

    expect(options()).toHaveLength(0)
  })
})

describe('insertion', () => {
  it('writes exactly "@handle " — the shape parseGroupChatMentions resolves', async () => {
    const { input, onChange } = await mount()

    typeInto(input, 'hey @al')
    const option = screen.getAllByRole('button').find(button => button.textContent?.startsWith('@alpha'))
    fireEvent.mouseDown(option!)

    expect(onChange).toHaveBeenLastCalledWith('hey @alpha ')
  })

  it('preventDefaults the mousedown so the input keeps focus', async () => {
    const { input } = await mount()

    typeInto(input, 'hey @al')
    const option = screen.getAllByRole('button').find(button => button.textContent?.startsWith('@alpha'))

    // fireEvent returns false when a handler called preventDefault.
    expect(fireEvent.mouseDown(option!)).toBe(false)
  })

  it('replaces the whole token, not just the typed suffix', async () => {
    const { input, onChange } = await mount()

    typeInto(input, '@bui and then some', 4)
    const option = screen.getAllByRole('button').find(button => button.textContent?.startsWith('@builder'))
    fireEvent.mouseDown(option!)

    expect(onChange).toHaveBeenLastCalledWith('@builder  and then some')
  })

  it('lists every seated member under a bare @, keyed by handle', async () => {
    const { input } = await mount()

    typeInto(input, '@')

    // Handles only — a profile name that never resolved would surface as
    // "@undefined" and route to nobody.
    expect(options().some(label => label.startsWith('@alpha'))).toBe(true)
    expect(options().some(label => label.startsWith('@builder'))).toBe(true)
    expect(options().some(label => label.startsWith('@undefined'))).toBe(false)
  })
})

// #89884: the composer used to be a single-line Input whose form submitted on
// every Enter, so multi-line room prompts were impossible.
describe('keyboard (#89884)', () => {
  it('submits on Enter and leaves Shift+Enter to the textarea', async () => {
    const { input, onSubmitDraft } = await mount('a room prompt')

    fireEvent.keyDown(input, { key: 'Enter' })

    expect(onSubmitDraft).toHaveBeenCalledTimes(1)

    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })

    expect(onSubmitDraft).toHaveBeenCalledTimes(1)
  })

  it('inserts the highlighted mention on Enter while the popover is open', async () => {
    const { input, onChange, onSubmitDraft } = await mount()

    typeInto(input, 'hey @alp')
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(onChange).toHaveBeenLastCalledWith('hey @alpha ')
    expect(onSubmitDraft).not.toHaveBeenCalled()
  })

  // #93528: Enter here confirms composed Chinese/Japanese/Korean text. It must
  // neither insert a mention nor submit the draft. isComposing covers Chromium;
  // keyCode 229 covers macOS Chinese IMEs that fire Enter after compositionend
  // with isComposing already false.
  it('swallows IME composition Enters (#93528)', async () => {
    const { input, onSubmitDraft } = await mount('中文')

    fireEvent.keyDown(input, { isComposing: true, key: 'Enter' })
    fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 })

    expect(onSubmitDraft).not.toHaveBeenCalled()
  })
})

describe('hosted approval history', () => {
  it('does not invent a Desktop-only transcript entry', async () => {
    const { $groupChats } = await import('./group-chat')
    const { GroupClarifyCard } = await import('./group-chat-parts')

    $groupChats.set({
      Core: {
        log: [],
        members: MEMBERS,
        watermarks: {}
      }
    })
    render(
      <GroupClarifyCard
        entry={{
          at: 1,
          choices: ['once', 'deny'],
          command: 'npm test',
          group: 'Core',
          hostedApproval: {
            executionGeneration: 2,
            memberId: 'builder',
            roomId: 'room-1',
            taskId: 'task-1'
          },
          kind: 'approval',
          member: 'builder',
          memberKey: 'builder',
          multiSelect: false,
          question: 'Run tests',
          requestId: 'approval-1'
        }}
        members={MEMBERS}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'once' }))
    fireEvent.click(screen.getByRole('button', { name: 'Respond' }))
    await waitFor(() => expect(answerGroupClarify).toHaveBeenCalledTimes(1))
    expect($groupChats.get().Core.log).toEqual([])
  })

  it('maps stale gateway approval errors to actionable copy', async () => {
    answerGroupClarify.mockRejectedValueOnce(
      Object.assign(new Error('authority epoch fencing mismatch'), { code: 5119 })
    )
    const { GroupClarifyCard } = await import('./group-chat-parts')

    render(
      <GroupClarifyCard
        entry={{
          at: 1,
          choices: ['once', 'deny'],
          command: 'npm test',
          group: 'Core',
          hostedApproval: {
            executionGeneration: 2,
            memberId: 'builder',
            roomId: 'room-1',
            taskId: 'task-1'
          },
          kind: 'approval',
          member: 'builder',
          memberKey: 'builder',
          multiSelect: false,
          question: 'Run tests',
          requestId: 'approval-1'
        }}
        members={MEMBERS}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'once' }))
    fireEvent.click(screen.getByRole('button', { name: 'Respond' }))
    await waitFor(() =>
      expect(host.notify).toHaveBeenCalledWith({
        kind: 'error',
        message: 'This approval is no longer available. Refresh the Group Chat and try again.'
      })
    )
    expect(JSON.stringify((host.notify as ReturnType<typeof vi.fn>).mock.calls)).not.toContain('fencing mismatch')
  })

  it('keeps transient hosted approval failures retryable', async () => {
    answerGroupClarify.mockRejectedValueOnce(
      Object.assign(new Error('room approval target is unavailable'), { code: 5119 })
    )
    const { GroupClarifyCard } = await import('./group-chat-parts')

    render(
      <GroupClarifyCard
        entry={{
          at: 1,
          choices: ['once', 'deny'],
          command: 'npm test',
          group: 'Core',
          hostedApproval: {
            executionGeneration: 2,
            memberId: 'builder',
            roomId: 'room-1',
            taskId: 'task-1'
          },
          kind: 'approval',
          member: 'builder',
          memberKey: 'builder',
          multiSelect: false,
          question: 'Run tests',
          requestId: 'approval-1'
        }}
        members={MEMBERS}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'once' }))
    fireEvent.click(screen.getByRole('button', { name: 'Respond' }))
    await waitFor(() =>
      expect(host.notify).toHaveBeenCalledWith({
        kind: 'error',
        message: 'Could not send this approval. Check the gateway connection and try again.'
      })
    )
  })
})
