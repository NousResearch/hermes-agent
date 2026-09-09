import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupTurns from './group-turns'
import { translateBots } from './i18n-test-helper'
import type { GroupMember, GroupPrompt } from './types'

// Group-composer mentions (#89049): the core composer's @-completion area
// doesn't mount inside workspace tiles, so the room's composers wrap the SDK
// input with a member-scoped popover of their own. Everything it inserts has
// to be a string parseGroupChatMentions resolves.

const { answerGroupClarify, host } = vi.hoisted(() => ({
  answerGroupClarify: vi.fn<typeof groupTurns.answerGroupClarify>(async () => undefined),
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
  it('labels approval decisions without changing choices, member identity or submitted payloads', async () => {
    const { GroupClarifyCard } = await import('./group-chat-parts')
    const { groupMemberKey } = await import('./group-membership')

    const reviewer: GroupMember = {
      display_name: 'Safety Reviewer',
      hostedIdentity: {
        installationId: 'review-host',
        memberId: 'reviewer',
        profile: 'approval-reviewer-20260905',
        roomId: 'room-1'
      },
      name: 'approval-reviewer-20260905'
    }

    const choices = ['once', 'session', 'always', 'deny', 'toString']
    const labels = ['Allow once', 'Allow this session', 'Always allow', 'Deny', 'toString']

    const entry: GroupPrompt = {
      at: 1,
      choices,
      command: 'npm test',
      group: 'Core',
      hostedApproval: { executionGeneration: 2, memberId: 'reviewer', roomId: 'room-1', taskId: 'task-1' },
      kind: 'approval',
      member: reviewer.name,
      memberKey: groupMemberKey(reviewer),
      multiSelect: false,
      question: 'Run tests',
      requestId: 'approval-1'
    }

    const original = structuredClone(entry)

    const { rerender } = render(
      <GroupClarifyCard entry={entry} members={[{ name: reviewer.name, title: 'Other reviewer' }, reviewer]} />
    )

    expect(screen.getByText('Safety Reviewer wants to run a command:')).toBeTruthy()
    expect(screen.getByText(entry.command!)).toBeTruthy()
    expect(screen.getByText(entry.question)).toBeTruthy()
    expect(screen.queryByRole('textbox')).toBeNull()
    const submit = screen.getByRole('button', { name: 'Submit decision' }) as HTMLButtonElement
    expect(submit.disabled).toBe(true)
    expect(options()).toEqual([...labels, 'Submit decision'])

    for (const [index, choice] of choices.entries()) {
      answerGroupClarify.mockClear()
      fireEvent.click(screen.getByRole('button', { name: labels[index] }))
      expect(answerGroupClarify).not.toHaveBeenCalled()
      fireEvent.click(submit)
      await waitFor(() => expect(submit.disabled).toBe(false))
      expect(answerGroupClarify).toHaveBeenCalledExactlyOnceWith(entry, reviewer, choice)
      expect(answerGroupClarify.mock.calls[0][0]).toBe(entry)
      expect(answerGroupClarify.mock.calls[0][1]).toBe(reviewer)
      expect(entry).toEqual(original)
    }

    rerender(<GroupClarifyCard entry={entry} members={[]} />)
    expect(screen.getByText('@approval-reviewer-20260905 wants to run a command:')).toBeTruthy()
    expect(submit.disabled).toBe(true)
  })

  it('preserves ordinary clarify labels, scalar, multi-select, batch and free-text answers', async () => {
    const { GroupClarifyCard } = await import('./group-chat-parts')
    const choices = ['once', 'session', 'always', 'deny']

    for (const mode of ['single', 'multi', 'batch', 'text']) {
      const entry: GroupPrompt = {
        at: 1,
        choices,
        group: 'Core',
        kind: 'clarify',
        member: 'builder',
        memberKey: 'builder',
        multiSelect: mode === 'multi',
        question: 'Choose a word',
        ...(mode === 'batch' ? { questions: [{ qid: 'word', question: 'Choose a word', choices }] } : {}),
        requestId: 'clarify-1'
      }

      const original = structuredClone(entry)
      const { unmount } = render(<GroupClarifyCard entry={entry} members={MEMBERS} />)
      expect(screen.getByText('@builder asks:')).toBeTruthy()
      expect(screen.getByText(entry.question)).toBeTruthy()
      expect(options()).toEqual([...choices, 'Answer'])
      const input = screen.getByRole('textbox')
      expect(input.getAttribute('placeholder')).toBe('Or type your own answer…')
      const submit = screen.getByRole('button', { name: 'Answer' }) as HTMLButtonElement

      for (const choice of choices) {
        answerGroupClarify.mockClear()
        fireEvent.click(screen.getByRole('button', { name: choice }))

        if (mode === 'text') {
          fireEvent.change(input, { target: { value: `  custom ${choice}  ` } })
        }

        expect(answerGroupClarify).not.toHaveBeenCalled()
        fireEvent.click(submit)
        await waitFor(() => expect(submit.disabled).toBe(false))

        const answer =
          mode === 'batch'
            ? { word: choice }
            : mode === 'multi'
              ? JSON.stringify([choice])
              : mode === 'text'
                ? `custom ${choice}`
                : choice

        expect(answerGroupClarify).toHaveBeenCalledExactlyOnceWith(entry, MEMBERS[1], answer)
        expect(entry).toEqual(original)

        if (mode === 'multi') {
          fireEvent.click(screen.getByRole('button', { name: choice }))
        }
      }

      unmount()
    }
  })

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

    expect(options()).toEqual(['Allow once', 'Deny', 'Submit decision'])
    fireEvent.click(screen.getByRole('button', { name: 'Allow once' }))
    fireEvent.click(screen.getByRole('button', { name: 'Submit decision' }))
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

    fireEvent.click(screen.getByRole('button', { name: 'Allow once' }))
    fireEvent.click(screen.getByRole('button', { name: 'Submit decision' }))
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

    fireEvent.click(screen.getByRole('button', { name: 'Allow once' }))
    fireEvent.click(screen.getByRole('button', { name: 'Submit decision' }))
    await waitFor(() =>
      expect(host.notify).toHaveBeenCalledWith({
        kind: 'error',
        message: translateBots('group.hostedApprovalRetry')
      })
    )
    const respond = screen.getByRole('button', { name: 'Submit decision' }) as HTMLButtonElement
    await waitFor(() => expect(respond.disabled).toBe(false))
    fireEvent.click(respond)
    await waitFor(() => expect(answerGroupClarify).toHaveBeenCalledTimes(2))
    expect(answerGroupClarify.mock.calls[1]).toEqual(answerGroupClarify.mock.calls[0])
  })
})
