/**
 * The leaf components a group-chat room composes: the room picture controls,
 * the mention-aware composer input, and the card a member's pending
 * clarify/approval prompt renders as.
 *
 * None of them reaches back into the room surface, so they sit below
 * `group-chat-view.tsx` — and the group-creation dialog can share the picture
 * controls without either surface importing the other.
 */

import { Button, cn, Codicon, host, Input, RowButton, Textarea, useI18n, useValue } from '@hermes/plugin-sdk'
import type { ClipboardEvent } from 'react'
import { useRef, useState } from 'react'

import { $imagenAvailable, normalizeAvatarImage, pickImageFromDevice, probeImagen } from './avatar-image'
import { $botMeta, botHandle, botMentionTag } from './data'
import { appendGroupChatEntry } from './group-chat'
import { groupMemberKey } from './group-membership'
import { answerGroupClarify } from './group-turns'
import { useBots } from './i18n'
import { displayName } from './labels'
import { botRosterMeta } from './routing'
import type { BotMeta, GroupMember, GroupPrompt } from './types'

/** The `image.generate` reply. Older gateways answer `image`, newer ones
 *  `image_data`; both are data URLs. */
interface ImageGenerateResponse {
  error?: string
  image?: string
  image_data?: string
  success?: boolean
}

interface GroupImageControlsProps {
  image: null | string
  onImage: (image: null | string) => void
  seedMembers?: string[]
  seedName: string
}

/** Compact picture controls shared by group-chat creation and settings:
 *  a live preview (image, else the organization glyph), Upload / Generate /
 *  Remove. Reuses the bot-avatar pipeline (device picker, 256px normalize,
 *  image.generate probe) so room pictures cost the same as bot avatars. */
export function GroupImageControls({ image, onImage, seedName, seedMembers }: GroupImageControlsProps) {
  const b = useBots()
  const { t } = useI18n()
  const imagen = useValue($imagenAvailable)
  const [busy, setBusy] = useState(false)

  if (imagen === null) {
    void probeImagen()
  }

  const upload = async () => {
    const raw = await pickImageFromDevice()

    if (raw) {
      onImage(await normalizeAvatarImage(raw))
    }
  }

  const generate = async () => {
    if (busy) {
      return
    }

    setBusy(true)

    try {
      const who = [seedName, seedMembers?.length ? `a team of ${seedMembers.join(', ')}` : '']
        .filter(Boolean)
        .join(' — ')

      const res = await host.request<ImageGenerateResponse>('image.generate', {
        prompt:
          `Group chat icon for an AI agent team called "${who || 'a bot team'}". ` +
          'Friendly minimal emblem, bold flat vector style, solid color background, centered, no text.',
        aspect_ratio: 'square'
      })

      if (!res?.success) {
        throw new Error(res?.error || 'generation failed')
      }

      const img = res.image_data || res.image

      if (img) {
        onImage(await normalizeAvatarImage(img))
      }
    } catch (err) {
      host.notifyError(err, b.group.pictureGenerationFailed)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex items-center gap-2">
      <div className="flex size-10 shrink-0 items-center justify-center overflow-hidden rounded-full bg-(--chrome-action-hover)">
        {image ? (
          <img alt="" className="size-full object-cover" src={image} />
        ) : (
          <Codicon className="text-(--ui-text-tertiary)" name="organization" />
        )}
      </div>
      <Button onClick={upload} size="sm" type="button" variant="secondary">
        {b.avatar.upload}
      </Button>
      {imagen ? (
        <Button disabled={busy} onClick={generate} size="sm" type="button" variant="secondary">
          {busy ? b.avatar.generating : b.avatar.generate}
        </Button>
      ) : null}
      {image ? (
        <Button onClick={() => onImage(null)} size="sm" type="button" variant="ghost">
          {t.common.remove}
        </Button>
      ) : null}
    </div>
  )
}

interface MentionToken {
  query: string
  /** Index of the '@' the token starts at. */
  start: number
}

/** Merged room view for one group: shared timeline with per-member
 *  attribution, a composer that drives the round-robin, and a working
 *  indicator while member turns run. Renders identically in the MAIN chat
 *  window (host.openWorkspace tile) and in the bots panel (older-desktop
 *  fallback); `onBack` is where the Back button routes — the main tile's
 *  closer, or clearing the in-panel workspace atom. */
/** The active @-token at the caret: text from the nearest '@' (that begins a
 *  word) up to the caret, or null when the caret isn't inside a mention. */
function mentionTokenAt(text: string, caret: number): MentionToken | null {
  const upto = String(text || '').slice(0, caret)
  const match = /(^|\s)@([a-z0-9._-]*)$/i.exec(upto)

  if (!match) {
    return null
  }

  return {
    query: match[2].toLowerCase(),
    start: caret - match[2].length - 1
  }
}

/** One row of the mention popover: the handle inserted, plus its subtitle. */
interface MentionOption {
  handle: string
  meta: string
}

interface GroupMentionInputProps {
  'aria-label'?: string
  autoFocus?: boolean
  className?: string
  members: GroupMember[]
  onChange: (value: string) => void
  onPaste?: (event: ClipboardEvent<HTMLTextAreaElement>) => void
  onSubmitDraft?: () => void
  placeholder?: string
  value: string
}

/** Mention-aware composer input for group rooms. The core composer's
 *  @-completion area doesn't mount inside workspace tiles (#89049), so this
 *  wraps the plain SDK Input with a member-scoped popover: @everyone/@all
 *  quick picks plus each seated member's handle. Insertion produces exactly
 *  the strings parseGroupChatMentions resolves. Keyboard: Up/Down navigate,
 *  Enter/Tab insert (Enter falls through to submit when the popover is
 *  closed), Escape dismisses. */
export function GroupMentionInput({ members, onChange, onSubmitDraft, value, ...inputProps }: GroupMentionInputProps) {
  const b = useBots()
  const allMeta: Record<string, BotMeta> = useValue($botMeta)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const [token, setToken] = useState<MentionToken | null>(null)
  const [selected, setSelected] = useState(0)
  const options: MentionOption[] = []

  if (token) {
    for (const pick of ['everyone', 'all']) {
      if (pick.startsWith(token.query)) {
        options.push({
          handle: pick,
          meta: b.group.everyoneMeta
        })
      }
    }

    for (const member of members) {
      const handle = String(member.handle || botHandle(member.name, member) || '').trim()
      const display = displayName(member, botRosterMeta(member, allMeta))
      // Renamed members complete on their friendly tag; parser resolves both.
      const tag = String(botMentionTag(member) || handle).trim()

      if (!tag) {
        continue
      }

      if (
        token.query &&
        !tag.toLowerCase().startsWith(token.query) &&
        !(handle && handle.toLowerCase().startsWith(token.query)) &&
        !display.toLowerCase().startsWith(token.query)
      ) {
        continue
      }

      options.push({
        handle: tag,
        meta: display
      })
    }
  }

  const open = Boolean(token) && options.length > 0
  const active = open ? Math.min(selected, options.length - 1) : 0

  // The onClick below passes `event.target`, which React types as a bare
  // EventTarget even though a click on this textarea always targets it.
  const refreshToken = (target: HTMLTextAreaElement) => {
    setToken(mentionTokenAt(target.value, target.selectionStart ?? target.value.length))
    setSelected(0)
  }

  const insert = (handle: string) => {
    if (!token) {
      return
    }

    const caret = inputRef.current?.selectionStart ?? value.length
    const next = `${value.slice(0, token.start)}@${handle} ${value.slice(caret)}`
    onChange(next)
    setToken(null)

    // Restore focus with the caret after the inserted mention.
    const pos = token.start + handle.length + 2
    requestAnimationFrame(() => {
      const el = inputRef.current

      if (el) {
        el.focus()

        try {
          el.setSelectionRange(pos, pos)
        } catch {
          /* input type without selection support */
        }
      }
    })
  }

  return (
    <div className="relative min-w-0 flex-1">
      {open ? (
        <div className="absolute bottom-full left-0 z-50 mb-1 max-h-48 w-64 overflow-y-auto rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) py-1 shadow-lg">
          {options.map((option, index) => (
            <RowButton
              className={cn(
                'flex w-full items-baseline gap-2 px-2 py-1 text-left text-xs',
                index === active ? 'bg-(--ui-control-hover-background) text-foreground' : 'text-(--ui-text-secondary)'
              )} // preventDefault on mousedown so the input keeps focus.
              key={option.handle}
              onMouseDown={event => {
                event.preventDefault()
                insert(option.handle)
              }}
              onMouseEnter={() => setSelected(index)}
            >
              <span className="font-medium">{`@${option.handle}`}</span>
              <span className="truncate text-[0.65rem] text-(--ui-text-quaternary)">{option.meta}</span>
            </RowButton>
          ))}
        </div>
      ) : null}
      <Textarea
        {...inputProps}
        // Input whose form submitted on every Enter — newlines were
        // impossible. Enter (no Shift) still submits via onSubmitDraft;
        // Shift+Enter falls through to the textarea's native newline.
        className={cn('max-h-40 min-h-9 resize-none', inputProps.className)}
        onBlur={() => setToken(null)}
        onChange={event => {
          onChange(event.target.value)
          refreshToken(event.target)
        }}
        onClick={event => refreshToken(event.target as HTMLTextAreaElement)}
        onKeyDown={event => {
          // IME composition guard (same as the core composer): Enter here
          // confirms the composed Chinese/Japanese/Korean text — it must not
          // insert a mention nor submit the draft. nativeEvent.isComposing
          // covers Chromium; keyCode 229 covers macOS Chinese IMEs that fire
          // Enter after compositionend with isComposing already false.
          if (event.nativeEvent?.isComposing || event.keyCode === 229) {
            return
          }

          if (open) {
            if (event.key === 'ArrowDown') {
              event.preventDefault()
              setSelected((active + 1) % options.length)

              return
            }

            if (event.key === 'ArrowUp') {
              event.preventDefault()
              setSelected((active - 1 + options.length) % options.length)

              return
            }

            if (event.key === 'Enter' || event.key === 'Tab') {
              event.preventDefault()
              insert(options[active].handle)

              return
            }

            if (event.key === 'Escape') {
              event.preventDefault()
              setToken(null)

              return
            }
          }

          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            onSubmitDraft?.()
          }
        }}
        ref={inputRef}
        rows={1} // Multi-line room prompts (#89884): the composer was a single-line
        value={value}
      />
    </div>
  )
}

/** A pending prompt as the room renders it. */
export interface GroupRoomPrompt extends GroupPrompt {
  /** TODO(bot-mode-types): nothing ever sets this — syncGroupClarify builds
   *  every entry without a thread — so the answer GroupClarifyCard echoes back
   *  into the room always lands in the 'legacy' thread instead of the thread
   *  the member asked from. */
  thread?: string
}

/** A sub-question normalized for rendering: one card row, one answer. */
interface GroupClarifyQuestion {
  choices: string[]
  multiSelect: boolean
  qid: string
  question: string
}

interface GroupClarifyCardProps {
  entry: GroupRoomPrompt
  members: GroupMember[]
}

/** One member's pending prompt, rendered in the room (#90694).
 *  - clarify: choices as tap buttons (multi-select stages; single-select
 *    stages one), free text always available, batch sub-questions each get
 *    their own input.
 *  - approval: the command in a code row plus the server's choice set
 *    (once/session/always/deny) as buttons — no free text; approvals are a
 *    closed choice. Answer sends via the member's own source. */
export function GroupClarifyCard({ entry, members }: GroupClarifyCardProps) {
  const b = useBots()
  const { group } = entry
  const isApproval = entry.kind === 'approval'
  const member = members.find(m => groupMemberKey(m) === entry.memberKey) || members.find(m => m.name === entry.member)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [picked, setPicked] = useState<Record<string, string[]>>({})
  const [sending, setSending] = useState(false)

  const questions: GroupClarifyQuestion[] =
    entry.questions && entry.questions.length
      ? entry.questions.map((q, i) => ({
          qid: q?.qid ?? q?.id ?? `q${i}`,
          question: typeof q?.question === 'string' ? q.question : '',
          choices: Array.isArray(q?.choices) ? q.choices.filter(c => typeof c === 'string' && c) : [],
          multiSelect: Boolean(q?.multi_select ?? q?.multiSelect)
        }))
      : [
          {
            qid: '__single__',
            question: entry.question,
            choices: entry.choices,
            multiSelect: entry.multiSelect
          }
        ]

  const answerFor = (q: GroupClarifyQuestion) => {
    const chosen = picked[q.qid] || []

    if (chosen.length) {
      return q.multiSelect ? JSON.stringify(chosen) : chosen[0]
    }

    return isApproval ? '' : (drafts[q.qid] || '').trim()
  }

  const allAnswered = questions.every(q => answerFor(q))

  const submit = async () => {
    if (!member || sending || !allAnswered) {
      return
    }

    setSending(true)

    try {
      if (isApproval || !(entry.questions && entry.questions.length)) {
        await answerGroupClarify(entry, member, answerFor(questions[0]))
      } else {
        const answers: Record<string, string> = {}

        for (const q of questions) {
          answers[q.qid] = answerFor(q)
        }

        await answerGroupClarify(entry, member, answers)
      }

      // Echo the exchange into the room log so the thread reads complete.
      const summary = isApproval
        ? `${answerFor(questions[0])} — ${entry.command || entry.question || b.group.commandApproval}`
        : questions.map(q => (questions.length > 1 ? `${q.question}: ${answerFor(q)}` : answerFor(q))).join('\n')

      appendGroupChatEntry(
        group,
        {
          kind: 'user',
          name: 'You'
        },
        summary,
        entry.thread || 'legacy'
      )
    } catch (err: any) {
      host.notify({
        kind: 'error',
        message: b.group.answerFailed(botHandle(entry.member, member), String(err?.message || err))
      })
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="grid gap-1.5 rounded-md border border-(--ui-accent)/50 bg-(--ui-accent)/5 px-2.5 py-2">
      <div className="flex items-center gap-1.5 text-xs font-medium">
        <Codicon className="shrink-0 text-(--ui-accent)" name={isApproval ? 'shield' : 'question'} />
        {isApproval
          ? b.group.wantsToRunCommand(botHandle(entry.member, member))
          : b.group.asks(botHandle(entry.member, member))}
      </div>
      {isApproval && entry.command ? (
        <code className="block overflow-x-auto rounded bg-(--ui-bg-secondary,rgba(0,0,0,0.25)) px-2 py-1 font-mono text-[0.7rem] whitespace-pre-wrap break-all">
          {entry.command}
        </code>
      ) : null}
      {questions.map(q => (
        <div className="grid gap-1" key={`q:${q.qid}`}>
          {q.question ? <div className="text-xs whitespace-pre-wrap">{q.question}</div> : null}
          {q.choices.length ? (
            <div className="flex flex-wrap gap-1">
              {q.choices.map(choice => {
                const chosen = (picked[q.qid] || []).includes(choice)

                return (
                  <Button
                    className={cn(
                      'h-6 px-2 text-[0.7rem]',
                      isApproval && choice === 'deny' && !chosen && 'text-destructive'
                    )}
                    key={`choice:${q.qid}:${choice}`}
                    onClick={() => {
                      setDrafts(prev => ({
                        ...prev,
                        [q.qid]: ''
                      }))
                      setPicked(prev => {
                        const current = prev[q.qid] || []

                        const next = q.multiSelect
                          ? chosen
                            ? current.filter(c => c !== choice)
                            : [...current, choice]
                          : chosen
                            ? []
                            : [choice]

                        return {
                          ...prev,
                          [q.qid]: next
                        }
                      })
                    }}
                    size="sm"
                    variant={chosen ? 'default' : 'secondary'}
                  >
                    {choice}
                  </Button>
                )
              })}
            </div>
          ) : null}
          {/* Approvals are a closed choice set — no free-text input. */}
          {isApproval ? null : (
            <Input
              aria-label={b.group.answerTo(entry.member)}
              className="h-7 text-xs"
              key={`input:${q.qid}`}
              onChange={event => {
                const value = event.target.value
                setPicked(prev => ({
                  ...prev,
                  [q.qid]: []
                }))
                setDrafts(prev => ({
                  ...prev,
                  [q.qid]: value
                }))
              }}
              onKeyDown={event => {
                // IME guard: Enter confirming a composed word must not submit.
                if (event.nativeEvent?.isComposing || event.keyCode === 229) {
                  return
                }

                if (event.key === 'Enter' && questions.length === 1) {
                  event.preventDefault()
                  void submit()
                }
              }}
              placeholder={q.choices.length ? 'Or type your own answer…' : 'Type your answer…'}
              value={drafts[q.qid] || ''}
            />
          )}
        </div>
      ))}
      <div className="flex justify-end">
        <Button disabled={sending || !allAnswered || !member} onClick={() => void submit()} size="sm">
          {sending ? 'Sending…' : isApproval ? 'Respond' : 'Answer'}
        </Button>
      </div>
    </div>
  )
}
