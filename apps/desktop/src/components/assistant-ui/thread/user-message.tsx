import { ActionBarPrimitive, BranchPickerPrimitive, MessagePrimitive, useAuiState } from '@assistant-ui/react'
import { type FC, type ReactNode, useCallback, useEffect, useRef, useState } from 'react'

import { DirectiveContent } from '@/components/assistant-ui/directive-text'
import { AgentExchangeCard } from '@/components/assistant-ui/thread/agent-exchange-card'
import { messageAttachmentRefs, messageContentText } from '@/components/assistant-ui/thread/content'
import { ReactionBadge, ReactionPicker } from '@/components/assistant-ui/thread/message-reactions'
import { MessageTimelineTimestamp } from '@/components/assistant-ui/thread/timeline-timestamp'
import { type RestoreMessageTarget } from '@/components/assistant-ui/thread/types'
import { useMessageReactions } from '@/components/assistant-ui/thread/use-message-reactions'
import { UserMessageText } from '@/components/assistant-ui/thread/user-message-text'
import { Codicon } from '@/components/ui/codicon'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tip } from '@/components/ui/tooltip'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Clipboard, StopFilled } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $gateway } from '@/store/gateway'
import { notifyThreadEditOpen } from '@/store/thread-scroll'
import { isWatchWindow } from '@/store/windows'

/** True when the user has a live text highlight (drag-select / triple-click). */
export function hasTextSelection(): boolean {
  const selection = window.getSelection()

  return Boolean(selection && !selection.isCollapsed && selection.toString().length > 0)
}

export function StickyHumanMessageContainer({
  attachments,
  children,
  messageId
}: {
  attachments?: ReactNode
  children: ReactNode
  messageId?: string
}) {
  return (
    // Fragment, not a wrapper: a wrapping element becomes the sticky's
    // containing block (it'd stick within its own height = never). The bubble
    // and attachments are flow siblings so the bubble pins against the scroller
    // while attachments below it scroll away.
    <>
      <div
        className="group/user-message sticky z-40 -mx-4 flex w-[calc(100%+2rem)] min-w-0 max-w-none flex-col items-stretch gap-0 self-end overflow-visible bg-(--ui-chat-surface-background) px-4 pb-(--conversation-turn-gap) pt-1"
        data-message-id={messageId}
        data-role="user"
        data-slot="aui_user-message-root"
      >
        {children}
      </div>
      {attachments}
    </>
  )
}

export function MessageAttachmentIndicator({
  attachmentRefs,
  label
}: {
  attachmentRefs: string[]
  label: string
}) {
  if (attachmentRefs.length === 0) {
    return null
  }

  return (
    <div className="-mt-3 mb-2 flex w-full justify-end px-1" data-slot="aui_user-message-attachments">
      <Popover>
        <Tip label={label}>
          <PopoverTrigger asChild>
            <button
              aria-label={label}
              className="relative grid size-7 place-items-center rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary) text-(--ui-text-secondary) transition-colors hover:border-(--ui-stroke-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--ui-accent)"
              data-slot="aui_user-message-attachment-trigger"
              type="button"
            >
              <Clipboard className="size-3.5" />
              <span className="absolute -right-1.5 -top-1.5 grid h-4 min-w-4 place-items-center rounded-full border border-(--ui-chat-surface-background) bg-(--ui-accent,#6e9fc5) px-1 text-[0.58rem] font-semibold leading-none text-white shadow-xs">
                {attachmentRefs.length}
              </span>
            </button>
          </PopoverTrigger>
        </Tip>
        <PopoverContent
          align="end"
          className="w-72 p-2"
          data-slot="aui_user-message-attachment-popover"
          side="top"
        >
          <div className="max-h-72 overflow-y-auto">
            <DirectiveContent text={attachmentRefs.join(' ')} />
          </div>
        </PopoverContent>
      </Popover>
    </div>
  )
}

// Shared "user bubble" base. Both the read-only message and the inline
// edit composer render the same bubble surface (rounded glass card);
// they only differ in border weight, cursor, and padding-right (the
// read-only view reserves room for the restore icon).
//
// no-drag: sticky bubbles park at --sticky-human-top (~4px), sliding under the
// titlebar's [-webkit-app-region:drag] strips (app-shell.tsx). Electron resolves
// drag regions at the compositor level — z-index and pointer-events don't help —
// so without the carve-out, clicking a stuck bubble drags the window instead of
// opening the edit composer.
export const USER_BUBBLE_BASE_CLASS =
  'composer-human-message standalone-glass relative flex w-full min-w-0 max-w-full flex-col gap-1.5 overflow-y-auto rounded-xl border border-r-4 border-r-(--ui-accent,#6e9fc5) bg-(--dt-user-bubble) px-3 py-2 text-left transition-shadow hover:shadow-[0_0_20px_color-mix(in_srgb,var(--ui-accent,#6e9fc5)_22%,transparent)] [-webkit-app-region:no-drag]'

const USER_MESSAGE_CARD_STYLE = {
  boxShadow: '0 0 10px color-mix(in srgb, var(--ui-accent,#6e9fc5) 9%, transparent)'
} as const

export const USER_ACTION_ICON_BUTTON_CLASS =
  'grid place-items-center rounded-md bg-transparent text-(--ui-text-secondary) transition-colors hover:bg-(--ui-control-active-background) hover:text-foreground disabled:cursor-default disabled:text-(--ui-text-quaternary) disabled:opacity-70'

export const USER_ACTION_ICON_SIZE = '0.6875rem'
export const StopGlyph = <StopFilled aria-hidden className="size-3.5 -translate-y-px" />

// Background-process notifications are injected into the conversation as user
// messages (the agent must react to them, and message-role alternation forbids
// a synthetic system row mid-loop). They are NOT something the human typed, so
// render them as a compact system-style notice instead of a user bubble.
// Shape: see tools/process_registry.py format_process_notification().
const PROCESS_NOTIFICATION_RE = /^\[IMPORTANT: Background process [\s\S]*\]$/

// Agent-to-agent deliveries ("Message from 🤖 <sender>: …", the Bot Mode /
// multi-profile convention; optional "(@<handle>)" carries the sender's
// profile name for avatar resolution; legacy "[Message from agent
// '<sender>'] …" too). They arrive on the user role because the recipient's
// turn runs on it, but they are NOT the human speaking — render them as a
// compact attributed timeline notice instead of a user bubble.
export const AGENT_MESSAGE_RE =
  /^(?:Message from (?:🤖\s*)?([^:\n(]{1,64}?)(?:\s*\(@([a-z0-9][a-z0-9_-]{0,63})\))?:\s*|\[Message from agent '([^']{1,64})'\]\s*)([\s\S]*)$/u

// sender handle -> avatar data URL. Module-level so a chat full of notices
// from one bot resolves once. Hits are cached for the window's lifetime;
// misses only briefly (30s) — an avatar can appear at any moment (bot just
// created, art backfill still running), and a permanent negative cache
// froze the 🤖 glyph until an app restart.
export const agentAvatarCache = new Map<string, null | string>()
const agentAvatarMissAt = new Map<string, number>()
const AVATAR_MISS_TTL_MS = 30_000
const agentAvatarInflight = new Map<string, Promise<null | string>>()

export async function resolveAgentAvatar(handle: string): Promise<null | string> {
  const key = handle.trim().toLowerCase()

  if (!key) {
    return null
  }

  if (agentAvatarCache.has(key)) {
    const hit = agentAvatarCache.get(key) ?? null

    if (hit !== null) {
      return hit
    }

    // Negative entry: honor it only within the TTL, then re-probe.
    if (Date.now() - (agentAvatarMissAt.get(key) ?? 0) < AVATAR_MISS_TTL_MS) {
      return null
    }

    agentAvatarCache.delete(key)
  }

  const inflight = agentAvatarInflight.get(key)

  if (inflight) {
    return inflight
  }

  const run = (async (): Promise<null | string> => {
    try {
      const gateway = $gateway.get()

      if (!gateway) {
        return null
      }

      const res = await gateway.request<{ profiles?: Array<{ has_avatar?: boolean; name: string }> }>('profiles.list', {
        include_sessions: false
      })

      const profiles = res?.profiles ?? []
      let profile = profiles.find(p => p.name.toLowerCase() === key)

      // 'hermes' is the conventional alias for the primary profile.
      if (!profile && key === 'hermes') {
        profile = profiles.find(p => p.name === 'default')
      }

      if (!profile?.has_avatar) {
        return null
      }

      const asset = await gateway.request<{ data?: string; found?: boolean }>('profiles.get_asset', {
        asset: 'avatar',
        name: profile.name
      })

      return asset?.found && asset.data ? asset.data : null
    } catch {
      // Older gateway (no profiles.* RPCs) or transient failure — the 🤖
      // glyph fallback is always correct.
      return null
    } finally {
      agentAvatarInflight.delete(key)
    }
  })()

  agentAvatarInflight.set(key, run)
  const out = await run
  agentAvatarCache.set(key, out)

  if (out === null) {
    agentAvatarMissAt.set(key, Date.now())
  }

  return out
}

const AgentMessageNote: FC<{ text: string }> = ({ text }) => {
  const match = AGENT_MESSAGE_RE.exec(text)
  const sender = (match?.[1] || match?.[3] || 'agent').trim()
  const handle = (match?.[2] || match?.[3] || sender).trim()
  const body = (match?.[4] || '').trim()
  const [avatar, setAvatar] = useState<null | string>(() => agentAvatarCache.get(handle.toLowerCase()) ?? null)

  useEffect(() => {
    let live = true

    void resolveAgentAvatar(handle).then(url => {
      if (live && url) {
        setAvatar(url)
      }
    })

    return () => {
      live = false
    }
  }, [handle])

  // Keep transport framing out of the transcript. The handoff identity and a
  // one-line preview remain visible; the exact delivered text stays one click
  // away in the same cohesive card.
  return (
    <AgentExchangeCard
      agent={sender}
      avatar={
        avatar ? (
          <img alt="" aria-hidden className="size-full object-cover" src={avatar} />
        ) : (
          <span aria-hidden className="text-[0.875rem] leading-none">
            🤖
          </span>
        )
      }
      body={body ? <UserMessageText text={body} /> : undefined}
      bodyText={body}
      kind="handoff"
      replyProfile={handle}
      slot="aui_agent-message-note"
    />
  )
}

export const ProcessNotificationNote: FC<{ text: string }> = ({ text }) => {
  const body = text.replace(/^\[IMPORTANT:\s*/, '').replace(/\]$/, '')
  const newline = body.indexOf('\n')
  const headline = (newline === -1 ? body : body.slice(0, newline)).trim()
  const detail = newline === -1 ? '' : body.slice(newline + 1).trim()

  const summary = /completed normally|exit code 0/i.test(headline) ? 'Agent work completed' : 'Background agent update'

  return (
    <article className="w-full min-w-0 overflow-hidden rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-widget-surface-background) text-[0.75rem] text-foreground/90 shadow-sm">
      {detail ? (
        <details className="group/process">
          <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 transition-colors hover:bg-(--ui-control-hover-background) [&::-webkit-details-marker]:hidden">
            <Codicon className="shrink-0 text-(--ui-accent,#6e9fc5)" name="terminal" size="0.875rem" />
            <span className="min-w-0 flex-1">
              <span className="block font-semibold">{summary}</span>
              <span className="wrap-anywhere block truncate text-[0.6875rem] text-(--ui-text-tertiary)">{headline}</span>
            </span>
            <span className="shrink-0 text-[0.6875rem] font-medium text-(--ui-text-secondary)">
              <span className="group-open/process:hidden">Show details</span>
              <span className="hidden group-open/process:inline">Hide details</span>
            </span>
            <Codicon
              className="shrink-0 text-(--ui-text-tertiary) transition-transform duration-150 group-open/process:rotate-90"
              name="chevron-right"
              size="0.75rem"
            />
          </summary>
          <pre
            className="max-h-44 overflow-auto whitespace-pre-wrap border-t border-(--ui-stroke-tertiary) bg-(--ui-surface-background) px-3 py-2.5 font-mono text-[0.6875rem] leading-5 text-foreground/75"
            data-selectable-text="true"
          >
            {detail}
          </pre>
        </details>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2">
          <Codicon className="shrink-0 text-(--ui-accent,#6e9fc5)" name="terminal" size="0.875rem" />
          <span className="font-semibold">{summary}</span>
          <span className="wrap-anywhere min-w-0 truncate text-[0.6875rem] text-(--ui-text-tertiary)">{headline}</span>
        </div>
      )}
    </article>
  )
}

export const UserMessage: FC<{
  onCancel?: () => Promise<void> | void
  onRequestRestoreConfirm?: (messageId: string, target: RestoreMessageTarget) => void
}> = ({ onCancel, onRequestRestoreConfirm }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const messageId = useAuiState(s => s.message.id)
  const content = useAuiState(s => s.message.content)
  const messageText = messageContentText(content)
  const threadRunning = useAuiState(s => s.thread.isRunning)

  const latestUserId = useAuiState(s => {
    for (let i = s.thread.messages.length - 1; i >= 0; i--) {
      const message = s.thread.messages[i] as { id?: string; role?: string }

      if (message.role === 'user') {
        return message.id ?? null
      }
    }

    return null
  })

  const runtimeUserOrdinal = useAuiState(s => {
    let ordinal = 0

    for (const message of s.thread.messages) {
      if (message.role !== 'user') {
        continue
      }

      if (message.id === s.message.id) {
        return ordinal
      }

      ordinal += 1
    }

    return null
  })

  const attachmentRefs = useAuiState(s => {
    const custom = (s.message.metadata?.custom ?? {}) as { attachmentRefs?: unknown }

    return messageAttachmentRefs(custom.attachmentRefs)
  })

  const [pickerOpen, setPickerOpen] = useState(false)
  const { enabled: reactionsEnabled, react, reactions: shownReactions } = useMessageReactions(messageId, 'user')

  const pickEmoji = useCallback(
    (emoji: null | string) => {
      setPickerOpen(false)
      react(emoji)
    },
    [react]
  )

  // Sticky human bubbles clamp to ~2 lines with a soft fade so a long prompt
  // doesn't dominate the viewport while the response streams underneath; the
  // clamp lifts on hover / focus (see styles.css). We measure the *unclamped*
  // inner wrapper so the ResizeObserver only fires on real content / width
  // changes, not on every frame while the outer max-height animates open.
  const clampInnerRef = useRef<HTMLDivElement | null>(null)
  const [bodyClamped, setBodyClamped] = useState(false)
  const lastClampHeightRef = useRef(-1)
  const lineHeightRef = useRef(0)

  // Watch windows spectate a subagent run driven elsewhere — prompts can't be
  // edited, restored, or stopped from here. The bubble stays a button that
  // toggles the 2-line clamp so long prompts are still fully readable.
  const readOnly = isWatchWindow()
  const [expanded, setExpanded] = useState(false)
  const clampActive = !(readOnly && expanded)

  const measureClamp = useCallback((entries: readonly ResizeObserverEntry[]) => {
    const inner = clampInnerRef.current
    const outer = inner?.parentElement

    if (!inner || !outer) {
      return
    }

    // Prefer the size the ResizeObserver already computed — reading
    // `scrollHeight` outside RO timing forces a synchronous layout, and with
    // many user bubbles observed at once those reads interleave with the
    // style write below into a read-write-read reflow cascade.
    const entryHeight = entries.find(entry => entry.target === inner)?.borderBoxSize?.[0]?.blockSize
    const fullHeight = Math.ceil(entryHeight ?? inner.scrollHeight)

    if (fullHeight === lastClampHeightRef.current) {
      return
    }

    lastClampHeightRef.current = fullHeight

    // Line-height is stable for the life of the bubble (font settings don't
    // change under it) — resolve the computed style once.
    if (!lineHeightRef.current) {
      const styles = getComputedStyle(inner)
      lineHeightRef.current = parseFloat(styles.lineHeight) || 1.5 * parseFloat(styles.fontSize) || 20
    }

    outer.style.setProperty('--human-msg-full', `${fullHeight}px`)
    setBodyClamped(fullHeight > lineHeightRef.current * 2 + 1)
  }, [])

  useResizeObserver(measureClamp, clampInnerRef)

  // Injected background-process notification, not a human prompt — render the
  // compact system-style notice (after all hooks above have run).
  if (PROCESS_NOTIFICATION_RE.test(messageText.trim())) {
    return (
      <MessagePrimitive.Root
        className="flex w-full min-w-0 flex-col items-stretch"
        data-role="user"
        data-slot="aui_user-message-root"
      >
        <ProcessNotificationNote text={messageText.trim()} />
        <MessageTimelineTimestamp className="self-center" />
      </MessagePrimitive.Root>
    )
  }

  // Agent-to-agent delivery, not a human prompt — attributed inter-agent card.
  if (AGENT_MESSAGE_RE.test(messageText.trim())) {
    return (
      <MessagePrimitive.Root
        className="flex w-full min-w-0 flex-col items-stretch pb-(--conversation-turn-gap)"
        data-role="user"
        data-slot="aui_user-message-root"
      >
        <AgentMessageNote text={messageText.trim()} />
      </MessagePrimitive.Root>
    )
  }

  const hasBody = messageText.trim().length > 0
  const isLatestUser = messageId === latestUserId
  const showStop = !readOnly && isLatestUser && threadRunning && Boolean(onCancel)
  // Restore (re-run this exact prompt) is available everywhere the Stop button
  // isn't — including mid-stream on older prompts, since the action interrupts
  // the live turn before rewinding.
  const showRestore = !readOnly && !showStop && Boolean(onRequestRestoreConfirm) && hasBody

  const bubbleClassName = cn(
    USER_BUBBLE_BASE_CLASS,
    'cursor-pointer pr-9 text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground/95 transition-colors',
    'border-(--ui-stroke-tertiary) hover:border-(--ui-stroke-secondary)'
  )

  const bubbleContent = hasBody && (
    // Render the user's text through a minimal markdown pipeline:
    // backtick `code` and ``` fenced ``` blocks, with directive chips
    // (`@file:` etc.) still resolved inside the plain-text spans.
    <div
      className={cn(clampActive && 'sticky-human-clamp')}
      data-clamped={clampActive && bodyClamped ? 'true' : undefined}
    >
      {/* Match the edit composer's collapsed line box (min-h-[1.25rem]) so
          clicking to edit can't grow the bubble by a sub-pixel and reflow the
          turn 1px. */}
      <div className="min-h-[1.25rem]" ref={clampInnerRef}>
        <UserMessageText className="wrap-anywhere" text={messageText} />
      </div>
    </div>
  )

  return (
    <MessagePrimitive.Root asChild>
      <StickyHumanMessageContainer
        attachments={
          attachmentRefs.length > 0 ? (
            <MessageAttachmentIndicator
              attachmentRefs={attachmentRefs}
              label={t.composer.attachments(attachmentRefs.length)}
            />
          ) : null
        }
        messageId={messageId}
      >
        <ActionBarPrimitive.Root className="relative w-full max-w-full" data-slot="aui_user-bubble-actions">
          <div className="human-message-with-todos-wrapper flex w-full flex-col gap-0">
            <ReactionPicker
              onOpenChange={setPickerOpen}
              onSelect={pickEmoji}
              open={pickerOpen}
              selected={shownReactions.find(reaction => reaction.author === 'user')?.emoji}
            >
              <div
                className="relative w-full"
                // The app context menu skips PLAIN right-clicks here (the
                // attr below) so this handler keeps the picker gesture; a
                // link/image/selection inside the bubble still gets the app
                // menu, and this handler's selection guard keeps ⌘C flows.
                data-context-menu-skip=""
                onContextMenu={
                  // Right-click is the desktop stand-in for iOS touch-and-hold —
                  // but only when there's nothing selected. A live highlight
                  // keeps the native Copy menu (and ⌘C) instead of the picker.
                  readOnly || !reactionsEnabled
                    ? undefined
                    : event => {
                        if (hasTextSelection()) {
                          return
                        }

                        event.preventDefault()
                        setPickerOpen(true)
                      }
                }
              >
                {readOnly ? (
                  // Spectator transcript: clicking only toggles the clamp so the
                  // full prompt is readable — never opens an edit composer.
                  <button
                    aria-expanded={bodyClamped ? expanded : undefined}
                    className={cn(bubbleClassName, !bodyClamped && 'cursor-default')}
                    onClick={() => {
                      // Drag-select ends on mouseup→click; don't collapse the
                      // clamp just because the highlight finished.
                      if (hasTextSelection() || !bodyClamped) {
                        return
                      }

                      triggerHaptic('selection')
                      setExpanded(value => !value)
                    }}
                    style={USER_MESSAGE_CARD_STYLE}
                    title={bodyClamped ? (expanded ? t.common.collapse : copy.expandMessage) : undefined}
                    type="button"
                  >
                    {bubbleContent}
                  </button>
                ) : (
                  // Always editable — clicking opens the edit composer even while a
                  // turn streams; sending the edit reverts (interrupt + rewind).
                  // A live text highlight wins: finishing a drag-select must not
                  // open the editor and throw the selection away.
                  <ActionBarPrimitive.Edit asChild>
                    <button
                      aria-label={copy.editMessage}
                      className={bubbleClassName}
                      onClick={event => {
                        if (hasTextSelection()) {
                          event.preventDefault()
                          event.stopPropagation()

                          return
                        }

                        triggerHaptic('selection')
                      }}
                      onPointerDown={() => {
                        if (hasTextSelection()) {
                          return
                        }

                        notifyThreadEditOpen()
                      }}
                      style={USER_MESSAGE_CARD_STYLE}
                      type="button"
                    >
                      {bubbleContent}
                    </button>
                  </ActionBarPrimitive.Edit>
                )}
                {(showStop || showRestore) && (
                  <div className="pointer-events-none absolute right-2 bottom-2 z-10 flex items-center justify-center opacity-0 transition-opacity group-hover/user-message:opacity-100 group-focus-within/user-message:opacity-100">
                    {showStop ? (
                      <button
                        aria-label={copy.stop}
                        className={cn('pointer-events-auto size-5', USER_ACTION_ICON_BUTTON_CLASS)}
                        onClick={event => {
                          event.preventDefault()
                          event.stopPropagation()
                          void onCancel?.()
                        }}
                        title={copy.stop}
                        type="button"
                      >
                        {StopGlyph}
                      </button>
                    ) : (
                      <button
                        aria-label={copy.restoreCheckpoint}
                        className={cn('pointer-events-auto size-6', USER_ACTION_ICON_BUTTON_CLASS)}
                        onClick={event => {
                          event.preventDefault()
                          event.stopPropagation()
                          triggerHaptic('selection')
                          onRequestRestoreConfirm?.(messageId, {
                            text: messageText,
                            userOrdinal: runtimeUserOrdinal
                          })
                        }}
                        onPointerDown={event => {
                          event.preventDefault()
                          event.stopPropagation()
                        }}
                        title={copy.restoreFromHere}
                        type="button"
                      >
                        <Codicon name="discard" size="0.875rem" />
                      </button>
                    )}
                  </div>
                )}
              </div>
            </ReactionPicker>
            {/* Below the bubble, same register as the assistant action row:
                same emoji size, same vertical padding, right-aligned to the
                sent bubble. Overlaying the corner read badly in practice. */}
            <ReactionBadge
              className="justify-end gap-1.5 py-1.5 pr-1.5"
              onRetract={() => react(null)}
              reactions={shownReactions}
            />
            <MessageTimelineTimestamp className="self-end pr-1.5" />
            <BranchPickerPrimitive.Root
              className={cn(
                'checkpoint-container flex items-center gap-1 pb-0 pt-1 pl-1.5 text-[0.75rem] leading-none text-(--ui-text-tertiary)',
                readOnly && 'hidden'
              )}
              hideWhenSingleBranch
            >
              <span aria-hidden className="checkpoint-icon size-1.5 rounded-full border border-current" />
              <BranchPickerPrimitive.Previous
                className="checkpoint-restore-text rounded-sm bg-transparent px-1 opacity-65 hover:opacity-100 disabled:hidden disabled:cursor-default"
                title={copy.restorePrevious}
              >
                {copy.restoreCheckpoint}
              </BranchPickerPrimitive.Previous>
              <span className="checkpoint-divider opacity-55">
                <BranchPickerPrimitive.Number />/<BranchPickerPrimitive.Count />
              </span>
              <BranchPickerPrimitive.Next
                className="checkpoint-restore-text rounded-sm bg-transparent px-1 opacity-65 hover:opacity-100 disabled:hidden disabled:cursor-default"
                title={copy.restoreNext}
              >
                {copy.goForward}
              </BranchPickerPrimitive.Next>
            </BranchPickerPrimitive.Root>
          </div>
        </ActionBarPrimitive.Root>
      </StickyHumanMessageContainer>
    </MessagePrimitive.Root>
  )
}
