import { ActionBarPrimitive, BranchPickerPrimitive, MessagePrimitive, useAuiState } from '@assistant-ui/react'
import { type FC, type ReactNode, useCallback, useRef, useState } from 'react'

import { DirectiveContent } from '@/components/assistant-ui/directive-text'
import { messageAttachmentRefs, messageContentText } from '@/components/assistant-ui/thread/content'
import { type RestoreMessageTarget } from '@/components/assistant-ui/thread/types'
import { UserMessageText } from '@/components/assistant-ui/thread/user-message-text'
import { Codicon } from '@/components/ui/codicon'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { StopFilled } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyThreadEditOpen } from '@/store/thread-scroll'
import { isWatchWindow } from '@/store/windows'

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
        className="group/user-message sticky z-40 -mx-4 flex w-[calc(100%+2rem)] min-w-0 max-w-none flex-col items-end gap-0 self-end overflow-visible bg-(--ui-chat-surface-background) px-4 pb-(--conversation-turn-gap) pt-1"
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
  'composer-human-message standalone-glass relative flex w-full min-w-0 max-w-full flex-col gap-1.5 overflow-y-auto rounded-xl border bg-(--dt-user-bubble) px-3 py-2 text-left [-webkit-app-region:no-drag]'

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

const ProcessNotificationNote: FC<{ text: string }> = ({ text }) => {
  const body = text.replace(/^\[IMPORTANT:\s*/, '').replace(/\]$/, '')
  const newline = body.indexOf('\n')
  const headline = (newline === -1 ? body : body.slice(0, newline)).trim()
  const detail = newline === -1 ? '' : body.slice(newline + 1).trim()

  return (
    <div className="flex max-w-[min(86%,44rem)] flex-col gap-0.5 self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/60">
      <span className="flex items-center gap-1.5">
        <Codicon className="shrink-0 text-muted-foreground/55" name="terminal" size="0.75rem" />
        <span className="wrap-anywhere">{headline}</span>
      </span>
      {detail && (
        <details className="pl-[1.3125rem]">
          <summary className="cursor-pointer select-none text-muted-foreground/45 hover:text-muted-foreground/70">
            output
          </summary>
          <pre
            className="mt-0.5 max-h-48 overflow-auto whitespace-pre-wrap font-mono text-[0.625rem] leading-4 text-muted-foreground/55"
            data-selectable-text="true"
          >
            {detail}
          </pre>
        </details>
      )}
    </div>
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
    'border-(--dt-user-bubble-border) hover:border-(--theme-primary)'
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
          // Attachments live BELOW the sticky bubble in normal flow, so they
          // scroll away behind the pinned bubble instead of riding along with
          // it. Image refs render as thumbnails, file refs as chips; no border.
          attachmentRefs.length > 0 ? (
            <div className="flex flex-wrap gap-1 -mt-3 mb-2">
              <DirectiveContent text={attachmentRefs.join(' ')} />
            </div>
          ) : null
        }
        messageId={messageId}
      >
        <ActionBarPrimitive.Root className="relative w-full max-w-[min(76%,42rem)]" data-slot="aui_user-bubble-actions">
          <div className="human-message-with-todos-wrapper flex w-full flex-col gap-0">
            <div className="relative w-full">
              {readOnly ? (
                // Spectator transcript: clicking only toggles the clamp so the
                // full prompt is readable — never opens an edit composer.
                <button
                  aria-expanded={bodyClamped ? expanded : undefined}
                  className={cn(bubbleClassName, !bodyClamped && 'cursor-default')}
                  onClick={() => {
                    if (!bodyClamped) {
                      return
                    }

                    triggerHaptic('selection')
                    setExpanded(value => !value)
                  }}
                  title={bodyClamped ? (expanded ? t.common.collapse : copy.expandMessage) : undefined}
                  type="button"
                >
                  {bubbleContent}
                </button>
              ) : (
                // Always editable — clicking opens the edit composer even while a
                // turn streams; sending the edit reverts (interrupt + rewind).
                <ActionBarPrimitive.Edit asChild>
                  <button
                    aria-label={copy.editMessage}
                    className={bubbleClassName}
                    onClick={() => triggerHaptic('selection')}
                    onPointerDown={() => notifyThreadEditOpen()}
                    title={copy.editMessage}
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
