import { type Extension, Prec } from '@codemirror/state'
import type { EditorView } from '@codemirror/view'
import { keymap, ViewPlugin } from '@codemirror/view'
import { useStore } from '@nanostores/react'
import { type KeyboardEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { type ApprovalChoice, approvalChoices } from '@/components/assistant-ui/tool/approval-choices'
import { sendApproval } from '@/components/assistant-ui/tool/approval-send'
import { type Translations, useI18n } from '@/i18n'
import { useKeybindHint } from '@/lib/keybinds/use-keybind-hint'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'

import { $backworkspaceSessionId, backworkspaceApprovalQueue } from './approvals'
import { caretRect } from './caret'

/** The action the keybind panel teaches, and the chord the page binds. */
export const APPROVAL_ACTION = 'backworkspace.approval'
const APPROVAL_KEY = 'Mod-Shift-a'
// How long a confirmation has to stand before it counts. Key auto-repeat and a
// double-click both land their second press inside this, and neither of them is
// a second decision.
const ARM_MS = 300

interface OpenCard {
  /** Opens upwards when the caret is low enough that it would otherwise hang off the page. */
  above: boolean
  left: number
  requestId: string
  /** The caret's own edge: the card's top when it opens down, its bottom when up. */
  top: number
}

// The card's own width, and the gap it keeps from the window edge, so a caret
// near the right-hand side does not push it off the page.
const CARD_WIDTH = 384
const EDGE_GUTTER = 16

function placeBeside(rect: { bottom: number; left: number; top: number }, requestId: string): OpenCard {
  const above = rect.bottom > window.innerHeight / 2

  return {
    above,
    left: Math.max(EDGE_GUTTER, Math.min(rect.left, window.innerWidth - CARD_WIDTH - EDGE_GUTTER)),
    requestId,
    top: above ? rect.top : rect.bottom
  }
}

function choiceLabel(choice: ApprovalChoice, copy: Translations['assistant']['approval']): string {
  const labels: Record<ApprovalChoice, string> = {
    always: copy.alwaysAllow,
    deny: copy.reject,
    once: copy.run,
    session: copy.allowSession
  }

  return labels[choice]
}

/**
 * The page's approval surface.
 *
 * Nothing appears while you write: an approval addressed to the agent the page
 * asked shows up as one line at the bottom, in the same grey that says the
 * agent is answering. A key opens a card beside the caret — where the `@` list
 * opens — and that card is where the command is read in full and answered.
 * Opening it is the reader's move, never the agent's: a panel that appeared
 * under the caret because something happened in the background would be the one
 * thing this page promised not to do.
 *
 * The card belongs to the ONE request it was opened for. If that request leaves
 * the queue — answered from the notification, withdrawn, timed out — the card
 * goes with it, rather than sliding the next command under a reader who is
 * already reaching for Enter.
 */
export function useApprovalCard(): { card: ReactNode; extension: Extension; notice: null | string } {
  const { t } = useI18n()
  const copy = t.assistant.approval
  const sessionId = useStore($backworkspaceSessionId)
  const requests = useStore(useMemo(() => backworkspaceApprovalQueue(sessionId), [sessionId]))
  const shortcut = useKeybindHint(APPROVAL_ACTION) ?? ''
  const [open, setOpen] = useState<null | OpenCard>(null)
  const [activeIndex, setActiveIndex] = useState(0)
  // "Always allow" writes a pattern to config.yaml for good, so the page asks
  // twice the way the chat's card does with its dialog — answering from the
  // keyboard should not make the widest promise on a single keystroke.
  const [armed, setArmed] = useState<null | { at: number; choice: ApprovalChoice }>(null)
  const viewRef = useRef<EditorView | null>(null)
  const cardRef = useRef<HTMLDivElement | null>(null)
  const busy = useRef(false)

  const waiting = requests[0] ?? null
  const shown = open ? (requests.find(item => item.requestId === open.requestId) ?? null) : null
  const choices = useMemo(() => (shown ? approvalChoices(shown) : []), [shown])
  const highlighted = Math.min(activeIndex, Math.max(0, choices.length - 1))

  const close = useCallback(() => {
    setOpen(null)
    setArmed(null)
    setActiveIndex(0)
    viewRef.current?.focus()
  }, [])

  // The request the card was opened for has gone. The card goes with it.
  useEffect(() => {
    if (open && !shown) {
      close()
    }
  }, [close, open, shown])

  // The card takes the keyboard while it is up, so it takes focus with it. It
  // is only ever up because the reader asked for it, which is what makes that
  // fair — and it is what keeps Enter out of the prose behind it.
  useEffect(() => {
    if (open) {
      cardRef.current?.focus({ preventScroll: true })
    }
  }, [open])

  const answer = useCallback(
    (choice: ApprovalChoice, repeated = false) => {
      if (!shown || busy.current || repeated) {
        return
      }

      if (choice === 'always' && (armed?.choice !== 'always' || Date.now() - armed.at < ARM_MS)) {
        setArmed({ at: Date.now(), choice })

        return
      }

      busy.current = true
      close()

      void sendApproval(shown, choice)
        .catch(error => notifyError(error, copy.sendFailed))
        .finally(() => {
          busy.current = false
        })
    },
    [armed, close, copy, shown]
  )

  const onCardKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault()
        setArmed(null)
        setActiveIndex(current =>
          choices.length === 0 ? 0 : (current + (event.key === 'ArrowDown' ? 1 : -1) + choices.length) % choices.length
        )

        return
      }

      if (event.key === 'Escape') {
        // Taken here, so the page's own Escape does not also turn the window
        // back: one cancel gesture does one thing.
        event.preventDefault()
        close()

        return
      }

      if (event.key === 'Enter') {
        event.preventDefault()
        answer(choices[highlighted], event.repeat)
      }
    },
    [answer, choices, close, highlighted]
  )

  // Only the chord belongs to the editor. Everything else is the card's, which
  // has the keyboard for as long as it is up.
  const extension = useMemo(
    () => [
      ViewPlugin.define(view => {
        viewRef.current = view

        return {
          // The page went away under the card — a profile switch, the window
          // turned back. The card goes with the writing it belonged to.
          destroy() {
            viewRef.current = null
            setOpen(null)
            setArmed(null)
          }
        }
      }),
      Prec.highest(
        keymap.of([
          {
            key: APPROVAL_KEY,
            run: view => {
              const next = backworkspaceApprovalQueue($backworkspaceSessionId.get()).get()[0]

              if (!next?.requestId) {
                return false
              }

              // Read straight out: a key handler runs on the event, not inside
              // an update, so the layout is there to be measured. (The `@` list
              // has to ask for a measurement because it tracks the caret from
              // inside the view's own update.)
              const rect = caretRect(view, view.state.selection.main.head)

              setActiveIndex(0)
              setArmed(null)
              setOpen(placeBeside(rect, next.requestId))

              return true
            }
          }
        ])
      )
    ],
    []
  )

  const card =
    open && shown ? (
      // The caret's own coordinates, like the `@` list: the question opens
      // where the writing is.
      <div
        className="absolute"
        style={
          open.above ? { bottom: window.innerHeight - open.top, left: open.left } : { left: open.left, top: open.top }
        }
      >
        <div
          aria-label={copy.jumpToApproval}
          className={cn(
            'w-96 max-w-[calc(100vw-2rem)] rounded-xl border p-3 outline-none',
            open.above ? 'mb-1' : 'mt-1',
            'border-(--stroke-nous) bg-(--ui-chat-surface-background) shadow-nous'
          )}
          data-backworkspace-approval=""
          data-glass-opaque=""
          onBlur={event => {
            if (!event.currentTarget.contains(event.relatedTarget)) {
              setOpen(null)
              setArmed(null)
            }
          }}
          onKeyDown={onCardKeyDown}
          ref={cardRef}
          role="menu"
          tabIndex={-1}
        >
          <p className="text-xs text-(--ui-text-secondary)">{shown.description}</p>
          {shown.command.trim() !== '' && (
            // The whole command, wrapped and scrollable: what is being agreed
            // to is the one thing this surface may not abbreviate.
            <pre className="mt-2 max-h-40 overflow-auto rounded-md bg-(--ui-bg-tertiary) p-2 font-mono text-xs break-words whitespace-pre-wrap">
              {shown.command}
            </pre>
          )}
          <ul className="mt-2">
            {choices.map((choice, index) => (
              <li key={choice}>
                <button
                  className={cn(
                    'flex w-full cursor-default items-center rounded-md px-2 py-1 text-left text-sm transition-colors',
                    index === highlighted ? 'bg-(--ui-bg-tertiary)' : 'hover:bg-(--ui-bg-tertiary)'
                  )}
                  data-approval-choice={choice}
                  onClick={() => answer(choice)}
                  onMouseEnter={() => {
                    setArmed(null)
                    setActiveIndex(index)
                  }}
                  role="menuitem"
                  tabIndex={-1}
                  type="button"
                >
                  {armed?.choice === choice ? t.backworkspace.approvalConfirm : choiceLabel(choice, copy)}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    ) : null

  const notice =
    !waiting || open
      ? null
      : requests.length > 1
        ? t.backworkspace.approvalWaitingMany(requests.length, shortcut)
        : t.backworkspace.approvalWaiting(shortcut)

  return { card, extension, notice }
}
