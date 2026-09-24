import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n/context'
import { isSubmitEnter } from '@/lib/ime'
import {
  initialQuickComposerState,
  QUICK_TARGET_CURRENT,
  QUICK_TARGET_NEW,
  type QuickComposerEvent,
  quickComposerReducer,
  type QuickEntrySubmitPayload
} from '@/store/quick-entry'

import type { ThoughtSnapshot } from '../../../electron/thought-capture'

/** Capture stays local; only Enter sends through the existing chat path. */
export function QuickEntryApp() {
  const { t } = useI18n()
  const copy = t.quickCapture
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const [state, setState] = useState(initialQuickComposerState)
  const stateRef = useRef(state)
  const [inbox, setInbox] = useState<ThoughtSnapshot | null>(null)
  const inboxRef = useRef(inbox)
  const [expanded, setExpanded] = useState(false)
  const [busy, setBusy] = useState(false)
  const busyRef = useRef(false)

  const [notice, setNotice] = useState<
    'saved' | 'loadFailed' | 'saveFailed' | 'handoffUnconfirmed' | 'handoffRejected' | null
  >(null)

  const generation = useRef(0)
  const draftVersion = useRef(0)
  const pendingDrafts = useRef(new Map<string, ThoughtSnapshot['draft']>())
  const ownerKey = (value: ThoughtSnapshot) => JSON.stringify(value.owner)

  function dispatch(event: QuickComposerEvent) {
    const { state: next, send } = quickComposerReducer(stateRef.current, event)

    if (send) {
      void handoff(send)

      return
    }

    stateRef.current = next
    setState(next)

    if (!next.visible) {
      window.hermesDesktop.quickEntry.dismiss()
    }
  }

  async function handoff(send: QuickEntrySubmitPayload) {
    const current = inboxRef.current

    if (busyRef.current || current?.draft.handoffAttempted) {
      return
    }

    busyRef.current = true
    setBusy(true)
    const request = generation.current

    try {
      if (current) {
        const draft = { ...current.draft, text: stateRef.current.draft, handoffAttempted: true }
        // Persist uncertainty before dispatch so a restart cannot silently retry a chat prompt.
        await window.hermesDesktop.quickEntry.saveThoughtDraft({ token: current.token, draft })

        if (request !== generation.current) {return}
        pendingDrafts.current.delete(ownerKey(current))
        adopt({ ...current, draft })
      }

      setNotice('handoffUnconfirmed')

      const forwarded = await window.hermesDesktop.quickEntry.submit({
        ...send,
        thoughtOwnerToken: current?.token
      })

      if (request === generation.current && !forwarded) {setNotice('handoffRejected')}
    } catch {
      if (request === generation.current)
        {setNotice(inboxRef.current?.draft.handoffAttempted ? 'handoffUnconfirmed' : 'saveFailed')}
    } finally {
      busyRef.current = false
      setBusy(false)
    }
  }

  function adopt(value: ThoughtSnapshot) {
    inboxRef.current = value
    setInbox(value)
    dispatch({ type: 'edit', draft: value.draft.text })
  }

  async function load() {
    const request = ++generation.current
    const version = draftVersion.current

    try {
      const value = await window.hermesDesktop.quickEntry.readThoughts()

      if (request !== generation.current || version !== draftVersion.current) {
        return
      }

      const pending = pendingDrafts.current.get(ownerKey(value))
      adopt(pending ? { ...value, draft: pending } : value)
      setNotice(pending ? 'saveFailed' : value.draft.handoffAttempted ? 'handoffUnconfirmed' : null)
    } catch {
      if (request === generation.current) {
        setNotice('loadFailed')
      }
    }
  }

  function edit(text: string) {
    dispatch({ type: 'edit', draft: text })
    const version = ++draftVersion.current

    if (inboxRef.current) {
      setNotice(null)
    }

    const current = inboxRef.current

    if (!current) {
      return
    }

    const draft = { id: current.draft.id, text }
    inboxRef.current = { ...current, draft }
    setInbox(inboxRef.current)
    const key = ownerKey(current)
    pendingDrafts.current.set(key, draft)
    void window.hermesDesktop.quickEntry
      .saveThoughtDraft({ token: current.token, draft })
      .then(() => {
        if (pendingDrafts.current.get(key) === draft) {
          pendingDrafts.current.delete(key)
        }
      })
      .catch(() => {
        if (draftVersion.current === version && inboxRef.current?.token === current.token) {
          setNotice('saveFailed')
        }
      })
  }

  async function save() {
    const current = inboxRef.current

    if (!current || busyRef.current || !stateRef.current.draft.trim()) {
      return
    }

    busyRef.current = true
    setBusy(true)
    const request = generation.current

    try {
      const value = await window.hermesDesktop.quickEntry.saveThought({
        token: current.token,
        draft: { ...current.draft, text: stateRef.current.draft }
      })

      pendingDrafts.current.delete(ownerKey(current))

      if (request !== generation.current) {
        return
      }

      adopt(value)
      setNotice('saved')
    } catch {
      if (request === generation.current) {
        setNotice('saveFailed')
      }
    } finally {
      busyRef.current = false
      setBusy(false)

      if (request === generation.current && stateRef.current.visible) {inputRef.current?.focus()}
    }
  }

  function ownerChanged(retiredOwner?: ThoughtSnapshot['owner']) {
    if (retiredOwner) {
      pendingDrafts.current.delete(JSON.stringify(retiredOwner))
    }

    setNotice(null)
    inboxRef.current = null
    setInbox(null)
    dispatch({ type: 'edit', draft: '' })
    void load()
  }

  function invalidateLoad() {
    generation.current++
  }

  useEffect(() => {
    const api = window.hermesDesktop.quickEntry
    api.expandThoughts(false)
    void load()

    const offShown = api.onShown(() => {
      dispatch({ type: 'shown' })
      setExpanded(false)
      api.expandThoughts(false)

      // Preserve an unconfirmed in-memory draft if its disk write failed.
      if (!stateRef.current.draft) {
        void load()
      }

      requestAnimationFrame(() => inputRef.current?.focus())
    })

    const offState = api.onState(payload =>
      dispatch({
        connected: payload?.connected === true,
        sessions: Array.isArray(payload?.sessions) ? payload.sessions : [],
        type: 'state'
      })
    )

    const offOwner = api.onThoughtOwnerChanged(ownerChanged)

    inputRef.current?.focus()

    return () => {
      invalidateLoad()
      offShown()
      offState()
      offOwner()
    }
    // IPC subscriptions live for this window; callbacks read current state from refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="flex h-screen w-screen items-start justify-center bg-transparent p-3">
      <div
        className="flex max-h-full w-full flex-col gap-2 overflow-y-auto rounded-xl border border-(--stroke-nous) bg-(--ui-bg-elevated) p-3 shadow-nous"
        onKeyDown={event => {
          if (event.key === 'Escape') {
            event.preventDefault()
            dispatch({ type: 'dismiss' })
          }
        }}
      >
        <textarea
          aria-label={copy.placeholder}
          autoCapitalize="off"
          autoComplete="off"
          autoCorrect="off"
          className="w-full resize-none bg-transparent text-sm text-(--ui-text-primary) outline-none"
          disabled={!inbox && !notice}
          onChange={event => edit(event.target.value)}
          onKeyDown={event => {
            if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's' && !event.nativeEvent.isComposing) {
              event.preventDefault()
              void save()

              return
            }

            if (isSubmitEnter(event) && !event.shiftKey && !busyRef.current) {
              event.preventDefault()
              dispatch({ type: 'submit' })
            }
          }}
          placeholder={copy.placeholder}
          readOnly={busy}
          ref={inputRef}
          rows={2}
          spellCheck={false}
          value={state.draft}
        />
        <div className="flex items-center gap-2">
          <Button
            aria-keyshortcuts="Meta+S Control+S"
            disabled={busy || !inbox || !state.draft.trim()}
            onClick={() => void save()}
            size="xs"
            variant="secondary"
          >
            {busy ? copy.saving : copy.save}
          </Button>
          <Button
            aria-expanded={expanded}
            onClick={() => {
              setExpanded(!expanded)
              window.hermesDesktop.quickEntry.expandThoughts(!expanded)
            }}
            size="xs"
            variant="text"
          >
            {expanded ? copy.back : copy.browse}
          </Button>
        </div>
        <div className="flex items-center gap-2 text-xs text-(--ui-text-secondary)">
          <label htmlFor="quick-entry-target">{copy.send}</label>
          <select
            aria-label={copy.send}
            className="min-w-0 bg-transparent"
            disabled={!state.connected || busy}
            id="quick-entry-target"
            onChange={event => dispatch({ type: 'target', target: event.target.value })}
            value={state.target}
          >
            <option value={QUICK_TARGET_CURRENT}>{copy.current}</option>
            <option value={QUICK_TARGET_NEW}>{copy.newSession}</option>
            {state.sessions.map(session => (
              <option key={session.id} value={session.id}>
                {session.title}
              </option>
            ))}
          </select>
        </div>
        <p className="text-xs text-(--ui-text-tertiary)">
          {copy.local}
          {inbox ? ` · ${inbox.owner.profile}` : ''}
        </p>
        {notice && (
          <p className="text-xs text-(--ui-text-secondary)" role="status">
            {copy[notice]}
          </p>
        )}
        {inbox?.draft.handoffAttempted && (
          <Button
            disabled={busy}
            onClick={() => {
              edit(stateRef.current.draft)
              inputRef.current?.focus()
            }}
            size="xs"
            variant="text"
          >
            {copy.allowResend}
          </Button>
        )}
        {!state.connected && <p className="text-xs text-(--ui-text-tertiary)">{copy.offline}</p>}
        {expanded && (
          <div className="min-h-0 overflow-y-auto border-t border-(--ui-stroke-tertiary)">
            {inbox?.thoughts.length === 0 && <p className="py-2 text-sm">{copy.empty}</p>}
            {inbox?.thoughts.map(thought => (
              <article className="flex flex-col gap-1 border-b border-(--ui-stroke-tertiary) py-2" key={thought.id}>
                <p className="whitespace-pre-wrap break-words text-sm text-(--ui-text-primary)">{thought.text}</p>
                <Button
                  disabled={busy}
                  onClick={() => {
                    edit(stateRef.current.draft ? `${stateRef.current.draft}\n${thought.text}` : thought.text)
                    setExpanded(false)
                    window.hermesDesktop.quickEntry.expandThoughts(false)
                    inputRef.current?.focus()
                  }}
                  size="xs"
                  variant="text"
                >
                  {state.draft ? copy.append : copy.open}
                </Button>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
