import { type KeyboardEvent, type ReactNode, useEffect, useRef, useState } from 'react'

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'

import { type LeaderOwner, leaderOwnerKey, type LeaderSession } from '../leader-sessions'

export type LeaderVoiceStatus = 'idle' | 'listening' | 'speaking' | 'thinking' | 'transcribing'

export interface LeaderVoiceBridge {
  active: boolean
  available: boolean
  end(): Promise<void> | void
  error?: string
  start(): Promise<void> | void
  status: LeaderVoiceStatus
  stopTurn(): void
}

export interface LeaderDialogueSessionState {
  awaitingResponse: boolean
  busy: boolean
  interrupted: boolean
  messages: readonly ChatMessage[]
}

export interface LeaderDialogueProps {
  cameraControls?: ReactNode
  leaderLabel: string
  onClose(): void
  onInterrupt(session: LeaderSession, owner: LeaderOwner): Promise<void> | void
  onOpenFullChat(storedId: string, owner: LeaderOwner): Promise<void> | void
  onSubmit(text: string, session: LeaderSession, owner: LeaderOwner): Promise<void> | void
  owner: LeaderOwner
  session: LeaderSession
  sessionAvailable: boolean
  sessionError?: string
  sessionState: LeaderDialogueSessionState
  voice: LeaderVoiceBridge
}

type DisplayState =
  'error' | 'idle' | 'interrupted' | 'listening' | 'speaking' | 'thinking' | 'transcribing' | 'unavailable'

function displayState(
  sessionAvailable: boolean,
  sessionError: string | undefined,
  sessionState: LeaderDialogueSessionState,
  voice: LeaderVoiceBridge
): { label: string; state: DisplayState } {
  const error = sessionError?.trim() || voice.error?.trim()

  if (error) {
    return { label: `Error: ${error}`, state: 'error' }
  }

  if (!sessionAvailable) {
    return { label: 'Unavailable', state: 'unavailable' }
  }

  if (sessionState.interrupted) {
    return { label: 'Interrupted', state: 'interrupted' }
  }

  if (voice.active && voice.status !== 'idle') {
    const label: Record<Exclude<LeaderVoiceStatus, 'idle'>, string> = {
      listening: 'Listening',
      speaking: 'Speaking',
      thinking: 'Thinking',
      transcribing: 'Transcribing'
    }

    return { label: label[voice.status], state: voice.status }
  }

  if (sessionState.busy || sessionState.awaitingResponse) {
    return { label: 'Thinking', state: 'thinking' }
  }

  return { label: 'Ready', state: 'idle' }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function visibleTranscript(messages: readonly ChatMessage[]) {
  return messages.flatMap(message => {
    if (message.hidden) {
      return []
    }

    const text = chatMessageText(message).trim() || message.error?.trim()

    return text ? [{ id: message.id, role: message.role, text }] : []
  })
}

function dialogueLifecycleKey(owner: LeaderOwner, session: LeaderSession): string {
  return `${leaderOwnerKey(owner)}::${encodeURIComponent(session.storedId)}::${encodeURIComponent(session.runtimeId)}`
}

interface DialogueLocalState {
  actionError: string | null
  draft: string
  lifecycleGeneration: number
  lifecycleKey: string
  submitting: boolean
}

function initialLocalState(lifecycleKey: string, lifecycleGeneration = 0): DialogueLocalState {
  return { actionError: null, draft: '', lifecycleGeneration, lifecycleKey, submitting: false }
}

export function LeaderDialogue({
  cameraControls,
  leaderLabel,
  onClose,
  onInterrupt,
  onOpenFullChat,
  onSubmit,
  owner,
  session,
  sessionAvailable,
  sessionError,
  sessionState,
  voice
}: LeaderDialogueProps) {
  const lifecycleKey = dialogueLifecycleKey(owner, session)
  const [localState, setLocalState] = useState<DialogueLocalState>(() => initialLocalState(lifecycleKey))

  if (localState.lifecycleKey !== lifecycleKey) {
    setLocalState(initialLocalState(lifecycleKey, localState.lifecycleGeneration + 1))
  }

  const returnFocusRef = useRef<HTMLElement | null>(
    typeof document !== 'undefined' && document.activeElement instanceof HTMLElement ? document.activeElement : null
  )

  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  const { actionError, draft, submitting } = localState
  const endVoice = voice.end
  const status = displayState(sessionAvailable, sessionError, sessionState, voice)
  const transcript = visibleTranscript(sessionState.messages)

  const turnCanInterrupt =
    sessionState.busy ||
    sessionState.awaitingResponse ||
    (voice.active && (voice.status === 'thinking' || voice.status === 'speaking'))

  const voiceActive = voice.active

  useEffect(() => {
    textareaRef.current?.focus()
  }, [lifecycleKey])

  useEffect(() => {
    if (!voiceActive) {
      return undefined
    }

    return () => {
      void Promise.resolve(endVoice()).catch(() => undefined)
    }
  }, [endVoice, lifecycleKey, voiceActive])

  const updateIfCurrent = (
    startedFor: string,
    startedGeneration: number,
    update: (current: DialogueLocalState) => DialogueLocalState
  ) => {
    setLocalState(current =>
      current.lifecycleGeneration === startedGeneration && current.lifecycleKey === startedFor
        ? update(current)
        : current
    )
  }

  const runAction = (action: () => Promise<void> | void) => {
    const startedFor = lifecycleKey
    const startedGeneration = localState.lifecycleGeneration

    updateIfCurrent(startedFor, startedGeneration, current => ({ ...current, actionError: null }))
    void Promise.resolve(action()).catch(error => {
      updateIfCurrent(startedFor, startedGeneration, current => ({ ...current, actionError: errorText(error) }))
    })
  }

  const close = () => {
    onClose()
    returnFocusRef.current?.focus()
  }

  const submit = async () => {
    const text = draft.trim()

    if (!text || submitting) {
      return
    }

    setLocalState(current =>
      current.lifecycleKey === lifecycleKey ? { ...current, actionError: null, submitting: true } : current
    )
    const startedFor = lifecycleKey
    const startedGeneration = localState.lifecycleGeneration

    try {
      await onSubmit(text, session, owner)

      updateIfCurrent(startedFor, startedGeneration, current =>
        current.draft.trim() === text ? { ...current, draft: '' } : current
      )
    } catch (error) {
      updateIfCurrent(startedFor, startedGeneration, current => ({ ...current, actionError: errorText(error) }))
    } finally {
      updateIfCurrent(startedFor, startedGeneration, current => ({ ...current, submitting: false }))
    }
  }

  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) {
      return
    }

    event.preventDefault()
    void submit()
  }

  return (
    <>
      {cameraControls ? <div className="lunar-city-dialogue-camera-controls">{cameraControls}</div> : null}
      <section
        aria-labelledby="lunar-city-leader-dialogue-title"
        aria-modal="false"
        className="lunar-city-leader-dialogue"
        onClick={event => event.stopPropagation()}
        onKeyDown={event => {
          if (event.key === 'Escape') {
            event.stopPropagation()
            close()
          }
        }}
        onPointerDown={event => event.stopPropagation()}
        onWheel={event => event.stopPropagation()}
        role="dialog"
      >
        <header>
          <div>
            <h2 id="lunar-city-leader-dialogue-title">{leaderLabel} conversation</h2>
            <p>
              {owner.connectionId} / {owner.profile}
            </p>
          </div>
          <button aria-label="Close conversation" onClick={close} type="button">
            Close
          </button>
        </header>

        <output aria-atomic="true" aria-live="polite" data-state={status.state} role="status">
          {status.label}
        </output>

        <ol aria-label="Conversation transcript" aria-live="polite" role="log">
          {transcript.map(entry => (
            <li key={entry.id}>
              <article aria-label={`${entry.role} message`}>
                <strong>{entry.role === 'assistant' ? leaderLabel : entry.role}</strong>
                <p>{entry.text}</p>
              </article>
            </li>
          ))}
        </ol>

        <label>
          <span>Message {leaderLabel}</span>
          <textarea
            aria-label={`Message ${leaderLabel}`}
            disabled={submitting}
            onChange={event => {
              const nextDraft = event.currentTarget.value

              setLocalState(current =>
                current.lifecycleKey === lifecycleKey ? { ...current, draft: nextDraft } : current
              )
            }}
            onKeyDown={onComposerKeyDown}
            ref={textareaRef}
            value={draft}
          />
        </label>
        <div>
          <button disabled={submitting || !draft.trim()} onClick={() => void submit()} type="button">
            Send message
          </button>
          <button onClick={() => runAction(() => onOpenFullChat(session.storedId, owner))} type="button">
            Open Full Chat
          </button>
          {voice.available ? (
            <button
              disabled={!voice.active && !sessionAvailable}
              onClick={() => runAction(() => (voice.active ? voice.end() : voice.start()))}
              type="button"
            >
              {voice.active ? 'End voice' : 'Start voice'}
            </button>
          ) : (
            <button disabled type="button">
              Voice unavailable
            </button>
          )}
          {voice.active && voice.status === 'listening' ? (
            <button onClick={voice.stopTurn} type="button">
              Finish speaking
            </button>
          ) : null}
          {turnCanInterrupt ? (
            <button onClick={() => runAction(() => onInterrupt(session, owner))} type="button">
              Interrupt response
            </button>
          ) : null}
        </div>

        {voice.error ? <p role="alert">{voice.error}</p> : null}
        {actionError ? <p role="alert">{actionError}</p> : null}
      </section>
    </>
  )
}
