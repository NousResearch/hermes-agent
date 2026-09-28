import { useCallback, useEffect, useRef, useState } from 'react'

import { createRealtimeVoiceSession } from '@/api/voice-realtime'
import { createDoubleClapDetector } from '@/lib/double-clap'
import { isJarvisIntroMusicPlaying, isJarvisMusicPhrase, startJarvisIntroMusic } from '@/lib/jarvis-intro-music'
import { startLiveVoice } from '@/lib/live-voice/start'
import type { RealtimeVoiceSession, RealtimeVoiceStatus } from '@/lib/realtime-voice'
import { notifyError } from '@/store/notifications'
import { requestForOwnedSession } from '@/store/session-states'
import { $subagentsBySession } from '@/store/subagents'

import { type AgentReply, type ReplyMessage, submitAndAwaitReply } from './agent-reply'
import type { ConversationStatus } from './use-voice-conversation'

/** How long one `ask_jarvis` turn may run before the voice gives up on it. */
const ASK_TIMEOUT_MS = 5 * 60_000

/**
 * How long the Live model waits for an inline answer before it is released
 * with a short spoken acknowledgement. Long enough for a quick turn, short
 * enough that the voice never goes silent mid-conversation.
 */
const ASK_INLINE_BUDGET_MS = 7_000

/** How often a queued report is offered to the model (it retries while it speaks). */
const ANNOUNCEMENT_DRAIN_MS = 400

/** The answer to an `ask_jarvis` call, or null when the inline budget ran out. */
function withInlineBudget(answer: Promise<AgentReply>): Promise<AgentReply | null> {
  return Promise.race([answer, new Promise<null>(resolve => window.setTimeout(() => resolve(null), ASK_INLINE_BUDGET_MS))])
}

/**
 * The honest outcome once the hard limit passes, whatever the turn is doing —
 * a turn whose submit itself hangs must not leave the model waiting forever.
 */
function afterHardLimit(): Promise<AgentReply> {
  return new Promise(resolve => window.setTimeout(() => resolve({ id: null, reason: 'timeout' }), ASK_TIMEOUT_MS))
}

interface UseRealtimeConversationArgs {
  sessionId: string | null
  enabled: boolean
  failureLabel: string
  /** Current transcript of this composer's chat. */
  messages: () => readonly ReplyMessage[]
  /** True while an agent turn is in flight. */
  busy: () => boolean
  /** Mark a reply as already voiced, so read-aloud never speaks it twice. */
  markSpoken: (id: string) => void
  onFatalError: () => void
  onInterrupt?: () => Promise<void> | void
  onSubmit: (text: string) => Promise<void> | void
}

const STATUS: Record<RealtimeVoiceStatus, ConversationStatus> = {
  connecting: 'transcribing',
  listening: 'listening',
  speaking: 'speaking',
  thinking: 'thinking'
}

/**
 * Live voice for the composer: the same conversation surface as the classic
 * loop (`status`, `muted`, `end`, …), backed by a Live session — OpenAI
 * Realtime or Gemini Live, whichever `voice.realtime.provider` names. A
 * spoken request that needs the agent runs as a normal turn in this chat, so
 * it stays in the transcript like any typed one.
 */
export function useRealtimeConversation({
  busy,
  sessionId,
  enabled,
  failureLabel,
  markSpoken,
  messages,
  onFatalError,
  onInterrupt,
  onSubmit
}: UseRealtimeConversationArgs) {
  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [muted, setMuted] = useState(false)
  const [level, setLevel] = useState(0)
  const sessionRef = useRef<null | RealtimeVoiceSession>(null)
  const clapDetector = useRef(createDoubleClapDetector())
  const listeningRef = useRef(false)
  const args = useRef({ busy, failureLabel, markSpoken, messages, onFatalError, onSubmit, onInterrupt })
  args.current = { busy, failureLabel, markSpoken, messages, onFatalError, onSubmit, onInterrupt }

  const announcements = useRef<string[]>([])
  const dispatching = useRef(false)
  const currentSession = useRef(sessionId)
  const pending = useRef<{ session: string | null; cancelled: boolean } | null>(null)
  /** Asks whose answer is still on its way, so a late report knows its conversation. */
  const slowAsks = useRef<{ cancelled: boolean; session: string | null }[]>([])

  if (currentSession.current !== sessionId && (pending.current || slowAsks.current.length)) {
    // A null session that just got an id is this composer's own conversation
    // being created, not a switch: whatever is in flight is adopted by it.
    const claimed = currentSession.current === null && sessionId !== null

    if (pending.current) {
      if (claimed) {
        pending.current.session = sessionId
      } else {
        pending.current.cancelled = true
        pending.current = null
        dispatching.current = false
      }
    }

    slowAsks.current = slowAsks.current.filter(task => {
      if (claimed) {
        task.session = sessionId

        return true
      }

      task.cancelled = true

      return false
    })
  }

  currentSession.current = sessionId

  // eslint-disable-next-line no-restricted-syntax -- queues backend transition events; does not mirror reactive state
  useEffect(() => {
    announcements.current = []
    const known = new Map(($subagentsBySession.get()[sessionId || ''] || []).map(item => [item.id, item.status]))

    return $subagentsBySession.subscribe(all => {
      for (const item of all[sessionId || ''] || []) {
        const before = known.get(item.id)
        known.set(item.id, item.status)

        if (before && before !== item.status && ['completed', 'failed', 'interrupted'].includes(item.status)) {
          announcements.current.push(
            `Zadanie: ${item.goal}. Stan potwierdzony przez backend: ${item.status}. ${item.summary || 'Brak raportu wyniku — nie potwierdzaj sukcesu.'}`
          )
        }
      }
    })
  }, [sessionId])

  const ask = useCallback(async (request: string) => {
    const sid = currentSession.current

    const answer = submitAndAwaitReply(
      { busy: () => args.current.busy(), messages: () => args.current.messages() },
      () => args.current.onSubmit(request),
      ASK_TIMEOUT_MS
    )

    const inline = await withInlineBudget(answer)

    if (inline) {
      if (inline.id === null) {
        return inline.reason === 'timeout'
          ? 'Hermes nadal pracuje nad odpowiedzią. Wynik pozostanie w rozmowie tekstowej.'
          : 'Hermes nie zwrócił odpowiedzi.'
      }

      args.current.markSpoken(inline.id)

      return inline.text
    }

    // Hermes is still working: release the model right away so it keeps
    // talking, and hand the real answer to it as soon as it lands.
    const task = { cancelled: false, session: sid }

    slowAsks.current.push(task)

    void Promise.race([answer, afterHardLimit()])
      .then(reply => {
        if (task.cancelled || currentSession.current !== task.session) {
          return
        }

        if (reply.id !== null) {
          args.current.markSpoken(reply.id)
          announcements.current.push(reply.text.slice(0, 3000))
        } else {
          announcements.current.push(
            reply.reason === 'timeout'
              ? 'Hermes nie zakończył tego w wyznaczonym czasie. Nie mam potwierdzonego wyniku i nie potwierdzam sukcesu.'
              : 'Hermes nie zwrócił potwierdzonego wyniku. Sprawdź rozmowę tekstową; nie traktuj tego jako ukończenia.'
          )
        }
      })
      .catch(() => {
        if (!task.cancelled && currentSession.current === task.session) {
          announcements.current.push('Nie udało się dokończyć tego zadania. Sprawdź połączenie i rozmowę tekstową.')
        }
      })
      .finally(() => {
        slowAsks.current = slowAsks.current.filter(item => item !== task)
      })

    return 'Sprawdzam. Powiem, jak będę wiedział.'
  }, [])

  const delegate = useCallback(async (request: string) => {
    const sid = currentSession.current

    const live = ($subagentsBySession.get()[sid || ''] || []).filter(
      item => item.status === 'running' || item.status === 'queued'
    )

    const stop = /^(?:proszę\s+)?(?:zatrzymaj|przerwij|stop)\b/i.test(request.trim())
    const steer = /^(?:zmień|zmien|doprecyzuj|popraw polecenie)(?=\s|$)/i.test(request.trim())

    if (stop && !live.length && args.current.busy() && args.current.onInterrupt) {
      await args.current.onInterrupt()

      return 'Wysłano prośbę o zatrzymanie bieżącego zadania. Poczekaj na potwierdzenie stanu w rozmowie.'
    }

    if ((stop || steer) && sid && live.length) {
      const results = await Promise.allSettled(
        live.map(item =>
          requestForOwnedSession<{ found?: boolean; status?: string }>(
            sid,
            async () => {
              throw new Error('Brak połączenia z sesją')
            },
            `subagent.${stop ? 'interrupt' : 'steer'}`,
            { session_id: sid, subagent_id: item.id, ...(stop ? {} : { text: request }) }
          )
        )
      )

      const accepted = results.filter(
        result => result.status === 'fulfilled' && (stop ? result.value.found : result.value.status === 'queued')
      ).length

      return `Backend przyjął polecenie dla ${accepted} z ${live.length} współpracowników. ${stop ? 'To prośba o zatrzymanie; poczekaj na potwierdzenie stanu.' : 'Zmiana została przekazana.'}`
    }

    if (/^(?:status|co robisz|jak idzie)/i.test(request.trim())) {
      return live.length
        ? live.map(item => `${item.goal}: ${item.stream.at(-1)?.text || item.status}`).join('\n')
        : 'Backend nie zgłasza aktywnych podagentów w tej rozmowie.'
    }

    if (dispatching.current || args.current.busy()) {
      return 'Poprzednie zlecenie jest jeszcze przyjmowane. Możemy dalej rozmawiać. Zapytaj o status albo doprecyzuj polecenie współpracownika.'
    }

    dispatching.current = true
    const source = args.current
    const task = { session: sid, cancelled: false }
    pending.current = task

    const ownedSource = {
      busy: () => {
        if (task.cancelled) {
          throw new Error('Conversation changed')
        }

        return args.current.busy()
      },
      messages: () => {
        if (task.cancelled) {
          throw new Error('Conversation changed')
        }

        return args.current.messages()
      }
    }

    void submitAndAwaitReply(ownedSource, () => source.onSubmit(request), ASK_TIMEOUT_MS)
      .then(reply => {
        if (task.cancelled || currentSession.current !== task.session) {
          return
        }

        if (reply.id !== null) {
          args.current.markSpoken(reply.id)
          announcements.current.push(reply.text.slice(0, 3000))
        } else {
          announcements.current.push(
            'Nie otrzymałem jeszcze potwierdzonego wyniku. Sprawdź stan zadania w rozmowie; nie traktuj tego jako ukończenia.'
          )
        }
      })
      .catch(() => {
        if (!task.cancelled && currentSession.current === task.session) {
          announcements.current.push('Przekazanie zadania nie powiodło się. Sprawdź połączenie i historię rozmowy.')
        }
      })
      .finally(() => {
        if (pending.current === task) {
          pending.current = null
          dispatching.current = false
        }
      })

    return 'Przekazuję zadanie do wykonania i powiem, gdy pojawi się wynik.'
  }, [])

  useEffect(() => {
    if (!enabled) {
      return
    }

    const timer = window.setInterval(() => {
      const next = announcements.current[0]

      if (next && sessionRef.current?.notify?.(next)) {
        announcements.current.shift()
      }
    }, ANNOUNCEMENT_DRAIN_MS)

    return () => window.clearInterval(timer)
  }, [enabled])

  const end = useCallback(async () => {
    sessionRef.current?.stop()
    sessionRef.current = null
    clapDetector.current.reset()
    listeningRef.current = false
    setMuted(false)
    setStatus('idle')
    setLevel(0)
  }, [])

  // eslint-disable-next-line no-restricted-syntax -- session lifecycle (open/close a WebRTC call), not an atom mirror
  useEffect(() => {
    if (!enabled) {
      void end()

      return undefined
    }

    let cancelled = false

    void startLiveVoice(
      {
        onAsk: ask,
        onDelegate: delegate,
        onTranscript: (role, text) => {
          if (role === 'user' && isJarvisMusicPhrase(text)) {
            startJarvisIntroMusic(true)
          }
        },
        onError: message => {
          notifyError(new Error(message), args.current.failureLabel)
          args.current.onFatalError()
        },
        // Quantized: the meter needs ~32 steps, not a re-render per sample.
        onLevel: next => {
          if (
            listeningRef.current &&
            !isJarvisIntroMusicPlaying() &&
            clapDetector.current.feed(next, performance.now())
          ) {
            startJarvisIntroMusic(true)
          }

          setLevel(Math.round(next * 32) / 32)
        },
        onStatus: next => {
          listeningRef.current = next === 'listening'
          setStatus(STATUS[next])
        }
      },
      { createSession: createRealtimeVoiceSession }
    ).then(
      session => {
        if (cancelled) {
          session.stop()
        } else {
          sessionRef.current = session
        }
      },
      error => {
        if (!cancelled) {
          notifyError(error, args.current.failureLabel)
          setStatus('idle')
          args.current.onFatalError()
        }
      }
    )

    return () => {
      cancelled = true
      void end()
    }
  }, [ask, delegate, enabled, end])

  const toggleMute = useCallback(() => {
    setMuted(value => {
      sessionRef.current?.setMuted(!value)

      return !value
    })
  }, [])

  // Turn detection is server-side: there is no local "end my turn" to force.
  const stopTurn = useCallback(() => undefined, [])
  const start = useCallback(async () => undefined, [])

  return { end, level, muted, start, status, stopTurn, toggleMute }
}
