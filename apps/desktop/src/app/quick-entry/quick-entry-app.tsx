import { useEffect, useReducer, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { analyzePromptDraft, type PromptCoachAnalysis } from '@/lib/prompt-coach'
import { setDesktopPetAgentProfile } from '@/store/pet-agent'
import {
  initialQuickComposerState,
  QUICK_TARGET_CURRENT,
  QUICK_TARGET_NEW,
  type QuickComposerEvent,
  quickComposerReducer,
  type QuickComposerState,
  type QuickEntryAgentOption,
  type QuickEntryGroupOption,
  type QuickEntryMode
} from '@/store/quick-entry'

import { PromptCoachPreviewCard } from '../chat/composer/prompt-coach-panel'

import { quickEntryAgentVisual } from './quick-entry-agent-visual'

const HERMES_JOKES = [
  'Why did the agent bring a compass? To stay on prompt.',
  'I told my model to think outside the box. It asked for a bigger context window.',
  'Why was the AI calm? It had all its tokens in order.',
  'My favorite workout is a clean push followed by a safe pull request.'
] as const

export function hermesJokeOfTheDay(date = new Date()): string {
  const day = Math.floor(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()) / 86_400_000)

  return HERMES_JOKES[day % HERMES_JOKES.length]
}

export function quickEntryModeFromSearch(search: string): QuickEntryMode {
  return new URLSearchParams(search).get('mode') === 'agents' ? 'agents' : 'composer'
}

/** Groups have no profile of their own, so borrow one of the agent accents by
 *  name hash — a room keeps the same colour every time it is drawn. */
const GROUP_ACCENTS = ['#35e8ff', '#c67cff', '#ffd54c', '#60ffd0', '#ff8a24', '#b7ff2a'] as const

export function quickEntryGroupAccent(name: string): string {
  let hash = 0

  for (let i = 0; i < name.length; i += 1) {
    hash = (hash * 31 + name.charCodeAt(i)) >>> 0
  }

  return GROUP_ACCENTS[hash % GROUP_ACCENTS.length]
}

/**
 * Pseudo-classes and media queries cannot be expressed inline, and this window
 * renders no app stylesheet, so the picker carries its own. Everything here is
 * either a state style inline styles cannot reach (`:focus-visible`), or a
 * user-preference override (`prefers-reduced-motion`, `prefers-contrast`).
 *
 * Colours come from Hermes tokens with literal fallbacks so the picker matches
 * the sidebar it launches from in both themes instead of painting one
 * hardcoded blue over whatever is behind it.
 */
const PICKER_CSS = `
.hq-surface {
  --hq-stroke: var(--ui-stroke-secondary, rgba(127, 127, 127, 0.28));
  background: color-mix(in srgb, var(--ui-sidebar-surface-background, var(--ui-bg-sidebar, #11161f)) 92%, transparent);
  border: 1px solid color-mix(in srgb, var(--hq-stroke) 72%, transparent);
  border-radius: 11px;
  box-shadow:
    0 8px 22px rgba(0, 0, 0, 0.2),
    0 1px 0 rgba(255, 255, 255, 0.08) inset;
  -webkit-backdrop-filter: blur(14px) saturate(125%);
  backdrop-filter: blur(14px) saturate(125%);
  padding: 5px;
}
.hq-row {
  position: relative;
  border: 1px solid transparent;
  border-radius: 6px;
  background: transparent;
  transition: background 110ms ease, color 110ms ease;
}
.hq-row[data-lit='true'] {
  background: var(--ui-control-active-background, color-mix(in srgb, var(--hq-accent, #7f7f7f) 14%, transparent));
  border-color: transparent;
}
.hq-row[data-lit='true'] .hq-chip {
  box-shadow: none;
}
.hq-row:focus-visible {
  outline: 2px solid var(--hq-accent, #7f7f7f);
  outline-offset: 1px;
}
.hq-row[disabled] { opacity: 0.42; }
.hq-chip {
  background: color-mix(in srgb, var(--hq-accent, #7f7f7f) 24%, transparent);
  border: 1px solid color-mix(in srgb, var(--hq-accent, #7f7f7f) 46%, transparent);
  color: var(--hq-accent, inherit);
  transition: box-shadow 130ms ease;
}
.hq-scroll { scrollbar-width: thin; scrollbar-color: var(--hq-stroke) transparent; }
.hq-scroll::-webkit-scrollbar { width: 5px; }
.hq-scroll::-webkit-scrollbar-thumb {
  background: var(--hq-stroke);
  border-radius: 3px;
}
.hq-tab:focus-visible { outline: 2px solid var(--foreground); outline-offset: 1px; }
@media (prefers-reduced-motion: reduce) {
  .hq-row, .hq-chip { transition: none; }
}
@media (prefers-contrast: more) {
  .hq-surface { -webkit-backdrop-filter: none; backdrop-filter: none; background: var(--ui-bg-elevated, #11161f); }
  .hq-row[data-lit='true'] { background: color-mix(in srgb, var(--hq-accent, #7f7f7f) 30%, transparent); }
}
`

/**
 * The Quick Entry composer — the whole renderer surface of the global-hotkey
 * mini window. It adds a compact profile-backed agent row above the existing
 * input and session-target picker: choosing an agent opens a fresh real HUD;
 * typing still uses the original capture path. This is not a second chat.
 *
 * All behavior rides `quickComposerReducer` (pure, unit-tested): submit sends
 * the trimmed text + target through the shell and asks to hide; an empty submit
 * does neither so a stray Enter can't make the window vanish; Escape and losing
 * focus dismiss without sending; a dead gateway disables the input entirely
 * (the reducer refuses the send AND the input paints the reconnect hint).
 *
 * The window itself has no gateway connection. Its view of backend truth — is
 * the gateway up, which recent sessions exist — is pushed in by the primary
 * renderer through main (`onState`), and its text goes back the same road to
 * the primary renderer's normal prompt-submit path.
 */
export function QuickEntryApp() {
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const agentButtonRefs = useRef<Array<HTMLButtonElement | null>>([])
  const groupButtonRefs = useRef<Array<HTMLButtonElement | null>>([])
  const pickerRef = useRef<HTMLElement>(null)
  // Main encodes the launch mode in the URL because Electron's first
  // did-finish-load event can beat React's onShown subscription. Subsequent
  // summons still switch mode through onShown on the reused window.
  const [mode, setMode] = useState<QuickEntryMode>(() => quickEntryModeFromSearch(window.location.search))
  const [pickerKind, setPickerKind] = useState<'agents' | 'groups'>('agents')
  // Hover highlight is presentation only and starts empty, so the picker opens
  // with NOTHING lit and exactly one row lights as the pointer crosses it.
  // Deliberately separate from `state.activeAgentIndex`, which is the
  // reducer's keyboard/commit cursor and must keep starting at 0.
  const [hoverIndex, setHoverIndex] = useState(-1)
  const [groupHoverIndex, setGroupHoverIndex] = useState(-1)
  const [groupIndex, setGroupIndex] = useState(0)
  const [coachAnalysis, setCoachAnalysis] = useState<PromptCoachAnalysis | null>(null)
  const [coachOpen, setCoachOpen] = useState(false)
  const [coachRequest, setCoachRequest] = useState<null | { requestId: string; text: string }>(null)

  // The reducer returns { send, state }; this wrapper performs the side effect
  // (hand the payload to the shell, ask to hide) and stores the next state, so
  // the decision stays pure and testable while the effects stay in one place.
  const [state, dispatch] = useReducer((current: QuickComposerState, event: QuickComposerEvent) => {
    const { send, state: next } = quickComposerReducer(current, event)
    const api = window.hermesDesktop?.quickEntry

    if (send) {
      api?.submit(send)
    } else if (!next.visible && current.visible) {
      api?.dismiss()
    }

    return next
  }, initialQuickComposerState)

  // Re-summoned by the chord: the shell reuses the window, so reset the draft
  // and take the keyboard back for a fresh capture. Also adopt gateway-state
  // pushes (connection + recent sessions) relayed from the primary renderer.
  useEffect(() => {
    const api = window.hermesDesktop?.quickEntry

    const offShown = api?.onShown(payload => {
      setCoachRequest(null)
      const nextMode = payload?.mode === 'agents' ? 'agents' : 'composer'
      setMode(nextMode)
      setPickerKind('agents')
      // A re-summon must open quiet: no row lit, and the keyboard owning focus
      // again even if the last session ended on a hover.
      setHoverIndex(-1)
      setGroupHoverIndex(-1)
      setGroupIndex(0)
      dispatch({ type: 'shown' })
      requestAnimationFrame(() => {
        if (nextMode === 'agents') {
          // Focus the PANEL, not a row. Focusing a row would light it, and the
          // picker must open with nothing chosen — the first arrow key steps
          // into the list, the pointer lights whatever it crosses.
          pickerRef.current?.focus()
        } else {
          inputRef.current?.focus()
        }
      })
    })

    const offLaunchResult = api?.onLaunchResult?.(result => {
      dispatch({ result, type: 'launch-result' })
    })

    const offState = api?.onState(payload => {
      dispatch({
        connected: payload?.connected === true,
        agents: Array.isArray(payload?.agents) ? payload.agents : [],
        groups: Array.isArray(payload?.groups) ? payload.groups : [],
        sessions: Array.isArray(payload?.sessions) ? payload.sessions : [],
        type: 'state'
      })
    })

    inputRef.current?.focus()

    return () => {
      offShown?.()
      offState?.()
      offLaunchResult?.()
    }
  }, [])

  useEffect(() => {
    const api = window.hermesDesktop?.quickEntry

    if (!api?.onPromptCoachResult || !coachRequest) {
      return
    }

    return api.onPromptCoachResult(result => {
      if (result.requestId !== coachRequest.requestId || result.text !== coachRequest.text) {
        return
      }

      setCoachRequest(null)

      if (result.analysis) {
        setCoachAnalysis(result.analysis)
      }
    })
  }, [coachRequest])

  // Quick Entry is a separate renderer, so it cannot consume the main
  // composer's draft suggestion bus. Diagnose locally after the same 600ms
  // quiet period; an explicit preview action is relayed to the primary
  // renderer for AI enhancement.
  useEffect(() => {
    setCoachOpen(false)

    const timer = window.setTimeout(() => setCoachAnalysis(analyzePromptDraft(state.draft)), 600)

    return () => window.clearTimeout(timer)
  }, [state.draft])

  const openCoach = (analysis: PromptCoachAnalysis) => {
    setCoachAnalysis({ ...analysis, generatedBy: 'pending' })
    setCoachOpen(true)

    const api = window.hermesDesktop?.quickEntry

    if (!api?.requestPromptCoach) {
      setCoachAnalysis(analysis)

      return
    }

    const requestId = crypto.randomUUID()
    setCoachRequest({ requestId, text: state.draft })
    api.requestPromptCoach({ requestId, target: state.target, text: state.draft })
  }

  const activeAgent = state.agents[state.activeAgentIndex]
  const activeVisual = quickEntryAgentVisual(activeAgent?.profile)

  const openAgent = (agent: QuickEntryAgentOption) => {
    setDesktopPetAgentProfile(agent.profile)
    dispatch({ profile: agent.profile, requestId: crypto.randomUUID(), type: 'open-agent' })
  }

  const openGroup = (group: QuickEntryGroupOption) => {
    dispatch({ groupId: group.groupId, requestId: crypto.randomUUID(), type: 'open-group' })
  }

  if (mode === 'agents') {
    const joke = hermesJokeOfTheDay()
    const showingGroups = pickerKind === 'groups'

    // Focus is moved here rather than from an effect so that ONLY a real arrow
    // press moves it — an effect keyed on the selection index also fires for
    // pointer hover, which is how the mouse used to steal keyboard focus.
    const moveAgent = (delta: -1 | 1) => {
      const reachable = state.agents.map((agent, index) => ({ agent, index })).filter(entry => entry.agent.reachable)

      if (!reachable.length) {
        return
      }

      const at = Math.max(
        0,
        reachable.findIndex(entry => entry.index === state.activeAgentIndex)
      )

      const next = reachable[(at + delta + reachable.length) % reachable.length]

      dispatch({ index: next.index, type: 'select-agent' })
      agentButtonRefs.current[next.index]?.focus()
    }

    const moveGroup = (delta: -1 | 1) => {
      const reachable = state.groups.map((group, index) => ({ group, index })).filter(entry => entry.group.reachable)

      if (!reachable.length) {
        return
      }

      const at = Math.max(
        0,
        reachable.findIndex(entry => entry.index === groupIndex)
      )

      const next = reachable[(at + delta + reachable.length) % reachable.length]

      setGroupIndex(next.index)
      groupButtonRefs.current[next.index]?.focus()
    }

    return (
      <div
        style={{
          alignItems: 'center',
          background: 'transparent',
          display: 'flex',
          height: '100vh',
          justifyContent: 'center',
          padding: 3,
          width: '100vw'
        }}
      >
        <style>{PICKER_CSS}</style>
        <section
          aria-label="Choose an agent"
          onKeyDown={event => {
            if (event.key === 'Escape') {
              event.preventDefault()
              dispatch({ type: 'dismiss' })

              return
            }

            const back = event.key === 'ArrowLeft' || event.key === 'ArrowUp'
            const forward = event.key === 'ArrowRight' || event.key === 'ArrowDown'

            if (!back && !forward) {
              return
            }

            event.preventDefault()

            // The arrows must drive whichever list is on screen. Before this
            // they always moved the agent cursor, so in the Groups tab they
            // silently walked an invisible selection and groups could not be
            // reached from the keyboard at all.
            if (showingGroups) {
              moveGroup(back ? -1 : 1)
            } else {
              moveAgent(back ? -1 : 1)
            }
          }}
          style={{
            background: 'transparent',
            border: 'none',
            borderRadius: 11,
            overflow: 'visible',
            padding: 0,
            outline: 'none',
            width: '100%'
          }}
          tabIndex={-1}
        >
          <div className="hq-surface" data-testid="agent-picker-surface">
            <div
              aria-label="Conversation type"
              role="tablist"
              style={{ alignItems: 'center', display: 'flex', gap: 3, marginBottom: 5 }}
            >
              {(['agents', 'groups'] as const).map(kind => (
                <button
                  aria-selected={pickerKind === kind}
                  className="hq-tab"
                  key={kind}
                  onClick={() => {
                    setPickerKind(kind)
                    setHoverIndex(-1)
                    setGroupHoverIndex(-1)
                  }}
                  role="tab"
                  style={{
                    background: pickerKind === kind ? 'var(--ui-bg-tertiary, rgba(127,127,127,0.14))' : 'transparent',
                    border: '1px solid',
                    borderColor:
                      pickerKind === kind ? 'var(--ui-stroke-secondary, rgba(127,127,127,0.22))' : 'transparent',
                    borderRadius: 7,
                    color: pickerKind === kind ? 'var(--foreground)' : 'var(--muted-foreground, #737373)',
                    cursor: 'pointer',
                    fontSize: 11,
                    fontWeight: 650,
                    letterSpacing: '0.01em',
                    padding: '3px 8px'
                  }}
                  type="button"
                >
                  {kind === 'agents' ? 'Agents' : `Groups${state.groups.length ? ` ${state.groups.length}` : ''}`}
                </button>
              ))}
              <span
                style={{
                  color: 'var(--muted-foreground, #9ca3af)',
                  fontSize: 9.5,
                  letterSpacing: '0.06em',
                  marginLeft: 'auto',
                  textTransform: 'uppercase'
                }}
              >
                Talk to
              </span>
            </div>

            {/* The window cannot resize, so the list scrolls instead of being
                clipped by the frame once the roster outgrows the frame. */}
            <div
              className="hq-scroll"
              style={{
                display: 'grid',
                gap: 2,
                gridTemplateColumns: 'minmax(0, 1fr)',
                maxHeight: 166,
                overflowY: 'auto',
                paddingRight: 2
              }}
            >
              {!showingGroups &&
                state.agents.map((agent, index) => {
                  const accent = quickEntryAgentVisual(agent.profile).accent

                  return (
                    <Button
                      aria-current={hoverIndex === index ? 'true' : undefined}
                      aria-label={agent.displayName}
                      className="hq-row"
                      data-lit={hoverIndex === index}
                      disabled={!agent.reachable || state.submitting}
                      key={agent.profile}
                      onBlur={() => setHoverIndex(current => (current === index ? -1 : current))}
                      onClick={() => openAgent(agent)}
                      onFocus={() => {
                        setHoverIndex(index)
                        dispatch({ index, type: 'select-agent' })
                      }}
                      onMouseEnter={() => {
                        setHoverIndex(index)
                        dispatch({ index, type: 'select-agent' })
                      }}
                      onMouseLeave={() => setHoverIndex(current => (current === index ? -1 : current))}
                      ref={element => {
                        agentButtonRefs.current[index] = element
                      }}
                      style={{
                        ['--hq-accent' as string]: accent,
                        height: 25,
                        justifyContent: 'flex-start',
                        minWidth: 0,
                        padding: '2px 6px'
                      }}
                      variant="ghost"
                    >
                      <span
                        aria-hidden
                        className="hq-chip"
                        style={{
                          alignItems: 'center',
                          borderRadius: 5,
                          display: 'inline-flex',
                          flexShrink: 0,
                          fontSize: 11,
                          fontWeight: 800,
                          height: 18,
                          justifyContent: 'center',
                          width: 18
                        }}
                      >
                        {agent.displayName.slice(0, 1).toUpperCase()}
                      </span>
                      <span
                        style={{
                          color: 'var(--foreground, #f7f8fa)',
                          fontSize: 11,
                          fontWeight: 650,
                          minWidth: 0,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}
                      >
                        {agent.displayName}
                      </span>
                    </Button>
                  )
                })}
              {showingGroups &&
                state.groups.map((group, index) => {
                  const accent = quickEntryGroupAccent(group.displayName)

                  return (
                    <Button
                      aria-current={groupHoverIndex === index ? 'true' : undefined}
                      aria-label={group.displayName}
                      className="hq-row"
                      data-lit={groupHoverIndex === index}
                      disabled={!group.reachable || state.submitting}
                      key={group.groupId}
                      onBlur={() => setGroupHoverIndex(current => (current === index ? -1 : current))}
                      onClick={() => openGroup(group)}
                      onFocus={() => {
                        setGroupHoverIndex(index)
                        setGroupIndex(index)
                      }}
                      onMouseEnter={() => {
                        setGroupHoverIndex(index)
                        setGroupIndex(index)
                      }}
                      onMouseLeave={() => setGroupHoverIndex(current => (current === index ? -1 : current))}
                      ref={element => {
                        groupButtonRefs.current[index] = element
                      }}
                      style={{
                        ['--hq-accent' as string]: accent,
                        height: 28,
                        justifyContent: 'flex-start',
                        minWidth: 0,
                        padding: '2px 6px'
                      }}
                      variant="ghost"
                    >
                      <span
                        aria-hidden
                        className="hq-chip"
                        style={{
                          alignItems: 'center',
                          borderRadius: 5,
                          display: 'inline-flex',
                          flexShrink: 0,
                          fontSize: 11,
                          fontWeight: 850,
                          height: 18,
                          justifyContent: 'center',
                          width: 18
                        }}
                      >
                        {group.displayName.slice(0, 1).toUpperCase()}
                      </span>
                      <span style={{ minWidth: 0, overflow: 'hidden', textAlign: 'left' }}>
                        <span
                          style={{
                            display: 'block',
                            fontSize: 11,
                            fontWeight: 650,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap'
                          }}
                        >
                          {group.displayName}
                        </span>
                        <span style={{ color: 'var(--muted-foreground, #9ca3af)', display: 'block', fontSize: 9 }}>
                          {group.memberCount ?? 0} agents
                        </span>
                      </span>
                    </Button>
                  )
                })}
            </div>

            {showingGroups && state.groups.length === 0 && (
              <div
                style={{
                  color: 'var(--muted-foreground, #9ca3af)',
                  fontSize: 11,
                  padding: '18px 8px',
                  textAlign: 'center'
                }}
              >
                No groups available
              </div>
            )}

            {/* Decoration, not status: it must never be announced over the
                agent the user is actually choosing, and never outgrow a row. */}
            <div
              aria-hidden
              data-testid="agent-picker-joke"
              style={{
                borderTop: '1px solid var(--ui-stroke-secondary, rgba(127,127,127,0.18))',
                color: 'var(--muted-foreground, #9ca3af)',
                fontSize: 9.5,
                lineHeight: 1.4,
                marginTop: 5,
                opacity: 0.75,
                overflow: 'hidden',
                paddingTop: 5,
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}
              title={joke}
            >
              {joke}
            </div>

            {state.launchError && (
              <div role="alert" style={{ color: 'var(--ui-text-danger)', fontSize: 10, margin: '6px 2px 0' }}>
                {state.launchError}
              </div>
            )}
          </div>
        </section>
      </div>
    )
  }

  return (
    <div
      style={{
        alignItems: 'center',
        background: 'transparent',
        display: 'flex',
        height: '100vh',
        justifyContent: 'center',
        padding: 12,
        width: '100vw'
      }}
    >
      <div
        style={{
          background: 'var(--ui-bg-elevated, var(--background))',
          border: '1px solid var(--ui-stroke-secondary, rgba(127,127,127,0.35))',
          borderRadius: 12,
          boxShadow: '0 18px 48px rgba(0,0,0,0.38)',
          display: 'grid',
          gap: 14,
          gridTemplateColumns: '230px minmax(0, 1fr)',
          minHeight: 360,
          overflow: 'hidden',
          padding: 14,
          width: '100%'
        }}
      >
        <div
          style={{
            alignItems: 'center',
            background: `radial-gradient(circle at 50% 58%, ${activeVisual.glow}, transparent 67%)`,
            border: `1px solid color-mix(in srgb, ${activeVisual.accent} 42%, transparent)`,
            borderRadius: 10,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'flex-end',
            minHeight: 330,
            overflow: 'hidden',
            padding: '8px 8px 10px',
            position: 'relative'
          }}
        >
          <img
            alt={`${activeAgent?.displayName || 'Hermes'} pose`}
            src={activeVisual.pose}
            style={{
              filter: `drop-shadow(0 0 18px ${activeVisual.glow})`,
              height: 280,
              objectFit: 'contain',
              transition: 'filter 140ms ease, opacity 120ms ease, transform 140ms ease',
              width: '100%'
            }}
          />
          <div
            aria-live="polite"
            style={{
              color: activeVisual.accent,
              fontSize: 14,
              fontWeight: 650,
              letterSpacing: '0.03em',
              lineHeight: 1.2,
              textAlign: 'center'
            }}
          >
            {activeAgent?.displayName || 'Hermes'}
          </div>
          <div style={{ color: 'var(--muted-foreground, #8a8a8a)', fontSize: 10, marginTop: 3 }}>
            {activeVisual.role}
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, minWidth: 0 }}>
          {state.agents.length > 0 && (
            <div
              style={{
                display: 'grid',
                gap: 5,
                gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                maxHeight: 190,
                overflowY: 'auto',
                paddingBottom: 2
              }}
            >
              {state.agents.map((agent, index) => (
                <Button
                  aria-label={agent.displayName}
                  aria-pressed={state.activeAgentIndex === index}
                  disabled={!state.connected || !agent.reachable || state.submitting}
                  key={agent.profile}
                  onClick={() =>
                    dispatch({ profile: agent.profile, requestId: crypto.randomUUID(), type: 'open-agent' })
                  }
                  onFocus={() => {
                    setDesktopPetAgentProfile(agent.profile)
                    dispatch({ index, type: 'select-agent' })
                  }}
                  onMouseEnter={() => {
                    setDesktopPetAgentProfile(agent.profile)
                    dispatch({ index, type: 'select-agent' })
                  }}
                  size="sm"
                  style={{
                    border:
                      state.activeAgentIndex === index
                        ? `1px solid ${quickEntryAgentVisual(agent.profile).accent}`
                        : '1px solid transparent',
                    justifyContent: 'flex-start',
                    minWidth: 0
                  }}
                  variant="ghost"
                >
                  <span
                    aria-hidden
                    style={{
                      alignItems: 'center',
                      background: agent.color || 'var(--ui-bg-quaternary)',
                      borderRadius: 999,
                      display: 'inline-flex',
                      height: 20,
                      justifyContent: 'center',
                      width: 20
                    }}
                  >
                    {agent.displayName.slice(0, 1).toUpperCase()}
                  </span>
                  <span>{agent.displayName}</span>
                </Button>
              ))}
            </div>
          )}
          <div style={{ alignItems: 'center', display: 'flex', gap: 10 }}>
            <span
              aria-hidden
              style={{
                color: 'var(--muted-foreground, #8a8a8a)',
                flexShrink: 0,
                fontSize: 15,
                lineHeight: 1,
                userSelect: 'none'
              }}
            >
              ›
            </span>
            <textarea
              aria-label="Quick Entry"
              autoCapitalize="off"
              autoComplete="off"
              autoCorrect="off"
              disabled={!state.connected}
              onBlur={event => {
                // Moving focus to the target picker is not leaving the window.
                if (!event.relatedTarget) {
                  dispatch({ type: 'blur' })
                }
              }}
              onChange={event => {
                setCoachRequest(null)
                dispatch({ draft: event.target.value, type: 'edit' })
              }}
              onKeyDown={event => {
                if (event.altKey && !event.ctrlKey && !event.metaKey && /^[1-9]$/.test(event.key)) {
                  const agent = state.agents[Number(event.key) - 1]

                  if (agent?.reachable) {
                    event.preventDefault()
                    dispatch({ profile: agent.profile, requestId: crypto.randomUUID(), type: 'open-agent' })
                  }
                } else if (!state.draft.trim() && (event.key === 'ArrowLeft' || event.key === 'ArrowUp')) {
                  event.preventDefault()
                  dispatch({ delta: -1, type: 'move-agent' })
                } else if (!state.draft.trim() && (event.key === 'ArrowRight' || event.key === 'ArrowDown')) {
                  event.preventDefault()
                  dispatch({ delta: 1, type: 'move-agent' })
                } else if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  const agent = state.draft.trim() ? null : state.agents[state.activeAgentIndex]

                  if (agent?.reachable) {
                    dispatch({ profile: agent.profile, requestId: crypto.randomUUID(), type: 'open-agent' })
                  } else {
                    const analysis = analyzePromptDraft(state.draft)

                    if (analysis) {
                      setCoachAnalysis(analysis)
                      openCoach(analysis)
                    } else {
                      dispatch({ type: 'submit' })
                    }
                  }
                } else if (event.key === 'Escape') {
                  event.preventDefault()
                  dispatch({ type: 'dismiss' })
                }
              }}
              placeholder={state.connected ? 'Ask Hermes…' : 'Not connected — open Hermes to reconnect'}
              ref={inputRef}
              rows={1}
              spellCheck={false}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--foreground, #eee)',
                flex: 1,
                fontFamily: 'inherit',
                fontSize: 15,
                minWidth: 0,
                opacity: state.connected ? 1 : 0.55,
                outline: 'none',
                resize: 'none'
              }}
              value={state.draft}
            />
          </div>
          {coachAnalysis && !coachOpen && (
            <Button
              aria-label="Improve Quick Entry prompt"
              onClick={() => openCoach(coachAnalysis)}
              size="sm"
              style={{ alignSelf: 'flex-start' }}
              type="button"
              variant="outline"
            >
              Improve prompt · {coachAnalysis.reason}
            </Button>
          )}
          {coachAnalysis && coachOpen && (
            <PromptCoachPreviewCard
              className="mx-0 mb-0 w-full shadow-none"
              onApply={text => {
                dispatch({ draft: text, type: 'edit' })
                setCoachOpen(false)
                requestAnimationFrame(() => inputRef.current?.focus())
              }}
              onClose={() => setCoachOpen(false)}
              onSendOriginal={() => dispatch({ type: 'submit' })}
              preview={{ ...coachAnalysis, original: state.draft }}
            />
          )}
          <div style={{ alignItems: 'center', display: 'flex', gap: 8 }}>
            <label
              htmlFor="quick-entry-target"
              style={{
                color: 'var(--muted-foreground, #8a8a8a)',
                flexShrink: 0,
                fontSize: 11,
                userSelect: 'none'
              }}
            >
              Send to
            </label>
            <select
              aria-label="Target session"
              disabled={!state.connected}
              id="quick-entry-target"
              onChange={event => dispatch({ target: event.target.value, type: 'target' })}
              onKeyDown={event => {
                if (event.key === 'Escape') {
                  event.preventDefault()
                  dispatch({ type: 'dismiss' })
                }
              }}
              style={{
                background: 'transparent',
                border: '1px solid var(--ui-stroke-secondary, rgba(127,127,127,0.35))',
                borderRadius: 6,
                color: 'var(--foreground, #eee)',
                fontSize: 11,
                maxWidth: 320,
                padding: '2px 6px'
              }}
              value={state.target}
            >
              <option value={QUICK_TARGET_CURRENT}>Current chat</option>
              <option value={QUICK_TARGET_NEW}>New session</option>
              {state.sessions.map(session => (
                <option key={session.id} value={session.id}>
                  {session.title}
                </option>
              ))}
            </select>
          </div>
          {state.launchError && (
            <div role="alert" style={{ color: 'var(--ui-text-danger)', fontSize: 11 }}>
              {state.launchError}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
