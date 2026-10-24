'use client'

import './decision-tree.css'

import {
  type CSSProperties,
  type KeyboardEvent,
  useCallback,
  useEffect,
  useId,
  useImperativeHandle,
  useMemo,
  useRef,
  useState
} from 'react'

import { Button } from '@/components/ui/button'
import { Kbd } from '@/components/ui/kbd'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { Check, ChevronLeft, ChevronRight } from '@/lib/icons'
import { cn } from '@/lib/utils'

import type {
  Decision,
  DecisionChoice,
  DecisionEffect,
  DecisionEffectFlag,
  DecisionOption,
  DecisionRisk,
  DecisionTreeProps
} from './types'

/** What the pointer is on: a branch, or one of its follow-up's options. */
interface Hovered {
  branch: number
  decision: string
  then?: number
}

/** The consequence flag being read, addressed by a stable key so two options in
 *  the same branch cannot claim the same note. */
interface OpenNote {
  decision: string
  key: string
}

interface Settled {
  answer?: string
  choice: DecisionChoice
}

const RISK_CLASS: Record<DecisionRisk, string> = {
  high: 'text-[color-mix(in_srgb,var(--dt-destructive)_88%,var(--dt-foreground))]',
  low: 'text-(--ui-text-tertiary)',
  medium: 'text-[color-mix(in_srgb,var(--theme-primary)_85%,var(--dt-foreground))]'
}

const BRANCH_CLASS =
  'flex w-full cursor-pointer items-start gap-2 rounded-[0.25rem] px-1.5 py-1 text-left text-(--ui-text-secondary) transition-colors duration-100 hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary) hover:transition-none focus-visible:text-(--ui-text-primary)'

const FLAG_CLASS =
  'cursor-pointer text-left text-(--ui-text-secondary) underline decoration-dotted decoration-(--ui-stroke-secondary) underline-offset-2 hover:text-(--ui-text-primary) hover:decoration-(--ui-text-tertiary)'

const ACTION_CLASS = 'shrink-0 text-[10px] font-medium tracking-wide uppercase'

function effectsOf(option: DecisionOption): DecisionEffect[] {
  if (!option.effects) {
    return []
  }

  return Array.isArray(option.effects) ? option.effects : [option.effects]
}

function isFlag(effect: DecisionEffect): effect is DecisionEffectFlag {
  return typeof effect !== 'string'
}

const noteKey = (decision: number, branch: number, then: number | undefined, effect: number) =>
  `${decision}-${branch}-${then ?? 'b'}-${effect}`

/** Every flagged consequence on a decision in reading order — the rotation `F`
 *  walks, and the set of notes the tree can have open. */
function flagKeysOf(decision: Decision, decisionIndex: number): string[] {
  return decision.options.flatMap((option, branch) => [
    ...effectsOf(option).flatMap((effect, effectIndex) =>
      isFlag(effect) ? [noteKey(decisionIndex, branch, undefined, effectIndex)] : []
    ),
    ...(option.next?.options ?? []).flatMap((then, thenIndex) =>
      effectsOf(then).flatMap((effect, effectIndex) =>
        isFlag(effect) ? [noteKey(decisionIndex, branch, thenIndex, effectIndex)] : []
      )
    )
  ])
}

interface OptionNodeProps {
  /** Takes a digit shortcut and sits in the tab order. False when the node is
   *  dormant (a follow-up whose branch has not been picked) or already settled. */
  active: boolean
  chosen: boolean
  dimmed: boolean
  index: number
  lit: boolean
  noteId: (effect: number) => string
  onChoose: () => void
  onHover: (index: number | null) => void
  onPeek: (effect: number) => void
  openEffect: number | null
  option: DecisionOption
  order: number
  rest: boolean
}

/**
 * One branch — its label, what it leads to. A branch is a real button: Enter and
 * Space pick it, and its digit is announced as a shortcut. A flagged
 * consequence is a button too, described by its note.
 */
function OptionNode({
  active,
  chosen,
  dimmed,
  index,
  lit,
  noteId,
  onChoose,
  onHover,
  onPeek,
  openEffect,
  option,
  order,
  rest
}: OptionNodeProps) {
  const { t } = useI18n()
  const effects = effectsOf(option)

  return (
    <div
      className="dt-node"
      data-chosen={chosen || undefined}
      data-dimmed={dimmed || undefined}
      data-lit={lit || undefined}
      data-rest={rest || undefined}
      data-slot="decision-tree-branch-node"
      style={{ '--dt-order': order } as CSSProperties}
    >
      <button
        aria-keyshortcuts={active ? String(index + 1) : undefined}
        className={BRANCH_CLASS}
        data-slot="decision-tree-branch"
        onClick={onChoose}
        onFocus={() => onHover(index)}
        onPointerEnter={() => onHover(index)}
        tabIndex={active ? undefined : -1}
        type="button"
      >
        {active ? (
          <Kbd className="mt-px shrink-0" size="sm">
            {index + 1}
          </Kbd>
        ) : (
          <span aria-hidden className="size-[1.125rem] shrink-0" />
        )}
        {option.icon ? (
          <span aria-hidden className="mt-px shrink-0 [&_svg]:size-3.5">
            {option.icon}
          </span>
        ) : null}
        <span className="min-w-0 flex-1 wrap-anywhere">{option.label}</span>
        {option.pick ? (
          <span className={cn(ACTION_CLASS, 'text-(--theme-primary)')} data-slot="decision-tree-pick">
            {t.decisionTree.agentsPick}
          </span>
        ) : null}
        {option.apps?.length ? (
          <span aria-hidden className="flex shrink-0 items-center gap-1 [&_svg]:size-3">
            {option.apps}
          </span>
        ) : null}
      </button>

      {effects.length > 0 ? (
        <ul className="grid gap-0.5 pr-1.5 pl-7 text-xs leading-relaxed" data-slot="decision-tree-effects">
          {effects.map((effect, effectIndex) =>
            isFlag(effect) ? (
              <li className="relative" key={effectIndex}>
                <button
                  aria-describedby={noteId(effectIndex)}
                  className={FLAG_CLASS}
                  data-slot="decision-tree-flag"
                  onClick={() => onPeek(effectIndex)}
                  onFocus={() => onPeek(effectIndex)}
                  onPointerEnter={() => onPeek(effectIndex)}
                  type="button"
                >
                  {effect.text}
                </button>
                {/* Always in the DOM so the flag can point at it while it is
                    closed; it only paints once it is the open note. */}
                <span
                  className={cn(openEffect === effectIndex ? 'dt-note-open' : 'sr-only')}
                  data-slot="decision-tree-note"
                  id={noteId(effectIndex)}
                  role="tooltip"
                >
                  {effect.note}
                  {effect.source ? <span className="mt-1 block text-(--ui-text-tertiary)">{effect.source}</span> : null}
                </span>
              </li>
            ) : (
              <li key={effectIndex}>{effect}</li>
            )
          )}
        </ul>
      ) : null}
    </div>
  )
}

/**
 * Decisions the agent will not take alone: a question, two to four branches, and
 * what each branch changes. The agent's own pick is marked; a person decides.
 *
 * Keyboard-first once focus is inside it — digits pick, the arrows (or J/K) move
 * between decisions, `F` walks the flagged consequences, `Esc` closes what is
 * open. The tree takes no key from a field it does not own, and it never moves
 * focus or navigates on its own.
 */
export function DecisionTree({ allowOther = true, className, decisions, onChoose, ref, title }: DecisionTreeProps) {
  const { t } = useI18n()
  const copy = t.decisionTree
  const noteBase = useId()

  const [activeId, setActiveId] = useState<string | null>(decisions[0]?.id ?? null)
  const [hovered, setHovered] = useState<Hovered | null>(null)
  const [openNote, setOpenNote] = useState<OpenNote | null>(null)
  const [otherId, setOtherId] = useState<string | null>(null)
  const [otherText, setOtherText] = useState('')
  /** decision id → the branch whose follow-up is now waiting on a second pick. */
  const [routes, setRoutes] = useState<Record<string, number>>({})
  const [settled, setSettled] = useState<Record<string, Settled>>({})
  const [pendingId, setPendingId] = useState<string | null>(null)
  const [announcement, setAnnouncement] = useState('')
  const otherFieldRef = useRef<HTMLTextAreaElement | null>(null)

  const active = useMemo(
    () => decisions.find(decision => decision.id === activeId) ?? decisions[0] ?? null,
    [activeId, decisions]
  )

  const activeIndex = active ? decisions.findIndex(decision => decision.id === active.id) : -1
  const activeFlagKeys = active ? flagKeysOf(active, activeIndex) : []

  const settle = useCallback(
    (decision: Decision, choice: DecisionChoice) => {
      const route = choice.then ? `${choice.option.label} · ${choice.then.label}` : choice.option.label

      setSettled(current => ({ ...current, [decision.id]: { choice } }))
      setOtherId(null)
      setOtherText('')
      setPendingId(decision.id)
      setAnnouncement(choice.other ? copy.instructionSent : copy.chose(route))

      // The agent's answer is the receipt's detail: a rejected reply leaves the
      // pick standing instead of taking the outcome away from the user.
      void Promise.resolve(onChoose?.(decision, choice))
        .then(answer => {
          if (!answer) {
            return
          }

          setSettled(current => {
            const entry = current[decision.id]

            return entry ? { ...current, [decision.id]: { ...entry, answer } } : current
          })
        })
        .catch(() => undefined)
        .finally(() => setPendingId(current => (current === decision.id ? null : current)))
    },
    [copy, onChoose]
  )

  const chooseBranch = useCallback(
    (decision: Decision, index: number) => {
      const option = decision.options[index]

      if (!option || settled[decision.id]) {
        return
      }

      if (option.next && option.next.options.length > 0) {
        // Waking a follow-up is not the decision itself: the route settles on
        // the second pick, and only then does the agent get the whole path.
        setRoutes(current => ({ ...current, [decision.id]: index }))
        setHovered(null)
        setOpenNote(null)
        setAnnouncement(option.next.question)

        return
      }

      settle(decision, { index, option })
    },
    [settle, settled]
  )

  const chooseFollowUp = useCallback(
    (decision: Decision, branch: number, index: number) => {
      const option = decision.options[branch]
      const then = option?.next?.options[index]

      if (!option || !then || settled[decision.id]) {
        return
      }

      settle(decision, { index: branch, option, then })
    },
    [settle, settled]
  )

  const sendOther = useCallback(
    (decision: Decision) => {
      const instruction = otherText.trim()

      if (!instruction || settled[decision.id]) {
        return
      }

      // A written instruction is the whole route: there is no option to point
      // at, so it arrives as `other` with the text as the label.
      settle(decision, { index: -1, option: { label: instruction }, other: instruction })
    },
    [otherText, settle, settled]
  )

  const peek = useCallback(
    (decision: Decision, decisionIndex: number) => {
      const keys = flagKeysOf(decision, decisionIndex)

      if (keys.length === 0) {
        return
      }

      const at = openNote?.decision === decision.id ? keys.indexOf(openNote.key) : -1
      const key = keys[(at + 1) % keys.length]

      setOpenNote({ decision: decision.id, key })
    },
    [openNote]
  )

  const move = useCallback(
    (step: 1 | -1) => {
      setActiveId(current => {
        const at = decisions.findIndex(decision => decision.id === current)

        if (at < 0) {
          return decisions[0]?.id ?? null
        }

        return decisions[Math.min(Math.max(at + step, 0), decisions.length - 1)]?.id ?? null
      })
      setHovered(null)
      setOpenNote(null)
      setOtherId(null)
    },
    [decisions]
  )

  const clearRoute = useCallback((decision: Decision) => {
    setRoutes(current => {
      if (current[decision.id] === undefined) {
        return current
      }

      const next = { ...current }
      delete next[decision.id]

      return next
    })
  }, [])

  useImperativeHandle(
    ref,
    () => ({
      choose: index => {
        if (!active) {
          return
        }

        const route = routes[active.id]

        if (route !== undefined && active.options[route]?.next) {
          chooseFollowUp(active, route, index)
        } else {
          chooseBranch(active, index)
        }
      },
      hover: index => {
        setHovered(active && index !== null ? { branch: index, decision: active.id } : null)
      },
      next: () => move(1),
      peek: () => {
        if (active) {
          peek(active, activeIndex)
        }
      },
      previous: () => move(-1),
      send: () => {
        if (active) {
          sendOther(active)
        }
      },
      setOther: text => {
        if (active) {
          setOtherId(active.id)
          setOtherText(text)
        }
      }
    }),
    [active, activeIndex, chooseBranch, chooseFollowUp, move, peek, routes, sendOther]
  )

  // Opening "Other…" is a direct request for a field: put the caret in it.
  useEffect(() => {
    if (otherId && otherId === active?.id) {
      otherFieldRef.current?.focus()
    }
  }, [active?.id, otherId])

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) {
      return
    }

    const target = event.target as HTMLElement

    // A field owns its keys; "Other…" handles Enter and Esc itself.
    if (target.tagName === 'TEXTAREA' || target.tagName === 'INPUT') {
      return
    }

    if (event.key === 'Escape') {
      if (openNote || otherId) {
        event.preventDefault()
        event.stopPropagation()
        setOpenNote(null)
        setOtherId(null)

        return
      }

      // A woken follow-up is a step the user can still take back.
      if (active && routes[active.id] !== undefined) {
        event.preventDefault()
        event.stopPropagation()
        clearRoute(active)
      }

      return
    }

    if (/^[1-9]$/.test(event.key)) {
      // Picking needs a live decision. Moving between decisions and reading a
      // note stay available once one has settled — the receipt is still part of
      // the transcript you are navigating.
      if (!active || settled[active.id]) {
        return
      }

      const route = routes[active.id]
      const followUp = route === undefined ? undefined : active.options[route]?.next
      const options = followUp ? followUp.options : active.options
      const slot = Number(event.key) - 1

      if (slot < options.length) {
        event.preventDefault()

        if (followUp && route !== undefined) {
          chooseFollowUp(active, route, slot)
        } else {
          chooseBranch(active, slot)
        }

        return
      }

      if (!followUp && allowOther && slot === options.length) {
        event.preventDefault()
        setOtherId(active.id)
      }

      return
    }

    if (event.key === 'ArrowRight' || event.key === 'j' || event.key === 'J') {
      event.preventDefault()
      move(1)

      return
    }

    if (event.key === 'ArrowLeft' || event.key === 'k' || event.key === 'K') {
      event.preventDefault()
      move(-1)

      return
    }

    if ((event.key === 'f' || event.key === 'F') && active) {
      event.preventDefault()
      peek(active, activeIndex)
    }
  }

  const onOtherKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>, decision: Decision) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      setOtherId(null)

      return
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      sendOther(decision)
    }
  }

  const renderOption = (
    decision: Decision,
    decisionIndex: number,
    option: DecisionOption,
    index: number,
    branch: number,
    then?: number
  ) => {
    const state = settled[decision.id]
    const route = routes[decision.id]
    const woken = route !== undefined && route === branch
    const lit = hovered?.decision === decision.id && hovered.branch === branch && hovered.then === then
    const hoveredHere = Boolean(hovered) && hovered?.decision === decision.id

    const chosen =
      then === undefined ? state?.choice.index === index && !state.choice.then : state?.choice.then === option

    const keyPrefix = `${decisionIndex}-${branch}-${then ?? 'b'}`

    const openEffect =
      openNote?.decision === decision.id && openNote.key.startsWith(`${keyPrefix}-`)
        ? Number(openNote.key.slice(keyPrefix.length + 1))
        : null

    return (
      <OptionNode
        active={!state && (woken || (then === undefined && route === undefined))}
        chosen={Boolean(state) && chosen}
        dimmed={Boolean(state) && !chosen}
        index={index}
        key={`${keyPrefix}-${index}`}
        lit={lit}
        noteId={effectIndex => `${noteBase}-${keyPrefix}-${effectIndex}`}
        onChoose={() => {
          if (then === undefined) {
            chooseBranch(decision, index)
          } else {
            chooseFollowUp(decision, branch, index)
          }
        }}
        onHover={hoveredIndex => {
          if (hoveredIndex === null) {
            setHovered(null)
          } else {
            setHovered({ branch, decision: decision.id, then })
          }
        }}
        onPeek={effectIndex => {
          const key = `${keyPrefix}-${effectIndex}`

          setOpenNote(current =>
            current?.decision === decision.id && current.key === key ? null : { decision: decision.id, key }
          )
        }}
        openEffect={openEffect}
        option={option}
        order={index}
        rest={hoveredHere && !lit}
      />
    )
  }

  return (
    <div
      aria-label={title ?? copy.title}
      className={cn('dt-root grid gap-1 text-sm text-(--ui-text-primary)', className)}
      data-slot="decision-tree"
      onKeyDown={onKeyDown}
      role="group"
    >
      <div className="flex items-center gap-2 px-0.5 pb-1" data-slot="decision-tree-bar">
        <span className="text-[0.6875rem] font-medium tracking-wide text-(--ui-text-tertiary) uppercase">
          {title ?? copy.title}
        </span>
        <span className="ml-auto flex items-center gap-1.5 text-[0.6875rem] text-(--ui-text-tertiary)">
          {activeFlagKeys.length > 0 ? (
            <span className="flex items-center gap-1">
              <Kbd size="sm">F</Kbd>
              {copy.showNote}
            </span>
          ) : null}
          <span className="flex items-center gap-1">
            <Kbd size="sm">1</Kbd>
            {copy.pickBranch}
          </span>
        </span>
        {decisions.length > 1 ? (
          <span className="flex items-center gap-0.5">
            <span aria-hidden className="text-[0.6875rem] tabular-nums text-(--ui-text-tertiary)">
              {activeIndex + 1}/{decisions.length}
            </span>
            <Button
              aria-label={copy.previousDecision}
              disabled={activeIndex <= 0}
              onClick={() => move(-1)}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <ChevronLeft />
            </Button>
            <Button
              aria-label={copy.nextDecision}
              disabled={activeIndex >= decisions.length - 1}
              onClick={() => move(1)}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <ChevronRight />
            </Button>
          </span>
        ) : null}
      </div>

      {decisions.length === 0 ? <p className="px-0.5 text-xs text-(--ui-text-tertiary)">{copy.empty}</p> : null}

      {decisions.map((decision, decisionIndex) => {
        const state = settled[decision.id]
        const route = routes[decision.id]

        return (
          <div
            className="dt-decision"
            data-decision-id={decision.id}
            data-settled={state ? '' : undefined}
            data-slot="decision-tree-decision"
            key={decision.id}
          >
            <div className="dt-decision-head grid gap-0.5" data-slot="decision-tree-head">
              <div className="flex items-center gap-1.5" data-slot="decision-tree-meta">
                {decision.icon ? (
                  <span aria-hidden className="shrink-0 [&_svg]:size-3.5">
                    {decision.icon}
                  </span>
                ) : null}
                <span className="text-xs font-medium text-(--ui-text-secondary)">{decision.agent}</span>
                {decision.model ? <span className="text-xs text-(--ui-text-tertiary)">{decision.model}</span> : null}
                {decision.risk ? (
                  <span className={cn(ACTION_CLASS, RISK_CLASS[decision.risk])} data-slot="decision-tree-risk">
                    {copy.risk[decision.risk]}
                  </span>
                ) : null}
              </div>
              <p className="leading-snug font-medium wrap-anywhere" data-slot="decision-tree-question">
                {decision.question}
              </p>
              {decision.context ? (
                <p className="text-xs leading-relaxed text-(--ui-text-secondary)" data-slot="decision-tree-context">
                  {decision.context}
                </p>
              ) : null}
            </div>

            {state ? (
              <p
                className="flex items-center gap-1.5 text-xs text-(--ui-text-secondary)"
                data-slot="decision-tree-outcome"
              >
                <Check aria-hidden className="size-3.5 shrink-0 text-(--theme-primary)" />
                <span className="wrap-anywhere">
                  {state.answer ?? (state.choice.other ? copy.instructionSent : copy.chose(state.choice.option.label))}
                </span>
                {pendingId === decision.id ? <span className="text-(--ui-text-tertiary)">{copy.pending}</span> : null}
              </p>
            ) : null}

            <div className="dt-branches grid gap-0.5" data-slot="decision-tree-branches">
              {decision.options.map((option, index) => (
                <div className="grid gap-0.5" key={`${decision.id}-${index}`}>
                  {renderOption(decision, decisionIndex, option, index, index)}
                  {option.next ? (
                    <div
                      className="grid gap-1 pt-0.5 pb-1 pl-7"
                      data-dormant={route === index ? undefined : ''}
                      data-slot="decision-tree-follow-up"
                      inert={route !== index}
                    >
                      <p className="text-xs text-(--ui-text-tertiary)" data-slot="decision-tree-follow-up-question">
                        {option.next.question}
                      </p>
                      <div className="grid gap-0.5 border-l border-(--ui-stroke-tertiary) pl-3">
                        {option.next.options.map((then, thenIndex) => (
                          <div key={`${decision.id}-${index}-${thenIndex}`}>
                            {renderOption(decision, decisionIndex, then, thenIndex, index, thenIndex)}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              ))}

              {allowOther && !settled[decision.id] ? (
                <div className="dt-node" data-slot="decision-tree-other">
                  {otherId === decision.id ? (
                    <div className="grid gap-1 px-1.5 py-1">
                      <Textarea
                        aria-label={copy.other}
                        className="field-sizing-content max-h-24 min-h-0 resize-none text-xs"
                        data-slot="decision-tree-other-field"
                        onChange={event => setOtherText(event.target.value)}
                        onKeyDown={event => onOtherKeyDown(event, decision)}
                        placeholder={copy.otherPlaceholder}
                        ref={otherFieldRef}
                        rows={2}
                        value={otherText}
                      />
                      <span aria-hidden className="flex items-center gap-1 text-[0.625rem] text-(--ui-text-tertiary)">
                        <Kbd size="sm">↵</Kbd>
                        {copy.other}
                      </span>
                    </div>
                  ) : (
                    <button
                      aria-keyshortcuts={route === undefined ? String(decision.options.length + 1) : undefined}
                      className={cn(BRANCH_CLASS, 'text-(--ui-text-tertiary)')}
                      data-slot="decision-tree-other-trigger"
                      onClick={() => setOtherId(decision.id)}
                      tabIndex={route === undefined ? undefined : -1}
                      type="button"
                    >
                      <Kbd className="mt-px shrink-0" size="sm">
                        {decision.options.length + 1}
                      </Kbd>
                      <span className="min-w-0 flex-1">{copy.other}</span>
                    </button>
                  )}
                </div>
              ) : null}
            </div>
          </div>
        )
      })}

      <p aria-live="polite" className="sr-only" data-slot="decision-tree-live">
        {announcement}
      </p>
    </div>
  )
}
