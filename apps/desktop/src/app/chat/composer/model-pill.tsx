import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { ModelMenuCloseContext } from '@/app/shell/model-menu-panel'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { releaseTypingFocus } from '@/components/ui/keyboard-first'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { compactNumber } from '@/lib/format'
import { ChevronDown } from '@/lib/icons'
import { formatModelStatusLabel } from '@/lib/model-status-label'
import { cn } from '@/lib/utils'
import { $currentModelSource, $defaultReasoningEffort, setModelPickerOpen } from '@/store/session'
import type { UsageStats } from '@/types/hermes'

import { onComposerModelMenuRequest } from './focus'
import { useComposerScope } from './scope'
import type { ChatBarState } from './types'

const PILL = cn(
  'h-(--composer-control-size) max-w-40 shrink-0 gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

const CONTEXT_RING_RADIUS = 5.5
const CONTEXT_RING_CIRCUMFERENCE = 2 * Math.PI * CONTEXT_RING_RADIUS

export interface ContextWindowUsage {
  max: number
  percent: number
  used: number
}

/** Normalizes the context data reported by the gateway for the compact ring. */
export function contextWindowUsage(usage: null | UsageStats | undefined): ContextWindowUsage | null {
  const max = usage?.context_max

  if (typeof max !== 'number' || !Number.isFinite(max) || max <= 0) {
    return null
  }

  const reportedPercent = usage?.context_percent
  const hasReportedPercent = typeof reportedPercent === 'number' && Number.isFinite(reportedPercent)
  const used = usage?.context_used
  const hasUsed = typeof used === 'number' && Number.isFinite(used)

  if (!hasReportedPercent && !hasUsed) {
    return null
  }

  const normalizedUsed = Math.max(0, Math.min(max, Math.round(hasUsed ? used : max * ((reportedPercent ?? 0) / 100))))

  const normalizedPercent = Math.max(
    0,
    Math.min(100, hasReportedPercent ? reportedPercent : (normalizedUsed / max) * 100)
  )

  return { max, percent: normalizedPercent, used: normalizedUsed }
}

/** Returns the remaining model context only when the gateway reported a real window. */
export function contextTokensRemaining(usage: null | UsageStats | undefined): null | number {
  const context = contextWindowUsage(usage)

  if (!context) {
    return null
  }

  return Math.max(0, Math.round(context.max - context.used))
}

/**
 * Composer model selector — the relocated status-bar pill. Reuses the live
 * `model.options` dropdown (`modelMenuContent`) verbatim; falls back to the
 * full picker when the gateway is closed and no live menu exists.
 *
 * Display follows THIS surface's SessionView (primary or tile) — never the
 * primary-only globals — so side-by-side panes each show their own model.
 */
export function ModelPill({
  compact = false,
  disabled,
  model
}: {
  compact?: boolean
  disabled: boolean
  model: ChatBarState['model']
}) {
  const { t } = useI18n()
  const copy = t.shell.statusbar
  const view = useSessionView()
  // Prefer the chat-bar snapshot (already view-scoped by ChatView); fall back
  // to the live SessionView atoms so a mid-flight session.info still paints.
  const viewModel = useStore(view.$model)
  const viewProvider = useStore(view.$provider)
  const currentModel = model.model || viewModel
  const currentProvider = model.provider || viewProvider
  const fastMode = useStore(view.$fast)
  const reasoningEffort = useStore(view.$reasoningEffort)
  const modelSource = useStore($currentModelSource)
  const defaultEffort = useStore($defaultReasoningEffort)
  const runtimeId = useStore(view.$runtimeId)
  const usage = useStore(view.$usage)
  const { requestGateway } = useGatewayRequest()
  const [requestedUsage, setRequestedUsage] = useState<{ runtimeId: string; usage: UsageStats } | null>(null)
  const [open, setOpen] = useState(false)
  const scope = useComposerScope()
  const hasLiveMenu = Boolean(model.modelMenuContent)
  const reportedContext = contextWindowUsage(usage)
  const hasReportedContext = Boolean(reportedContext)

  // `session.info` normally includes its context window after a turn. On a
  // resumed chat that metadata can arrive a little later, though. Fetch its
  // lightweight usage snapshot once in the meantime so the ring never
  // depends on opening the status-bar's detailed context popover.
  useEffect(() => {
    if (compact || !runtimeId || hasReportedContext) {
      return
    }

    let cancelled = false

    void requestGateway<UsageStats>('session.usage', { session_id: runtimeId })
      .then(nextUsage => {
        if (!cancelled) {
          setRequestedUsage({ runtimeId, usage: nextUsage })
        }
      })
      .catch(() => undefined)

    return () => {
      cancelled = true
    }
  }, [compact, hasReportedContext, requestGateway, runtimeId, usage?.total])

  // The `composer.modelPicker` hotkey, routed to exactly one surface (the pane
  // under the pointer, else the active composer — see requestModelMenuToggle).
  // Toggles the live dropdown; with no live menu (gateway closed) it opens the
  // full picker dialog, same as clicking the pill.
  useEffect(
    () =>
      onComposerModelMenuRequest(target => {
        if (target !== scope.target || disabled) {
          return
        }

        if (hasLiveMenu) {
          setOpen(prev => !prev)
        } else {
          setModelPickerOpen(true)
        }
      }),
    [scope.target, disabled, hasLiveMenu]
  )

  // The composer pick is sticky: a manual selection is pinned and every NEW
  // chat uses it instead of the Settings → Model default — silently, which has
  // cost users real money on a forgotten paid-model pick (#62055). Surface the
  // pin whenever a draft (no live session) is running on a manual override. A
  // live session's footer reflects that session's model, so no badge there.
  // Tiles always have a runtime — pin badge is primary-draft only.
  const pinnedOverride =
    view.kind === 'primary' && !runtimeId && modelSource === 'manual' && Boolean(currentModel.trim())

  // The model resolves a beat after the gateway/session comes up. Rather than
  // flash a literal "No model", show a quiet loader (inherits the pill text
  // color at half opacity) until a model lands.
  const label = compact ? (
    <ChevronDown className="size-3.5 shrink-0 opacity-70" />
  ) : (
    <>
      {currentModel.trim() ? (
        <span className="truncate">
          {formatModelStatusLabel(currentModel, { defaultEffort, fastMode, reasoningEffort })}
        </span>
      ) : (
        <GlyphSpinner className="opacity-50" spinner="braille" />
      )}
      {pinnedOverride && (
        <span
          aria-label={copy.modelPinned}
          className="size-1 shrink-0 rounded-full bg-(--ui-accent)"
          data-testid="model-pinned-dot"
          role="img"
        />
      )}
      <ChevronDown className="size-2.5 shrink-0 opacity-50" />
    </>
  )

  // Compact (floating composer): a snug square holding just the chevron — no pill
  // padding, sized to match the other composer icon buttons.
  const pillClass = compact
    ? cn(
        'size-(--composer-control-size) shrink-0 justify-center gap-0 rounded-md p-0',
        'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
      )
    : PILL

  const baseTitle = currentProvider
    ? copy.modelTitle(currentProvider, currentModel || copy.modelNone)
    : copy.switchModel

  const title = pinnedOverride ? `${baseTitle} — ${copy.modelPinned}` : baseTitle
  const requestedContext = requestedUsage?.runtimeId === runtimeId ? contextWindowUsage(requestedUsage.usage) : null
  const context = compact ? null : (reportedContext ?? requestedContext)
  const contextPercent = Math.round(context?.percent ?? 0)

  const contextDetail = context
    ? {
        percentFull: t.composer.contextWindowFull(contextPercent),
        tokensUsed: t.composer.contextWindowUsage(compactNumber(context.used), compactNumber(context.max))
      }
    : null

  const contextAriaLabel = contextDetail
    ? `${t.composer.contextWindow}. ${contextDetail.percentFull}. ${contextDetail.tokensUsed}.`
    : t.composer.contextWindowUnavailable

  const contextTooltip = contextDetail ? (
    <span className="flex flex-col items-center whitespace-nowrap text-center font-normal">
      <span>{t.composer.contextWindow}</span>
      <span>{contextDetail.percentFull}</span>
      <span>{contextDetail.tokensUsed}</span>
    </span>
  ) : (
    contextAriaLabel
  )

  const contextRingOffset = CONTEXT_RING_CIRCUMFERENCE * (1 - contextPercent / 100)

  const contextCounter =
    !compact && currentModel.trim() ? (
      <Tip label={contextTooltip} side="top">
        <span
          aria-label={contextAriaLabel}
          className={cn(
            'inline-flex h-(--composer-control-size) shrink-0 cursor-default items-center justify-center rounded-full text-(--ui-text-tertiary)',
            context && 'text-(--ui-text-secondary)'
          )}
          data-testid="context-window"
        >
          <svg aria-hidden="true" className="size-3.5 -rotate-90" fill="none" viewBox="0 0 16 16">
            <circle
              className="opacity-25"
              cx="8"
              cy="8"
              r={CONTEXT_RING_RADIUS}
              stroke="currentColor"
              strokeWidth="2"
            />
            {context && (
              <circle
                cx="8"
                cy="8"
                r={CONTEXT_RING_RADIUS}
                stroke="currentColor"
                strokeDasharray={CONTEXT_RING_CIRCUMFERENCE}
                strokeDashoffset={contextRingOffset}
                strokeLinecap="round"
                strokeWidth="2"
                style={{ transition: 'stroke-dashoffset 300ms ease-out' }}
              />
            )}
          </svg>
        </span>
      </Tip>
    ) : null

  if (!model.modelMenuContent) {
    return (
      <div className="flex shrink-0 items-center gap-1">
        {contextCounter}
        <Tip label={pinnedOverride ? `${copy.openModelPicker} — ${copy.modelPinned}` : copy.openModelPicker} side="top">
          <Button
            aria-label={copy.openModelPicker}
            className={pillClass}
            disabled={disabled}
            onClick={() => setModelPickerOpen(true)}
            type="button"
            variant="ghost"
          >
            {label}
          </Button>
        </Tip>
      </div>
    )
  }

  // Closing the menu ends its claim on the keyboard: Radix restores focus to
  // this pill (a toolbar button), so without the release the Enter that
  // committed a model also swallows whatever you type next.
  const setMenuOpen = (next: boolean) => {
    setOpen(next)

    if (!next) {
      releaseTypingFocus()
    }
  }

  return (
    <div className="flex shrink-0 items-center gap-1">
      {contextCounter}
      <DropdownMenu onOpenChange={setMenuOpen} open={open}>
        <Tip label={title} side="top">
          <DropdownMenuTrigger asChild>
            <Button aria-label={title} className={pillClass} disabled={disabled} type="button" variant="ghost">
              {label}
            </Button>
          </DropdownMenuTrigger>
        </Tip>
        <DropdownMenuContent align="end" className="w-64 p-0" side="top" sideOffset={8}>
          <ModelMenuCloseContext.Provider value={() => setMenuOpen(false)}>
            {model.modelMenuContent}
          </ModelMenuCloseContext.Provider>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
