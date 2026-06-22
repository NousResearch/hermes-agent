import { useStore } from '@nanostores/react'
import { type ReactElement, useEffect, useRef, useState } from 'react'

import { ModelMenuCloseContext } from '@/app/shell/model-menu-panel'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { formatModelStatusLabel } from '@/lib/model-status-label'
import { cn } from '@/lib/utils'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  setModelPickerOpen
} from '@/store/session'

import { CodexQuotaCard } from './codex-quota-card'
import type { ChatBarState } from './types'

const PILL = cn(
  'h-(--composer-control-size) max-w-40 shrink-0 gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

const CODEX_PROVIDER = 'openai-codex'
const QUOTA_CLOSE_DELAY_MS = 120

/**
 * Composer model selector — the relocated status-bar pill. Reuses the live
 * `model.options` dropdown (`modelMenuContent`) verbatim; falls back to the
 * full picker when the gateway is closed and no live menu exists.
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
  const copy = useI18n().t.shell.statusbar
  const currentModel = useStore($currentModel)
  const currentProvider = useStore($currentProvider)
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const fastMode = useStore($currentFastMode)
  const reasoningEffort = useStore($currentReasoningEffort)
  const [open, setOpen] = useState(false)
  const [quotaOpen, setQuotaOpen] = useState(false)
  const quotaCloseTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // The model resolves a beat after the gateway/session comes up. Rather than
  // flash a literal "No model", show a quiet loader (inherits the pill text
  // color at half opacity) until a model lands.
  const label = compact ? (
    <ChevronDown className="size-3.5 shrink-0 opacity-70" />
  ) : (
    <>
      {currentModel.trim() ? (
        <span className="truncate">{formatModelStatusLabel(currentModel, { fastMode, reasoningEffort })}</span>
      ) : (
        <GlyphSpinner className="opacity-50" spinner="braille" />
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

  const title = currentProvider ? copy.modelTitle(currentProvider, currentModel || copy.modelNone) : copy.switchModel
  const normalizedProvider = currentProvider.trim().toLowerCase()
  const quotaProfile = activeGatewayProfile.trim() || 'default'
  const quotaSupported = normalizedProvider === CODEX_PROVIDER
  const quotaEnabled = quotaSupported && !disabled && !open

  const clearQuotaCloseTimer = () => {
    if (quotaCloseTimer.current) {
      clearTimeout(quotaCloseTimer.current)
      quotaCloseTimer.current = null
    }
  }

  const openQuota = () => {
    if (!quotaEnabled) {
      return
    }

    clearQuotaCloseTimer()
    setQuotaOpen(true)
  }

  const closeQuotaSoon = () => {
    clearQuotaCloseTimer()
    quotaCloseTimer.current = setTimeout(() => {
      setQuotaOpen(false)
      quotaCloseTimer.current = null
    }, QUOTA_CLOSE_DELAY_MS)
  }

  useEffect(
    () => () => {
      clearQuotaCloseTimer()
    },
    []
  )

  const handleMenuOpenChange = (nextOpen: boolean) => {
    setOpen(nextOpen)

    if (nextOpen) {
      clearQuotaCloseTimer()
      setQuotaOpen(false)
    }
  }

  const withQuotaHover = (content: ReactElement) => {
    if (!quotaSupported) {
      return content
    }

    return (
      <Popover open={quotaOpen && quotaEnabled}>
        <PopoverAnchor asChild>
          <span
            className="inline-flex"
            onBlur={closeQuotaSoon}
            onFocus={openQuota}
            onMouseEnter={openQuota}
            onMouseLeave={closeQuotaSoon}
          >
            {content}
          </span>
        </PopoverAnchor>
        {quotaOpen && quotaEnabled && (
          <PopoverContent
            align="end"
            className="w-auto p-0"
            onMouseEnter={openQuota}
            onMouseLeave={closeQuotaSoon}
            side="top"
            sideOffset={8}
          >
            <CodexQuotaCard enabled profile={quotaProfile} provider={currentProvider} />
          </PopoverContent>
        )}
      </Popover>
    )
  }

  if (!model.modelMenuContent) {
    return withQuotaHover(
      <Button
        aria-label={copy.openModelPicker}
        className={pillClass}
        disabled={disabled}
        onClick={() => setModelPickerOpen(true)}
        title={copy.openModelPicker}
        type="button"
        variant="ghost"
      >
        {label}
      </Button>
    )
  }

  return withQuotaHover(
    <DropdownMenu onOpenChange={handleMenuOpenChange} open={open}>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={title}
          className={pillClass}
          disabled={disabled}
          title={title}
          type="button"
          variant="ghost"
        >
          {label}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64 p-0" side="top" sideOffset={8}>
        <ModelMenuCloseContext.Provider value={() => setOpen(false)}>
          {model.modelMenuContent}
        </ModelMenuCloseContext.Provider>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
