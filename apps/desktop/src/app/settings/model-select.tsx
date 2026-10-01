import type { ModelOptionProvider } from '@hermes/shared'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectSeparator, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { Plus, X } from '@/lib/icons'
import { isSubmitEnter } from '@/lib/ime'
import { cn } from '@/lib/utils'
import { addCustomModel, customModelSlug } from '@/store/custom-models'

import { CONTROL_TEXT } from './constants'

// Radix <Select> renders a blank trigger when `value` matches no <SelectItem>.
// A custom model (e.g. one added via config that isn't in the provider's
// curated list) would vanish — surface the active value so it stays selectable.
export const withActive = (models: readonly string[], active: string): readonly string[] =>
  active && !models.includes(active) ? [active, ...models] : models

// Radix <Select> items cannot be typed into; an action row swaps the control
// for a text field. Model ids live under their own prefix so no slug — this
// feature admits any whitespace-free string — can alias the action's value.
const CUSTOM_ITEM = 'custom'
const MODEL_PREFIX = 'model:'

export const toItemValue = (model: string): string => (model ? MODEL_PREFIX + model : '')

/** The model id behind a Radix item value; `null` for the custom action row. */
export const fromItemValue = (item: string): string | null =>
  item.startsWith(MODEL_PREFIX) ? item.slice(MODEL_PREFIX.length) : null

interface ModelSelectProps {
  'aria-label'?: string
  className?: string
  /** The provider's catalog row when known: the typed id is remembered under
   *  it and made visible in the pickers. Absent (provider not in the catalog)
   *  the id is still applied, just not remembered. */
  provider?: ModelOptionProvider
  /** Provider slug the id belongs to. Empty while none is picked. */
  providerSlug: string
  models: readonly string[]
  onValueChange: (model: string) => void
  value: string
}

/**
 * A model <Select> with a "Custom model…" row that turns into an Input, so a
 * slug the provider's catalog lacks can be typed in place. Every keystroke
 * reaches `onValueChange` (the parent's Apply/autosave reads live state); the
 * id is remembered as a custom model when the field commits (Enter/blur).
 */
export function ModelSelect({
  'aria-label': ariaLabel,
  className,
  models,
  onValueChange,
  provider,
  providerSlug,
  value
}: ModelSelectProps) {
  const { t } = useI18n()
  const m = t.settings.model
  const [typing, setTyping] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (typing) {
      inputRef.current?.focus()
      inputRef.current?.select()
    }
  }, [typing])

  // A provider change swaps the catalog; drop back to the list for it.
  useEffect(() => {
    setTyping(false)
  }, [providerSlug])

  const remember = () => {
    const slug = customModelSlug(value)

    if (!slug || !providerSlug) {
      return
    }

    if (slug !== value) {
      onValueChange(slug)
    }

    if (!models.includes(slug)) {
      addCustomModel(providerSlug, slug, provider)
    }
  }

  if (typing) {
    return (
      <Input
        aria-label={ariaLabel}
        className={cn(CONTROL_TEXT, 'font-mono')}
        containerClassName={className}
        onBlur={remember}
        onChange={event => onValueChange(event.target.value)}
        onKeyDown={event => {
          if (isSubmitEnter(event)) {
            remember()
            setTyping(false)
          } else if (event.key === 'Escape') {
            setTyping(false)
          }
        }}
        placeholder={m.customModelPlaceholder}
        ref={inputRef}
        suffix={
          <Button
            aria-label={m.chooseFromList}
            className="pointer-events-auto -my-1.5 -mr-1.5 text-muted-foreground"
            onClick={() => setTyping(false)}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <X />
          </Button>
        }
        value={value}
      />
    )
  }

  return (
    <Select
      onValueChange={next => {
        const model = fromItemValue(next)

        if (model === null) {
          setTyping(true)
        } else {
          onValueChange(model)
        }
      }}
      value={toItemValue(value)}
    >
      <SelectTrigger aria-label={ariaLabel} className={cn(className, CONTROL_TEXT)}>
        <SelectValue placeholder={m.model} />
      </SelectTrigger>
      <SelectContent>
        {withActive(models, value).map(model => (
          <SelectItem key={model} value={toItemValue(model)}>
            {model}
          </SelectItem>
        ))}
        <SelectSeparator />
        <SelectItem className="text-muted-foreground" value={CUSTOM_ITEM}>
          <span className="inline-flex items-center gap-1.5">
            <Plus className="size-3" />
            {m.customModel}
          </span>
        </SelectItem>
      </SelectContent>
    </Select>
  )
}
