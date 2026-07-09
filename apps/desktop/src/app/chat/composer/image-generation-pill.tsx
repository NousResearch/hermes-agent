import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Tip } from '@/components/ui/tooltip'
import {
  getImageGenerationOptions,
  type ImageGenerationModelOption,
  type ImageGenerationProviderOption,
  selectImageGeneration
} from '@/hermes'
import { useI18n } from '@/i18n'
import { Check, ChevronDown, iconSize, ImageIcon } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'

const PILL = cn(
  'h-(--composer-control-size) max-w-48 shrink-0 gap-1.5 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground',
  'data-[active=true]:bg-(--chrome-action-hover) data-[active=true]:text-foreground'
)

function modelCapabilityLabel(model: ImageGenerationModelOption, copy: ReturnType<typeof useI18n>['t']['composer']['imageGeneration']): string {
  return model.modalities.includes('image') ? copy.imageEditing : copy.textOnly
}

function fallbackModel(provider: ImageGenerationProviderOption): string | undefined {
  return provider.models[0]?.id || provider.default_model || undefined
}

/** Composer image-generation selector — mirrors the model pill for image_gen. */
export function ImageGenerationPill({ compact = false, disabled }: { compact?: boolean; disabled: boolean }) {
  const { t } = useI18n()
  const copy = t.composer.imageGeneration
  const [open, setOpen] = useState(false)
  const [saving, setSaving] = useState(false)

  const query = useQuery({
    queryKey: ['image-generation-options'],
    queryFn: getImageGenerationOptions,
    staleTime: 30_000
  })

  const selected = useMemo(() => {
    const data = query.data
    const provider = data?.providers.find(item => item.id === data.provider_id || item.name === data.provider)
    const model = provider?.models.find(item => item.id === data?.model)

    return { provider, model }
  }, [query.data])

  const choose = async (provider: ImageGenerationProviderOption, model?: string) => {
    if (saving) {
      return
    }

    setSaving(true)

    try {
      await selectImageGeneration(provider.id, model || fallbackModel(provider))
      await query.refetch()
      setOpen(false)
    } catch (error) {
      notifyError(error, copy.switchFailed)
    } finally {
      setSaving(false)
    }
  }

  const activeLabel = selected.model?.display || query.data?.model || selected.provider?.name || copy.title
  const title = selected.provider ? `${copy.title}: ${selected.provider.name}${query.data?.model ? ` · ${query.data.model}` : ''}` : copy.title

  const pillClass = compact
    ? cn(
        'size-(--composer-control-size) shrink-0 justify-center gap-0 rounded-md p-0',
        'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
      )
    : PILL

  const label = compact ? (
    <ImageIcon className={iconSize.sm} />
  ) : (
    <>
      <ImageIcon className={iconSize.sm} />
      <span className="truncate">{query.isLoading ? copy.loading : activeLabel}</span>
      {saving || query.isFetching ? (
        <GlyphSpinner className="opacity-60" spinner="braille" />
      ) : (
        <ChevronDown className="size-2.5 shrink-0 opacity-50" />
      )}
    </>
  )

  return (
    <DropdownMenu onOpenChange={setOpen} open={open}>
      <Tip label={title} side="top">
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={title}
            className={pillClass}
            data-active={selected.provider ? 'true' : 'false'}
            disabled={disabled}
            type="button"
            variant="ghost"
          >
            {label}
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end" className="w-80 p-0" side="top" sideOffset={8}>
        <div className="px-2.5 py-2">
          <div className="flex items-center gap-2 text-xs font-medium text-foreground">
            <ImageIcon className={iconSize.sm} />
            <span>{copy.title}</span>
          </div>
          {query.data?.requires_new_session && (
            <p className="mt-1 text-[0.6875rem] leading-snug text-amber-500">{copy.restartRequired}</p>
          )}
        </div>
        <DropdownMenuSeparator />
        {query.isLoading ? (
          <DropdownMenuItem disabled>
            <GlyphSpinner spinner="braille" />
            <span>{copy.loading}</span>
          </DropdownMenuItem>
        ) : !query.data?.providers.length ? (
          <DropdownMenuItem disabled>{copy.noProviders}</DropdownMenuItem>
        ) : (
          query.data.providers.map(provider => {
            const selectedProvider = selected.provider?.id === provider.id

            if (!provider.models.length) {
              return (
                <DropdownMenuItem
                  disabled={saving}
                  key={provider.id}
                  onSelect={() => {
                    void choose(provider)
                  }}
                >
                  <ProviderRow active={selectedProvider} copy={copy} provider={provider} />
                </DropdownMenuItem>
              )
            }

            return (
              <DropdownMenuSub key={provider.id}>
                <DropdownMenuSubTrigger disabled={saving}>
                  <ProviderRow active={selectedProvider} copy={copy} provider={provider} />
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="w-72">
                  <DropdownMenuLabel>{provider.name}</DropdownMenuLabel>
                  {provider.models.map(model => {
                    const active = selectedProvider && model.id === query.data?.model

                    return (
                      <DropdownMenuItem
                        disabled={saving}
                        key={model.id}
                        onSelect={() => {
                          void choose(provider, model.id)
                        }}
                      >
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-xs">{model.display}</span>
                          <span className="block truncate text-[0.625rem] text-(--ui-text-tertiary)">
                            {[modelCapabilityLabel(model, copy), model.speed, model.price].filter(Boolean).join(' · ')}
                          </span>
                        </span>
                        {active && <Check className={iconSize.sm} />}
                      </DropdownMenuItem>
                    )
                  })}
                </DropdownMenuSubContent>
              </DropdownMenuSub>
            )
          })
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function ProviderRow({
  active,
  copy,
  provider
}: {
  active: boolean
  copy: ReturnType<typeof useI18n>['t']['composer']['imageGeneration']
  provider: ImageGenerationProviderOption
}) {
  return (
    <span className="flex min-w-0 flex-1 items-center gap-2">
      <span className="min-w-0 flex-1">
        <span className="flex min-w-0 items-center gap-1.5">
          <span className="truncate text-xs">{provider.name}</span>
          {provider.badge && <span className="shrink-0 rounded bg-muted px-1 text-[0.5625rem] text-muted-foreground">{provider.badge}</span>}
        </span>
        <span className="block truncate text-[0.625rem] text-(--ui-text-tertiary)">
          {provider.available ? provider.tag : copy.setupRequired}
        </span>
      </span>
      {active && (
        <span className="inline-flex items-center gap-1 text-[0.625rem] text-foreground">
          <Check className={iconSize.sm} />
          <span>{copy.active}</span>
        </span>
      )}
    </span>
  )
}
