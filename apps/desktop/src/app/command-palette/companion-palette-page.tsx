/**
 * Cmd-K "Companions…" page — browse the companiondex gallery, adopt/switch, toggle off.
 *
 * A thin view over the `companion-gallery` store: it subscribes to the shared atoms
 * and calls the store's actions. The store owns fetching, caching, the thumb
 * cache, and optimistic mutations, so reopening this page is instant and a
 * toggle never re-pulls the network gallery.
 */

import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import { StatusRow } from '@/app/command-palette/status-row'
import { HUD_ITEM, HUD_TEXT } from '@/app/floating-hud'
import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { CompanionThumb } from '@/components/companion/companion-thumb'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Egg, Loader2, PawPrint } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $companionBusy,
  $companionGallery,
  $companionGalleryError,
  $companionGalleryStatus,
  adoptCompanion,
  loadCompanionGallery,
  loadCompanionThumb,
  rankedGalleryCompanions,
  setCompanionEnabled
} from '@/store/companion-gallery'

interface CompanionPalettePageProps {
  search: string
  /** Navigate to the "generate a companion" page (rendered as a header action). */
  onGenerate?: () => void
}

export function CompanionPalettePage({ search, onGenerate }: CompanionPalettePageProps) {
  const { t } = useI18n()
  const copy = t.commandCenter.companions
  const { requestGateway } = useGatewayRequest()

  const gallery = useStore($companionGallery)
  const status = useStore($companionGalleryStatus)
  const error = useStore($companionGalleryError)
  const busy = useStore($companionBusy)

  useEffect(() => {
    void loadCompanionGallery(requestGateway)
  }, [requestGateway])

  const enabled = gallery?.enabled ?? false
  const active = gallery?.active ?? ''

  const shown = useMemo(() => rankedGalleryCompanions(gallery, search).slice(0, 50), [gallery, search])

  const adopt = (slug: string) => {
    void adoptCompanion(requestGateway, slug, copy.adoptFailed).then(ok => ok && triggerHaptic('crisp'))
  }

  if (status === 'loading' && !gallery) {
    return <StatusRow icon={<Loader2 className="size-3.5 animate-spin" />} text={copy.loading} />
  }

  if (status === 'stale') {
    return <StatusRow text={copy.staleBackend} tone="error" />
  }

  if (!gallery?.companions.length && error) {
    return <StatusRow text={error} tone="error" />
  }

  const mutating = Boolean(busy)

  return (
    <div role="listbox">
      {onGenerate && (
        <button
          className={cn(
            'flex w-full items-center gap-2 rounded-md text-left text-foreground transition-colors hover:bg-(--chrome-action-hover)',
            HUD_ITEM,
            HUD_TEXT
          )}
          onClick={onGenerate}
          onMouseDown={event => event.preventDefault()}
          type="button"
        >
          <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-(--chrome-action-hover)">
            <Egg className="size-4" />
          </span>
          <span className="font-medium">{t.commandCenter.generateCompanion.title}</span>
        </button>
      )}

      {error && <p className="px-2 pb-1 pt-1.5 text-[0.6875rem] text-(--ui-red)">{error}</p>}

      {shown.length === 0 ? (
        <StatusRow text={copy.empty} />
      ) : (
        shown.map(companion => {
          const isActive = enabled && companion.slug === active
          const isBusy = busy === companion.slug

          return (
            <button
              className={cn(
                'flex w-full items-center gap-2 rounded-md text-left transition-colors hover:bg-(--chrome-action-hover) disabled:opacity-60',
                HUD_ITEM,
                HUD_TEXT,
                isActive && 'bg-(--chrome-action-hover)/70'
              )}
              disabled={mutating && !isBusy}
              key={companion.slug}
              onClick={() => adopt(companion.slug)}
              onMouseDown={event => event.preventDefault()}
              role="option"
              type="button"
            >
              <CompanionThumb
                alt={companion.displayName}
                load={(slug, url) => loadCompanionThumb(requestGateway, slug, url)}
                size={32}
                slug={companion.slug}
                url={companion.spritesheetUrl}
              />
              <span className="flex min-w-0 flex-col">
                <span className="flex items-center gap-1.5">
                  <span className="truncate font-medium">{companion.displayName}</span>
                  {companion.generated && (
                    <span className="shrink-0 rounded-full bg-primary/15 px-1.5 py-px text-[0.625rem] font-medium text-primary">
                      {copy.generatedTag}
                    </span>
                  )}
                </span>
                <span className="truncate text-[0.6875rem] text-muted-foreground/80">
                  {companion.slug}
                  {companion.installed ? ` · ${copy.installed}` : ''}
                </span>
              </span>
              <span className="ml-auto flex shrink-0 items-center text-[0.6875rem] text-muted-foreground">
                {isBusy ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : isActive ? (
                  <Check className="size-3.5 text-foreground" />
                ) : null}
              </span>
            </button>
          )
        })
      )}
    </div>
  )
}

/**
 * Single on/off toggle, rendered inline on the palette's search row (see
 * `CommandInput`'s `right` slot). The paw lights up when companions are on. Reads the
 * same shared gallery atoms, so it stays in sync with the list below.
 */
export function CompanionInlineToggle() {
  const { t } = useI18n()
  const copy = t.commandCenter.companions
  const { requestGateway } = useGatewayRequest()
  const gallery = useStore($companionGallery)
  const busy = useStore($companionBusy)

  if (!gallery) {
    return null
  }

  const enabled = gallery.enabled

  const toggle = () => {
    void setCompanionEnabled(requestGateway, !enabled, {
      noneAvailable: copy.noneAvailable,
      fallback: copy.toggleFailed(!enabled)
    }).then(ok => ok && triggerHaptic('crisp'))
  }

  return (
    <button
      aria-label={enabled ? copy.turnOff : copy.turnOn}
      aria-pressed={enabled}
      className={cn(
        'flex shrink-0 items-center justify-center rounded-md p-1.5 transition-colors disabled:opacity-50',
        enabled
          ? 'bg-(--chrome-action-hover) text-foreground'
          : 'text-muted-foreground hover:bg-(--chrome-action-hover)/60'
      )}
      disabled={Boolean(busy)}
      onClick={toggle}
      // Don't steal focus from the search input on click.
      onMouseDown={event => event.preventDefault()}
      type="button"
    >
      {busy ? <Loader2 className="size-4 animate-spin" /> : <PawPrint className="size-4" />}
    </button>
  )
}
