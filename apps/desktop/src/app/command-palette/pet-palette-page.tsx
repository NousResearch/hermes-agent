/**
 * Cmd-K "Pets…" page — browse the petdex gallery, adopt/switch, toggle off.
 *
 * A thin view over the `pet-gallery` store: it subscribes to the shared atoms
 * and calls the store's actions. The store owns fetching, caching, the thumb
 * cache, and optimistic mutations, so reopening this page is instant and a
 * toggle never re-pulls the network gallery.
 */

import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import { HUD_ITEM, HUD_TEXT } from '@/app/floating-hud'
import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { PetThumb } from '@/components/pet/pet-thumb'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Loader2, PawPrint } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $petBusy,
  $petGallery,
  $petGalleryError,
  $petGalleryStatus,
  adoptPet,
  loadPetGallery,
  loadPetThumb,
  rankedGalleryPets,
  setPetEnabled
} from '@/store/pet-gallery'

interface PetPalettePageProps {
  search: string
}

export function PetPalettePage({ search }: PetPalettePageProps) {
  const { t } = useI18n()
  const copy = t.commandCenter.pets
  const { requestGateway } = useGatewayRequest()

  const gallery = useStore($petGallery)
  const status = useStore($petGalleryStatus)
  const error = useStore($petGalleryError)
  const busy = useStore($petBusy)

  useEffect(() => {
    void loadPetGallery(requestGateway)
  }, [requestGateway])

  const enabled = gallery?.enabled ?? false
  const active = gallery?.active ?? ''

  const shown = useMemo(() => rankedGalleryPets(gallery, search).slice(0, 50), [gallery, search])

  const adopt = (slug: string) => {
    void adoptPet(requestGateway, slug, copy.adoptFailed).then(ok => ok && triggerHaptic('crisp'))
  }

  if (status === 'loading' && !gallery) {
    return <Status icon={<Loader2 className="size-3.5 animate-spin" />} text={copy.loading} />
  }

  if (status === 'stale') {
    return <Status text={copy.staleBackend} tone="error" />
  }

  if (!gallery?.pets.length && error) {
    return <Status text={error} tone="error" />
  }

  const mutating = Boolean(busy)

  return (
    <div role="listbox">
      {error && <p className="px-2 pb-1 pt-1.5 text-[0.6875rem] text-(--ui-red)">{error}</p>}

      {shown.length === 0 ? (
        <Status text={copy.empty} />
      ) : (
        shown.map(pet => {
          const isActive = enabled && pet.slug === active
          const isBusy = busy === pet.slug

          return (
            <button
              className={cn(
                'flex w-full items-center gap-2 rounded-md text-left transition-colors hover:bg-(--chrome-action-hover) disabled:opacity-60',
                HUD_ITEM,
                HUD_TEXT,
                isActive && 'bg-(--chrome-action-hover)/70'
              )}
              disabled={mutating && !isBusy}
              key={pet.slug}
              onClick={() => adopt(pet.slug)}
              onMouseDown={event => event.preventDefault()}
              role="option"
              type="button"
            >
              <PetThumb
                alt={pet.displayName}
                load={(slug, url) => loadPetThumb(requestGateway, slug, url)}
                size={32}
                slug={pet.slug}
                url={pet.spritesheetUrl}
              />
              <span className="flex min-w-0 flex-col">
                <span className="truncate font-medium">{pet.displayName}</span>
                <span className="truncate text-[0.6875rem] text-muted-foreground/80">
                  {pet.slug}
                  {pet.installed ? ` · ${copy.installed}` : ''}
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
 * `CommandInput`'s `right` slot). The paw lights up when pets are on. Reads the
 * same shared gallery atoms, so it stays in sync with the list below.
 */
export function PetInlineToggle() {
  const { t } = useI18n()
  const copy = t.commandCenter.pets
  const { requestGateway } = useGatewayRequest()
  const gallery = useStore($petGallery)
  const busy = useStore($petBusy)

  if (!gallery) {
    return null
  }

  const enabled = gallery.enabled

  const toggle = () => {
    void setPetEnabled(requestGateway, !enabled, {
      noneAvailable: copy.noneAvailable,
      fallback: copy.toggleFailed
    }).then(ok => ok && triggerHaptic('crisp'))
  }

  return (
    <button
      aria-label={enabled ? copy.turnOff : copy.turnOn}
      aria-pressed={enabled}
      className={cn(
        'flex shrink-0 items-center justify-center rounded-md p-1.5 transition-colors disabled:opacity-50',
        enabled ? 'bg-(--chrome-action-hover) text-foreground' : 'text-muted-foreground hover:bg-(--chrome-action-hover)/60'
      )}
      disabled={Boolean(busy)}
      onClick={toggle}
      // Don't steal focus from the search input on click.
      onMouseDown={event => event.preventDefault()}
      title={enabled ? copy.turnOff : copy.turnOn}
      type="button"
    >
      {busy ? <Loader2 className="size-4 animate-spin" /> : <PawPrint className="size-4" />}
    </button>
  )
}

function Status({ icon, text, tone }: { icon?: React.ReactNode; text: string; tone?: 'error' }) {
  return (
    <div
      className={cn(
        'flex items-center justify-center gap-2 px-2 py-6 text-xs',
        tone === 'error' ? 'text-(--ui-red)' : 'text-muted-foreground'
      )}
    >
      {icon}
      {text}
    </div>
  )
}
