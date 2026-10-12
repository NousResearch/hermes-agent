import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { selectableCardClass } from '@/lib/selectable-card'
import { isLocalScope } from '@/lib/simple-agent-defaults'
import { cn } from '@/lib/utils'
import { setInterfaceMode } from '@/store/interface-mode'
import {
  $desktopOnboarding,
  dismissFirstRunOnboarding,
  type OnboardingContext,
  refreshOnboardingProviders,
  startProviderOAuth
} from '@/store/onboarding'

import { Status } from './flow'
import { NousMark, RecommendedTag, SIMPLE_SIGN_IN_TILES } from './providers'

/**
 * Simple mode's connect screen: account sign-ins a non-technical user is likely
 * to already have, one plain requirement each. Anything else (API keys, local
 * models) is Advanced, one click away.
 */
export function SimplePicker({ ctx }: { ctx: OnboardingContext }) {
  const { t } = useI18n()
  const { manual, providers } = useStore($desktopOnboarding)

  if (providers === null) {
    return <Status>{t.onboarding.lookingUpProviders}</Status>
  }

  // An `external` sign-in (Claude) runs Hermes' own CLI in a terminal on THIS
  // machine, so it only signs in a backend that lives here.
  const local = isLocalScope(ctx.scope ?? {})

  const tiles = SIMPLE_SIGN_IN_TILES.flatMap(tile => {
    const provider = providers.find(p => p.id === tile.providerId)

    return provider && (provider.flow !== 'external' || local) ? [{ ...tile, provider }] : []
  })

  return (
    <div className="grid gap-3">
      {tiles.length > 0 ? (
        <div className="grid gap-2 sm:grid-cols-2">
          {tiles.map(({ provider, tile, title }) => (
            <SimpleTile
              featured={tile === 'nous'}
              key={tile}
              mark={tile === 'nous' ? <NousMark /> : null}
              onClick={() => void startProviderOAuth(provider, ctx)}
              requirement={t.onboarding.simpleTiles[tile]}
              title={title}
            />
          ))}
        </div>
      ) : (
        // The sign-in list failed or this backend offers none of them: a way
        // forward that is neither an API key form nor a dead end.
        <div className="grid justify-items-center gap-2 text-center">
          <p className="text-sm text-muted-foreground">{t.onboarding.simpleNoSignIns}</p>
          <Button onClick={() => void refreshOnboardingProviders()} size="sm" type="button" variant="outline">
            {t.onboarding.tryAgain}
          </Button>
        </div>
      )}
      <div className="flex flex-wrap items-center justify-center gap-2">
        {/* First run only, like Advanced's picker: a manual open already has a close. */}
        {manual ? null : (
          <Button
            className="font-medium"
            onClick={() => dismissFirstRunOnboarding()}
            size="xs"
            type="button"
            variant="text"
          >
            {t.onboarding.chooseLater}
          </Button>
        )}
        <Button
          className="font-medium"
          onClick={() => setInterfaceMode('advanced')}
          size="xs"
          type="button"
          variant="text"
        >
          {t.interfaceMode.switchToAdvanced}
        </Button>
      </div>
    </div>
  )
}

// The Nous tile wears the same animated border and "Recommended" chip as the
// featured Nous row in the full picker and Settings.
function SimpleTile({
  featured = false,
  mark,
  onClick,
  requirement,
  title
}: {
  featured?: boolean
  mark?: ReactNode
  onClick: () => void
  requirement: string
  title: string
}) {
  return (
    <button
      className={cn(
        selectableCardClass({ prominent: true }),
        'relative grid min-h-24 content-start gap-1.5 p-4 text-left'
      )}
      onClick={onClick}
      type="button"
    >
      {featured ? <span aria-hidden className="arc-border arc-reverse arc-nous" /> : null}
      <span className="flex flex-wrap items-center gap-2 text-[length:var(--conversation-text-font-size)] font-semibold">
        {mark}
        {title}
        {featured ? <RecommendedTag /> : null}
      </span>
      <span className="text-xs leading-5 text-muted-foreground">{requirement}</span>
    </button>
  )
}
