import { useContributions } from '@/contrib'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'

/**
 * Appearance settings plugin seams. The extra area adds controls to the
 * root page; chatDisplay adds controls alongside conversation display
 * preferences. Plugins use the app's own primitives instead of injecting
 * nodes into the page and driving its widgets through React internals.
 *
 * The app's swatch grid (`ColorSwatches`, exported from the SDK) is the
 * sanctioned control for colour picking: it renders the same grid the profile
 * rail and project dialog use, with the plugin's own `onChange`.
 */
export const APPEARANCE_AREAS = {
  /** Appended to the root Appearance settings page. */
  extra: 'appearance.extra',
  /** Appended to Appearance → Conversation display settings. */
  chatDisplay: 'appearance.chatDisplay'
} as const

/** Mounts every `appearance.extra` registration (own error boundary each, so a
 *  broken contribution degrades to an inline error instead of a dead page). */
export function AppearanceExtraSlot() {
  return <AppearanceContributionSlot area={APPEARANCE_AREAS.extra} />
}

export function AppearanceChatDisplaySlot() {
  return <AppearanceContributionSlot area={APPEARANCE_AREAS.chatDisplay} />
}

function AppearanceContributionSlot({ area }: { area: string }) {
  const contributions = useContributions(area)

  if (contributions.length === 0) {
    return null
  }

  return (
    <>
      {contributions.map(contribution => (
        <ContribBoundary id={contribution.id} key={`${contribution.source ?? 'core'}:${contribution.id}`}>
          {contribution.render && <ContribRender render={contribution.render} />}
        </ContribBoundary>
      ))}
    </>
  )
}
