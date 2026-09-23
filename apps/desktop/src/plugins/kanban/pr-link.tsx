/**
 * A card's PR link. `pr_url` comes from the backend (the accepted completion
 * contract, else the newest run's `metadata.published_pr`), and the number is
 * what an operator says out loud — the repo is on the other end of the link.
 * `stopPropagation` keeps a click on a card's chip from opening the drawer
 * behind it.
 */

import { ExternalLink } from '@hermes/plugin-sdk'

import { useKanban } from './ui'

export const prNumber = (url: string): string => /\/pull\/(\d+)/.exec(url)?.[1] ?? ''

export function PrLink({ className, url }: { className?: string; url: string }) {
  const k = useKanban()
  const number = prNumber(url)

  if (!number) {
    return null
  }

  return (
    <ExternalLink
      className={className ?? 'text-(--ui-text-tertiary) hover:text-foreground'}
      href={url}
      onClick={event => event.stopPropagation()}
      title={k.prTip(url)}
    >
      #{number}
    </ExternalLink>
  )
}
