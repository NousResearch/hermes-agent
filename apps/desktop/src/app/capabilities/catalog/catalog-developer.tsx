import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { openExternalLink } from '@/lib/external-link'

import type { CatalogKind } from './catalog-data'

/** One banner for every catalog. The wing, copy and docs links are shared; the page supplies its own management actions. */
export function CatalogDeveloper({ kind, actions }: { kind: CatalogKind; actions?: ReactNode }) {
  return (
    <section data-catalog-section="developer">
      <header className="catalog-section-heading"><h2>Developer Mode</h2></header>
      <div className="catalog-developer">
        <div aria-hidden className="catalog-developer-art" />
        <div className="catalog-developer-content">
          <span aria-hidden className="catalog-developer-mark" />
          <h3>Build for Hermes</h3>
          <p className="text-sm text-(--ui-text-secondary)">Built a tool, connector, plugin, or Mod?<br />Want to but don’t know how?</p>
          <div className="catalog-developer-ctas">
            <Button onClick={() => void openExternalLink('https://github.com/NousResearch/hermes-agent/compare')} size="sm" variant="default">
              Submit Mod<Codicon name="arrow-up-right" />
            </Button>
            <Button onClick={() => void openExternalLink(kind === 'plugins' ? 'https://hermes-agent.nousresearch.com/docs/user-guide/features/plugin-catalog' : 'https://hermes-agent.nousresearch.com/docs/skills/')} size="sm" variant="secondary">
              Developer Docs<Codicon name="arrow-up-right" />
            </Button>
          </div>
        </div>
      </div>
      {actions && <div className="catalog-developer-management">{actions}</div>}
    </section>
  )
}
