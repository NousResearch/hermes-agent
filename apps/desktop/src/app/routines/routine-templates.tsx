import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { SearchField } from '@/components/ui/search-field'
import { Skeleton } from '@/components/ui/skeleton'
import type { AutomationBlueprint } from '@/hermes'
import { useI18n } from '@/i18n'

import { capitalized } from './routine-schedule'
import { type RoutinesCopy, useRoutinesCopy } from './routines-copy'

/** The catalog's own categories first, in this order; a plugin's own ones follow by name. */
const CATEGORY_ORDER = ['daily', 'weekly', 'email', 'general']

type KnownCategory = keyof RoutinesCopy['templateCategories']

const isKnown = (category: string): category is KnownCategory => CATEGORY_ORDER.includes(category)

/** Templates by category, each matching every word of the search in its title, description or
 *  category. Pure, so the order and the search are tested as data. */
export function templateGroups(templates: readonly AutomationBlueprint[], query: string): [string, AutomationBlueprint[]][] {
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean)
  const groups = new Map<string, AutomationBlueprint[]>()

  for (const template of templates) {
    const text = `${template.title} ${template.description} ${template.category}`.toLowerCase()

    if (words.every(word => text.includes(word))) {
      groups.set(template.category, [...(groups.get(template.category) ?? []), template])
    }
  }

  const rank = (category: string) => (isKnown(category) ? CATEGORY_ORDER.indexOf(category) : CATEGORY_ORDER.length)

  return [...groups.entries()].sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b))
}

interface RoutineTemplatesProps {
  templates: readonly AutomationBlueprint[]
  loading: boolean
  failed: boolean
  onRetry: () => void
  onPick: (template: AutomationBlueprint) => void
}

/** Template: a searchable list in the catalog's categories (rev 1399:23104). No "Suggested for…"
 *  group: the data for it is not there yet. */
export function RoutineTemplates({ failed, loading, onPick, onRetry, templates }: RoutineTemplatesProps) {
  const { locale, t } = useI18n()
  const copy = useRoutinesCopy()
  const [query, setQuery] = useState('')
  const groups = templateGroups(templates, query)
  const categoryLabel = (category: string) => (isKnown(category) ? copy.templateCategories[category] : capitalized(category, locale))

  return (
    <div className="routines-templates">
      <SearchField
        aria-label={copy.searchTemplates}
        onChange={setQuery}
        placeholder={copy.searchTemplates}
        value={query}
        variant="box"
      />
      {loading ? (
        <div aria-hidden className="routines-templates-list">
          {Array.from({ length: 6 }, (_, index) => (
            <div className="routines-template" key={index}>
              <Skeleton className="h-3 w-40 max-w-full" />
            </div>
          ))}
        </div>
      ) : failed ? (
        <p className="routines-templates-note" role="alert">
          {copy.templatesFailed}{' '}
          <Button onClick={onRetry} size="inline" variant="textStrong">
            {t.common.retry}
          </Button>
        </p>
      ) : groups.length ? (
        <div className="routines-templates-list">
          {groups.map(([category, items]) => (
            <section aria-label={categoryLabel(category)} className="routines-templates-group" key={category}>
              <h3>{categoryLabel(category)}</h3>
              {items.map(template => (
                <button className="routines-template" key={template.key} onClick={() => onPick(template)} type="button">
                  <span className="routines-template-title">{template.title}</span>
                  <span className="routines-template-description">{template.description}</span>
                </button>
              ))}
            </section>
          ))}
        </div>
      ) : (
        <p className="routines-templates-note">{copy.noTemplates}</p>
      )}
    </div>
  )
}
