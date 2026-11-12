import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { EmptyState } from '@/components/ui/empty-state'
import { Field } from '@/components/ui/field'
import { SearchField } from '@/components/ui/search-field'
import { Skeleton } from '@/components/ui/skeleton'
import type { AutomationBlueprint } from '@/hermes'

import { AgentSelect } from './routine-fields'
import { useRoutinesCopy } from './routines-copy'

/** The ready-made table shows this many before "Show all N" (Figma 1099:9000). */
const FIRST_ROWS = 6

interface RoutinePickerProps {
  blueprints: readonly AutomationBlueprint[]
  loading: boolean
  profile: string
  lockProfile: boolean
  onProfileChange: (profile: string) => void
  onUse: (blueprint: AutomationBlueprint) => void
  onBlank: () => void
  onAskInChat: () => void
}

/** R3, "What kind of routine?" (Figma 1099:8945): the agent, the ready-made routines the backend
 *  serves for it (its blueprint catalog) with search, and the two other ways in: describe it in a chat
 *  with the agent, or start blank. Nothing is created until the next step's Create. */
export function RoutinePicker({
  blueprints,
  loading,
  lockProfile,
  onAskInChat,
  onBlank,
  onProfileChange,
  onUse,
  profile
}: RoutinePickerProps) {
  const copy = useRoutinesCopy()
  const [query, setQuery] = useState('')
  const [expanded, setExpanded] = useState(false)
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean)

  const matches = blueprints.filter(item => {
    const text = `${item.title} ${item.description} ${item.category}`.toLowerCase()

    return words.every(word => text.includes(word))
  })

  const shown = expanded || words.length ? matches : matches.slice(0, FIRST_ROWS)

  return (
    <div className="routines-picker">
      {!lockProfile && (
        <div className="routines-picker-agent">
          <Field htmlFor="routine-pick-agent" label={copy.forAgent}>
            <AgentSelect id="routine-pick-agent" onChange={onProfileChange} value={profile} />
          </Field>
        </div>
      )}
      <div className="routines-picker-head">
        <span>{copy.readyMade}</span>
        {!loading && <span>{blueprints.length}</span>}
      </div>
      <SearchField
        aria-label={copy.searchReadyMade}
        onChange={setQuery}
        placeholder={copy.searchReadyMade}
        value={query}
        variant="box"
      />
      {loading ? (
        <div aria-hidden className="routines-picker-list">
          {Array.from({ length: FIRST_ROWS }, (_, index) => (
            <div className="routines-picker-row" key={index}>
              <Skeleton className="h-3 w-40 max-w-full" />
            </div>
          ))}
        </div>
      ) : shown.length ? (
        <ul className="routines-picker-list">
          {shown.map(item => (
            <li className="routines-picker-row" key={item.key}>
              <span className="routines-picker-title">{item.title}</span>
              <span className="routines-picker-description">{item.description}</span>
              <Button aria-label={`${copy.use}: ${item.title}`} onClick={() => onUse(item)} size="sm" variant="link">
                {copy.use}
              </Button>
            </li>
          ))}
        </ul>
      ) : (
        <EmptyState description={copy.noReadyMade} />
      )}
      {!words.length && matches.length > FIRST_ROWS && (
        <Button className="routines-picker-more" onClick={() => setExpanded(!expanded)} size="sm" variant="text">
          {expanded ? copy.showFewer : copy.showAll(matches.length)}
        </Button>
      )}
      <div className="routines-picker-foot">
        <Codicon aria-hidden name="comment-discussion" />
        <span>{copy.describeInChat}</span>
        <Button onClick={onAskInChat} size="sm" variant="link">
          {copy.askInChat}
        </Button>
        <Button className="routines-picker-blank" onClick={onBlank} variant="secondary">
          {copy.startBlank}
        </Button>
      </div>
    </div>
  )
}
