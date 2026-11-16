import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import type { DelegationProfileRoutes, ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { Users } from '@/lib/icons'
import { isSubmitEnter } from '@/lib/ime'

import { Pill, SectionHeading } from '../primitives'

import { describeProfileRoute, setDefaultProfileRoute, toggleProfileRoute } from './routes'
import { saveProfileRoutes } from './routing-data'

interface ProfileRoutesEditorProps {
  scope: ProfileScope
  routes: DelegationProfileRoutes
  /** Profiles that exist on this backend host. */
  profiles: readonly string[]
}

/** `delegation.backends.profile_routes`: check the profiles subagents may run as, describe them, pick a default. */
export function ProfileRoutesEditor({ scope, routes, profiles }: ProfileRoutesEditorProps) {
  const { t } = useI18n()
  const r = t.settings.model.subagents.routing
  const save = (next: DelegationProfileRoutes) => void saveProfileRoutes(scope, next, r.saveFailed)
  // A listed profile that was deleted since stays visible (and uncheckable) instead of vanishing.
  const missing = routes.profiles.map(p => p.name).filter(name => !profiles.includes(name))

  return (
    <section data-slot="delegation-profile-routes">
      <SectionHeading icon={Users} title={r.profileRoutesTitle} />
      <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {r.profileRoutesHint}
      </p>
      {profiles.length === 0 ? (
        <p className="py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {r.profileEmpty}
        </p>
      ) : (
        <ul className="mt-2 grid">
          {[...profiles, ...missing].map(name => {
            const entry = routes.profiles.find(p => p.name === name)

            return (
              <ProfileRow
                checked={Boolean(entry)}
                description={entry?.description ?? ''}
                isDefault={routes.default === name}
                key={name}
                label={missing.includes(name) ? r.profileMissing(name) : name}
                name={name}
                onDescribe={text => save(describeProfileRoute(routes, name, text))}
                onMakeDefault={() => save(setDefaultProfileRoute(routes, name))}
                onToggle={on => save(toggleProfileRoute(routes, name, on))}
              />
            )
          })}
        </ul>
      )}
    </section>
  )
}

interface ProfileRowProps {
  name: string
  label: string
  checked: boolean
  description: string
  isDefault: boolean
  onToggle: (on: boolean) => void
  onDescribe: (text: string) => void
  onMakeDefault: () => void
}

function ProfileRow({
  name,
  label,
  checked,
  description,
  isDefault,
  onToggle,
  onDescribe,
  onMakeDefault
}: ProfileRowProps) {
  const { t } = useI18n()
  const r = t.settings.model.subagents.routing
  const [draft, setDraft] = useState(description)
  const id = `delegation-profile-${name}`

  useEffect(() => setDraft(description), [description])

  const commit = () => {
    if (draft.trim() !== description.trim()) {
      onDescribe(draft)
    }
  }

  return (
    <li className="grid min-h-12 grid-cols-[minmax(8rem,12rem)_minmax(0,1fr)_auto] items-center gap-3 border-t border-(--ui-stroke-tertiary) py-2 first:border-t-0">
      <label className="flex min-w-0 items-center gap-2.5" htmlFor={id}>
        <Checkbox checked={checked} id={id} onCheckedChange={value => onToggle(value === true)} />
        <span className="truncate text-[length:var(--conversation-text-font-size)] text-foreground">{label}</span>
      </label>
      {checked ? (
        <Input
          aria-label={`${name}: ${r.descriptionLabel}`}
          onBlur={commit}
          onChange={event => setDraft(event.target.value)}
          onKeyDown={event => {
            if (isSubmitEnter(event)) {
              event.currentTarget.blur()
            }
          }}
          placeholder={r.profileDescriptionPlaceholder}
          size="sm"
          value={draft}
        />
      ) : (
        <span />
      )}
      <span className="flex w-24 justify-end">
        {checked &&
          (isDefault ? (
            <Pill tone="primary">{r.profileDefault}</Pill>
          ) : (
            <Button onClick={onMakeDefault} size="xs" variant="outline">
              {r.makeDefault}
            </Button>
          ))}
      </span>
    </li>
  )
}
