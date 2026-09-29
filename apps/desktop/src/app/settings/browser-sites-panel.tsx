import { useCallback, useState } from 'react'

import { Button } from '@/components/ui/button'
import { type ProfileScope, saveHermesConfigRecord } from '@/hermes'
import { useI18n } from '@/i18n'
import { Plus, X } from '@/lib/icons'
import { notifyError } from '@/store/notifications'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '../hooks/use-config-record'

import { normalizeSiteRule, readBlocklist, type WebsiteBlocklist, withSiteAdded, withSiteRemoved } from './browser-sites'
import { setNested } from './helpers'
import { ToggleRow } from './primitives'

const COPY = {
  en: {
    add: 'Add site',
    description: 'Czesiek will not open these sites in the browser or read them with web tools.',
    empty: 'No blocked sites yet. Banks and admin panels are good first entries.',
    invalid: 'That does not look like a website address.',
    label: 'Blocked sites',
    placeholder: 'e.g. mybank.com',
    remove: 'Remove'
  },
  pl: {
    add: 'Dodaj stronę',
    description: 'Czesiek nie otworzy tych stron w przeglądarce ani nie przeczyta ich narzędziami sieciowymi.',
    empty: 'Nie ma jeszcze zablokowanych stron. Dobre na początek: bank i panele administracyjne.',
    invalid: 'To nie wygląda na adres strony.',
    label: 'Zablokowane strony',
    placeholder: 'np. mojbank.pl',
    remove: 'Usuń'
  }
} as const

/**
 * Sites the agent may not touch. Enforced by the backend's website policy for
 * the browser and the web tools alike; this is its editor. It is a blocklist —
 * "ask before visiting" is a separate, not yet built, kind of control.
 */
export function BrowserSitesPanel({ profile }: { profile?: ProfileScope }) {
  const { locale } = useI18n()
  const copy = locale === 'pl' ? COPY.pl : COPY.en
  const { data: config } = useHermesConfigRecord(profile)
  const setConfig = hermesConfigCacheWriter(profile)
  const [busy, setBusy] = useState(false)
  const [draft, setDraft] = useState('')
  const [invalid, setInvalid] = useState(false)

  const list = readBlocklist(config)

  const save = useCallback(
    async (next: WebsiteBlocklist) => {
      if (!config) {
        return
      }

      const record = setNested(setNested(config, 'security.website_blocklist.enabled', next.enabled), 'security.website_blocklist.domains', next.domains)

      setBusy(true)
      setConfig(record)

      try {
        await saveHermesConfigRecord(record, profile)
      } catch (error) {
        setConfig(config)
        notifyError(error, copy.label)
      } finally {
        setBusy(false)
      }
    },
    [config, copy.label, profile, setConfig]
  )

  const add = () => {
    if (!normalizeSiteRule(draft)) {
      setInvalid(true)

      return
    }

    setInvalid(false)
    setDraft('')
    void save(withSiteAdded(list, draft))
  }

  return (
    <div className="grid gap-2">
      <ToggleRow
        checked={list.enabled}
        description={copy.description}
        disabled={busy || !config}
        label={copy.label}
        onChange={on => void save({ ...list, enabled: on })}
      />
      <form
        className="flex items-center gap-2"
        onSubmit={event => {
          event.preventDefault()
          add()
        }}
      >
        <input
          aria-invalid={invalid}
          aria-label={copy.placeholder}
          className="jarvis-well min-w-0 flex-1 px-3 py-2 text-sm text-(--ui-text-primary) outline-none placeholder:text-(--ui-text-tertiary)"
          disabled={busy || !config}
          onChange={event => {
            setDraft(event.target.value)
            setInvalid(false)
          }}
          placeholder={copy.placeholder}
          value={draft}
        />
        <Button disabled={busy || !config || !draft.trim()} size="sm" type="submit" variant="secondary">
          <Plus />
          {copy.add}
        </Button>
      </form>
      {invalid ? (
        <p className="text-xs text-destructive" role="alert">
          {copy.invalid}
        </p>
      ) : null}
      {list.domains.length === 0 ? (
        <p className="text-xs text-(--ui-text-tertiary)">{copy.empty}</p>
      ) : (
        <ul className="flex flex-wrap gap-1.5">
          {list.domains.map(domain => (
            <li className="jarvis-well flex items-center gap-1 py-1 pl-3 pr-1 text-sm" key={domain}>
              {domain}
              <Button
                aria-label={`${copy.remove} ${domain}`}
                disabled={busy}
                onClick={() => void save(withSiteRemoved(list, domain))}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <X />
              </Button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
