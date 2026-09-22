import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import type { McpCatalogEntry } from '@/hermes'
import { useI18n } from '@/i18n'

import type { ConnectorWayLocal } from './types'

export type InstallField = McpCatalogEntry['required_env'][number]

export function LocalServerControl({
  name,
  onAuthenticate,
  onServerToggle,
  way
}: {
  name: string
  onAuthenticate?: () => void
  onServerToggle?: (next: boolean) => void
  way: ConnectorWayLocal
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage

  return (
    <div className="grid gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="min-w-0 flex-1 truncate text-[0.7rem] text-(--ui-text-tertiary)">
          {copy.card.state[way.serverEnabled === true ? 'serverOn' : 'serverOff']}
        </span>
        {onServerToggle ? (
          <Switch
            aria-label={way.serverEnabled === true ? copy.card.turnServerOff(name) : copy.card.turnServerOn(name)}
            checked={way.serverEnabled ?? false}
            onCheckedChange={onServerToggle}
            size="xs"
          />
        ) : null}
      </div>

      {way.verb === 'authenticate' && onAuthenticate ? (
        <Button className="justify-self-start" onClick={onAuthenticate} size="xs">
          {copy.card.verb.authenticate}
        </Button>
      ) : null}
    </div>
  )
}

export function LocalInstall({
  installFields = [],
  installing = false,
  onInstall
}: {
  installFields?: readonly InstallField[]
  installing?: boolean
  onInstall: (env: Record<string, string>) => void
}) {
  const { t } = useI18n()
  const [draft, setDraft] = useState<Record<string, string>>({})
  const missing = installFields.some(field => field.required === true && !(draft[field.name] ?? '').trim())

  return (
    <div className="grid justify-items-start gap-2">
      {installFields.length > 0 ? (
        <>
          <p className="text-[0.7rem] text-(--ui-text-tertiary)">{t.settings.mcp.catalogEnvRequired}</p>
          {installFields.map(field => (
            <label className="grid w-full gap-1" key={field.name}>
              <span className="text-[0.65rem] text-(--ui-text-secondary)">
                {field.prompt || field.name}
                {field.required ? ' *' : ''}
              </span>
              <Input
                className="h-7 text-xs"
                onChange={event => setDraft({ ...draft, [field.name]: event.currentTarget.value })}
                type="password"
                value={draft[field.name] ?? ''}
              />
            </label>
          ))}
        </>
      ) : null}

      <Button
        disabled={installing || missing}
        loading={installing}
        onClick={() => onInstall(draft)}
        size="xs"
        variant="outline"
      >
        {t.connectorsPage.card.verb.install}
      </Button>
    </div>
  )
}
