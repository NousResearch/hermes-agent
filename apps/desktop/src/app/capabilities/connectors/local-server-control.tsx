import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { McpCatalogEntry } from '@/hermes'
import { useI18n } from '@/i18n'

export type InstallField = McpCatalogEntry['required_env'][number]

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
