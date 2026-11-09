import { useEffect, useId, useRef } from 'react'
import { useSearchParams } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import type { ConfigFieldSchema, HermesConfigRecord } from '@/types/hermes'

import { ConfigField } from './config-field'
import { configFieldValue, enumOptionsFor, setConfigField } from './helpers'
import { SharedMetricsSettings } from './shared-metrics-settings'

interface PortalSafetyFieldsProps {
  config: HermesConfigRecord
  fields: [string, ConfigFieldSchema][]
  onChange: (config: HermesConfigRecord) => void
  onManagePasswords: () => void
  advancedOpen: boolean
}

/** Presentation only; the parent retains scoped autosave, validation and error handling. */
export function PortalSafetyFields({ config, fields, onChange, onManagePasswords, advancedOpen }: PortalSafetyFieldsProps) {
  const { t } = useI18n()
  const copy = t.shell.approvalMode
  const groupName = useId()
  const advancedRef = useRef<HTMLDetailsElement>(null)
  useEffect(() => {
    if (advancedOpen && advancedRef.current) {advancedRef.current.open = true}
  }, [advancedOpen])
  const approval = fields.find(([key]) => key === 'approvals.mode')
  const privacy = fields.filter(([key]) => key === 'security.redact_secrets')
  const advanced = fields.filter(([key]) => key !== 'approvals.mode' && key !== 'security.redact_secrets')
  const modes = ['smart', 'manual', 'off'] as const
  const descriptions = { smart: copy.smartDescription, manual: copy.manualDescription, off: copy.offDescription }

  const renderField = ([key, schema]: [string, ConfigFieldSchema]) => (
    <div className="scroll-mt-6 rounded-lg" id={`setting-field-${key}`} key={key}>
      <ConfigField enumOptions={enumOptionsFor(key, configFieldValue(config, key), config)}
        onChange={value => onChange(setConfigField(config, key, value))}
        schema={schema} schemaKey={key} value={configFieldValue(config, key)} />
    </div>
  )

  return (
    <div className="@container/safety grid gap-7" data-slot="portal-safety-settings">
      {approval && (
        <fieldset id="setting-field-approvals.mode">
          <legend className="mb-3 text-sm font-medium">{copy.title}</legend>
          <div className="grid grid-cols-1 gap-3 @min-[32rem]/safety:grid-cols-3">
            {modes.map(mode => (
              <label className="relative flex cursor-pointer flex-col gap-3 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary) p-4 has-checked:border-primary has-checked:bg-primary/10 has-focus-visible:outline-2 has-focus-visible:outline-ring" key={mode}>
                <input checked={configFieldValue(config, 'approvals.mode') === mode} className="sr-only"
                  name={groupName} onChange={() => onChange(setConfigField(config, 'approvals.mode', mode))}
                  type="radio" value={mode} />
                <Codicon className="text-(--ui-text-tertiary)" name={mode === 'smart' ? 'shield' : mode === 'manual' ? 'checklist' : 'circle-slash'} />
                <span className="text-sm font-medium">{copy[mode]}</span>
                <span className="text-xs leading-relaxed text-(--ui-text-tertiary)">{descriptions[mode]}</span>
              </label>
            ))}
          </div>
        </fieldset>
      )}
      <section aria-label={t.settings.subpages.safetyPrivacy}>
        <h3 className="mb-3 text-sm font-medium">{t.settings.subpages.safetyPrivacy}</h3>
        <div className="rounded-md border border-(--ui-stroke-secondary) px-4">
          {privacy.map(renderField)}
          <div className="flex items-center justify-between gap-3 py-3 text-sm">
            <span>{t.settings.nav.vault}</span>
            <Button onClick={onManagePasswords} size="sm" variant="outline">{t.settings.nav.vault}<Codicon name="arrow-right" /></Button>
          </div>
        </div>
      </section>
      {advanced.length > 0 && (
        <details className="group rounded-md border border-(--ui-stroke-secondary)" ref={advancedRef}>
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 text-sm focus-visible:outline-2 focus-visible:outline-ring [&::-webkit-details-marker]:hidden">
            {t.settings.sections.advanced}<Codicon className="group-open:rotate-180" name="chevron-down" />
          </summary>
          <div className="border-t border-(--ui-stroke-secondary) px-4">{advanced.map(renderField)}</div>
        </details>
      )}
    </div>
  )
}

export function PortalSafetySettings({ config, fields = [], onChange, subpage }: {
  config: HermesConfigRecord
  fields?: [string, ConfigFieldSchema][]
  onChange: (config: HermesConfigRecord) => void
  subpage?: string
}) {
  const [searchParams, setSearchParams] = useSearchParams()
  const targetField = searchParams.get('field')

  return <>
    <PortalSafetyFields
      advancedOpen={Boolean(targetField && targetField !== 'approvals.mode' && targetField !== 'security.redact_secrets') || subpage === 'checkpoints'}
      config={config}
      fields={fields}
      onChange={onChange}
      onManagePasswords={() => setSearchParams(previous => {
        const next = new URLSearchParams(previous)
        next.set('tab', 'vault')
        next.delete('page')
        next.delete('field')

        return next
      })}
    />
    {subpage === 'privacy' && <SharedMetricsSettings />}
  </>
}
