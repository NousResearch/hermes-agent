import './portal-safety-fields.css'

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
export function PortalSafetyFields({
  config,
  fields,
  onChange,
  onManagePasswords,
  advancedOpen
}: PortalSafetyFieldsProps) {
  const { t } = useI18n()
  const copy = t.shell.approvalMode
  const groupName = useId()
  const advancedRef = useRef<HTMLDetailsElement>(null)
  useEffect(() => {
    if (advancedOpen && advancedRef.current) {
      advancedRef.current.open = true
    }
  }, [advancedOpen])
  const approval = fields.find(([key]) => key === 'approvals.mode')
  const privacy = fields.filter(([key]) => key === 'security.redact_secrets')
  const advanced = fields.filter(([key]) => key !== 'approvals.mode' && key !== 'security.redact_secrets')
  const modes = ['smart', 'manual', 'off'] as const
  const descriptions = { smart: copy.smartDescription, manual: copy.manualDescription, off: copy.offDescription }

  const renderField = ([key, schema]: [string, ConfigFieldSchema]) => (
    <div className="scroll-mt-6 rounded-lg" id={`setting-field-${key}`} key={key}>
      <ConfigField
        enumOptions={enumOptionsFor(key, configFieldValue(config, key), config)}
        onChange={value => onChange(setConfigField(config, key, value))}
        schema={schema}
        schemaKey={key}
        value={configFieldValue(config, key)}
      />
    </div>
  )

  return (
    <div className="@container/safety grid gap-9" data-slot="portal-safety-settings">
      {approval && (
        <fieldset id="setting-field-approvals.mode">
          <legend className="mb-3 text-[15px] font-medium leading-[1.1]">{copy.title}</legend>
          <div className="portal-safety-options grid grid-cols-1 gap-3 @min-[32rem]/safety:grid-cols-3">
            {modes.map(mode => (
              <label
                className="portal-safety-mode relative flex cursor-pointer flex-col gap-2 p-3 has-focus-visible:outline-2 has-focus-visible:outline-ring"
                key={mode}
              >
                <input
                  checked={configFieldValue(config, 'approvals.mode') === mode}
                  className="sr-only"
                  name={groupName}
                  onChange={() => onChange(setConfigField(config, 'approvals.mode', mode))}
                  type="radio"
                  value={mode}
                />
                <Codicon
                  className="portal-safety-icon text-(--ui-text-tertiary)"
                  name={mode === 'smart' ? 'shield' : mode === 'manual' ? 'checklist' : 'circle-slash'}
                />
                <span className="text-sm font-medium">{copy[mode]}</span>
                <span className="text-[13px] leading-[1.4] text-(--ui-text-tertiary)">{descriptions[mode]}</span>
              </label>
            ))}
          </div>
        </fieldset>
      )}
      <section aria-label={t.settings.subpages.safetyPrivacy}>
        <h3 className="mb-3 text-[15px] font-medium leading-[1.1]">{t.settings.subpages.safetyPrivacy}</h3>
        <div className="portal-safety-card px-4">
          {privacy.map(renderField)}
          <div className="flex min-h-16 flex-wrap items-center justify-between gap-6 py-3 text-sm">
            <span>{t.settings.nav.vault}</span>
            <Button onClick={onManagePasswords} size="sm" variant="outline">
              {t.settings.nav.vault}
              <Codicon name="arrow-right" />
            </Button>
          </div>
        </div>
      </section>
      {advanced.length > 0 && (
        <details className="portal-safety-advanced group" ref={advancedRef}>
          <summary className="flex min-h-[68px] cursor-pointer list-none items-center justify-between gap-3 px-4 py-3.5 text-sm font-semibold focus-visible:outline-2 focus-visible:outline-ring [&::-webkit-details-marker]:hidden">
            {t.settings.sections.advanced}
            <Codicon className="group-open:rotate-180" name="chevron-down" />
          </summary>
          <div className="px-4">{advanced.map(renderField)}</div>
        </details>
      )}
    </div>
  )
}

export function PortalSafetySettings({
  config,
  fields = [],
  onChange,
  subpage
}: {
  config: HermesConfigRecord
  fields?: [string, ConfigFieldSchema][]
  onChange: (config: HermesConfigRecord) => void
  subpage?: string
}) {
  const [searchParams, setSearchParams] = useSearchParams()
  const targetField = searchParams.get('field')

  return (
    <>
      <PortalSafetyFields
        advancedOpen={
          Boolean(targetField && targetField !== 'approvals.mode' && targetField !== 'security.redact_secrets') ||
          subpage === 'checkpoints'
        }
        config={config}
        fields={fields}
        onChange={onChange}
        onManagePasswords={() =>
          setSearchParams(previous => {
            const next = new URLSearchParams(previous)
            next.set('tab', 'vault')
            next.delete('page')
            next.delete('field')

            return next
          })
        }
      />
      {subpage === 'privacy' && <SharedMetricsSettings />}
    </>
  )
}
