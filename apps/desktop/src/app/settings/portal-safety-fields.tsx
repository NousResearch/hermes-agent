import './portal-safety-fields.css'

import { useEffect, useId, useRef } from 'react'
import { useSearchParams } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import type { ConfigFieldSchema, HermesConfigRecord } from '@/types/hermes'

import { useAgentSettingsLayoutCopy } from '../agent-settings/settings-layout-copy'

import { ConfigField } from './config-field'
import { configFieldValue, enumOptionsFor, setConfigField } from './helpers'
import { ListRow } from './primitives'
import { SettingsCard, SettingsDisclosure, SettingsGroup, SettingsOverview } from './settings-card'
import { useSettingsPagesCopy } from './settings-pages-copy'
import { SharedMetricsSettings } from './shared-metrics-settings'

const APPROVAL_MODES = ['smart', 'manual', 'off'] as const

const MODE_ICONS: Record<(typeof APPROVAL_MODES)[number], string> = {
  smart: 'shield',
  manual: 'checklist',
  off: 'circle-slash'
}

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
  const pages = useSettingsPagesCopy()
  const layout = useAgentSettingsLayoutCopy()
  const [vaultTitle, vaultDescription] = layout.rows.vault
  const descriptions = { smart: copy.smartDescription, manual: copy.manualDescription, off: copy.offDescription }

  const renderField = ([key, schema]: [string, ConfigFieldSchema]) => (
    <div className="scroll-mt-6" id={`setting-field-${key}`} key={key}>
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
    <SettingsOverview>
      {approval && (
        <SettingsGroup title={copy.title}>
          <fieldset className="portal-safety-options global-settings-card" id="setting-field-approvals.mode">
            <legend className="sr-only">{copy.title}</legend>
            {APPROVAL_MODES.map(mode => (
              <label className="portal-safety-mode" key={mode}>
                <input
                  checked={configFieldValue(config, 'approvals.mode') === mode}
                  className="portal-safety-radio"
                  name={groupName}
                  onChange={() => onChange(setConfigField(config, 'approvals.mode', mode))}
                  type="radio"
                  value={mode}
                />
                <Codicon className="portal-safety-icon" name={MODE_ICONS[mode]} />
                <span className="portal-safety-mode-title">{copy[mode]}</span>
                <span className="portal-safety-mode-description">{descriptions[mode]}</span>
              </label>
            ))}
          </fieldset>
        </SettingsGroup>
      )}
      <SettingsGroup title={t.settings.subpages.safetyPrivacy}>
        <SettingsCard>
          {privacy.map(renderField)}
          <ListRow
            action={
              <Button
                aria-label={`${pages.gateway.manage}: ${vaultTitle}`}
                onClick={onManagePasswords}
                variant="secondary"
              >
                {pages.gateway.manage}
              </Button>
            }
            description={vaultDescription}
            title={vaultTitle}
          />
        </SettingsCard>
      </SettingsGroup>
      {advanced.length > 0 && (
        <SettingsDisclosure
          description={layout.summaries.safety}
          ref={advancedRef}
          title={t.settings.sections.advanced}
        >
          <SettingsCard>{advanced.map(renderField)}</SettingsCard>
        </SettingsDisclosure>
      )}
    </SettingsOverview>
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
