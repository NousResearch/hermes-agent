import './agent-config-fields.css'

import type { ReactNode } from 'react'

import { useI18n } from '@/i18n'
import type { ConfigFieldSchema } from '@/types/hermes'

import { useAgentSettingsEditorCopy } from '../agent-settings/settings-copy'
import { type AgentSettingsLayoutCopy, useAgentSettingsLayoutCopy } from '../agent-settings/settings-layout-copy'

import { fieldCopyForSchemaKey } from './field-copy'
import { ListRow } from './primitives'
import { SettingsDisclosure } from './settings-card'

type GroupLabel = keyof AgentSettingsLayoutCopy['groups'] | 'conversation' | 'listening' | 'speaking'
type SummaryKey = keyof AgentSettingsLayoutCopy['summaries']

// The primary cards each agent page draws (Figma 895:*); every other field of the section goes
// under the page's Advanced disclosure.
const PRIMARY_GROUPS: Record<string, readonly { label: GroupLabel; keys: readonly string[]; voice?: true }[]> = {
  chat: [{ label: 'responses', keys: ['display.personality', 'display.show_reasoning'] }],
  memory: [
    { label: 'memory', keys: ['memory.memory_enabled', 'memory.user_profile_enabled', 'memory.provider'] },
    { label: 'longChats', keys: ['compression.enabled'] }
  ],
  voice: [
    { label: 'conversation', keys: ['voice.voice_chat_mode'] },
    { label: 'listening', keys: ['stt.enabled', 'stt.provider'] },
    { label: 'speaking', keys: ['voice.auto_tts', 'tts.provider'], voice: true }
  ],
  workspace: [{ label: 'project', keys: ['terminal.cwd', 'code_execution.mode'] }],
  advanced: [{ label: 'execution', keys: ['terminal.backend', 'agent.max_turns'] }]
}

const SUMMARIES: Record<string, SummaryKey> = {
  chat: 'chat',
  memory: 'memory',
  workspace: 'workspace',
  advanced: 'limits'
}

type FieldEntry = [string, ConfigFieldSchema]

interface AgentConfigFieldsProps {
  sectionId: string
  fields: FieldEntry[]
  renderField: (field: FieldEntry) => ReactNode
  /** Section reference content filed at the top of Advanced (memory's provider status). */
  advancedExtra?: ReactNode
}

export function AgentConfigFields({ advancedExtra, sectionId, fields, renderField }: AgentConfigFieldsProps) {
  const { t } = useI18n()
  const c = useAgentSettingsEditorCopy()
  const layout = useAgentSettingsLayoutCopy()
  const definitions = PRIMARY_GROUPS[sectionId]

  if (!definitions) {
    return <div className="grid gap-1">{fields.map(renderField)}</div>
  }

  const byKey = new Map(fields.map(field => [field[0], field]))

  const groups = definitions.map(group => {
    const selected = group.keys.flatMap(key => {
      const field = byKey.get(key)

      return field ? [field] : []
    })

    if (group.voice) {
      selected.push(...fields.filter(([key]) => key.startsWith('tts.') && /\.voice(?:_id)?$/.test(key)))
    }

    const label =
      group.label === 'conversation' || group.label === 'listening' || group.label === 'speaking'
        ? c.voiceGroups[group.label]
        : layout.groups[group.label]

    return { key: group.label, label, fields: selected }
  })

  const primaryKeys = new Set(groups.flatMap(group => group.fields.map(([key]) => key)))
  const advanced = fields.filter(([key]) => !primaryKeys.has(key))
  const voiceMode = sectionId === 'voice' ? byKey.get('voice.voice_chat_mode') : undefined

  const voiceHelp =
    voiceMode && (fieldCopyForSchemaKey(t.settings.fieldDescriptions, voiceMode[0]) ?? voiceMode[1].description)

  const summaryKey = SUMMARIES[sectionId]
  const summary = sectionId === 'voice' ? c.voiceAdvancedDescription : summaryKey && layout.summaries[summaryKey]

  return (
    <div className="agent-config-fields">
      {groups
        .filter(group => group.fields.length > 0)
        .map(group => (
          <section key={group.key}>
            <h3>{group.label}</h3>
            <div className="agent-config-card agent-settings-card">{group.fields.map(renderField)}</div>
          </section>
        ))}
      {(advanced.length > 0 || sectionId === 'voice' || advancedExtra) && (
        <SettingsDisclosure
          count={advanced.length || undefined}
          description={summary}
          title={t.settings.sections.advanced}
        >
          {advancedExtra}
          {voiceHelp && <ListRow description={voiceHelp} title={t.settings.subpages.voiceConversation} />}
          {sectionId === 'voice' && (
            <ListRow
              description={t.settings.config.voiceShortcutHintDesc}
              title={t.settings.config.voiceShortcutHintTitle}
            />
          )}
          {advanced.map(renderField)}
        </SettingsDisclosure>
      )}
    </div>
  )
}
