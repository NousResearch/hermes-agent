import './agent-config-fields.css'

import type { ReactNode } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import type { ConfigFieldSchema } from '@/types/hermes'

import { useAgentSettingsEditorCopy } from '../agent-settings/settings-copy'

import { fieldCopyForSchemaKey } from './field-copy'
import { ListRow } from './primitives'

const PRIMARY_GROUPS = {
  chat: [{ label: 'chatBehavior', keys: ['display.personality', 'display.show_reasoning'] }],
  memory: [
    { label: 'memoryPersistent', keys: ['memory.memory_enabled', 'memory.user_profile_enabled', 'memory.provider'] },
    { label: 'memoryContext', keys: ['compression.enabled'] }
  ],
  voice: [
    { label: 'voiceConversation', heading: 'conversation', keys: ['voice.voice_chat_mode'] },
    { label: 'voiceTranscription', heading: 'listening', keys: ['stt.enabled', 'stt.provider'] },
    { label: 'voiceSpeech', heading: 'speaking', keys: ['voice.auto_tts', 'tts.provider'], voice: true }
  ],
  workspace: [{ label: 'workspaceProjects', keys: ['terminal.cwd', 'code_execution.mode'] }],
  advanced: [{ label: 'advancedRuntime', keys: ['terminal.backend', 'agent.max_turns'] }]
} as const

type FieldEntry = [string, ConfigFieldSchema]

interface AgentConfigFieldsProps {
  sectionId: string
  fields: FieldEntry[]
  renderField: (field: FieldEntry) => ReactNode
}

export function AgentConfigFields({ sectionId, fields, renderField }: AgentConfigFieldsProps) {
  const { t } = useI18n()
  const c = useAgentSettingsEditorCopy()
  const definitions = PRIMARY_GROUPS[sectionId as keyof typeof PRIMARY_GROUPS]

  if (!definitions) {
    return <div className="grid gap-1">{fields.map(renderField)}</div>
  }

  const byKey = new Map<string, FieldEntry>()

  for (const field of fields) {
    byKey.set(field[0], field)
  }

  const groups = definitions.map(group => {
    const selected: FieldEntry[] = []

    for (const key of group.keys) {
      const field = byKey.get(key)

      if (field) {
        selected.push(field)
      }
    }
    if ('voice' in group) {
      for (const field of fields) {
        if (field[0].startsWith('tts.') && /\.voice(?:_id)?$/.test(field[0])) {
          selected.push(field)
        }
      }
    }

    return {
      key: group.label,
      label: 'heading' in group ? c.voiceGroups[group.heading] : t.settings.subpages[group.label],
      fields: selected
    }
  })
  const primaryKeys = new Set(groups.flatMap(group => group.fields.map(([key]) => key)))
  const advanced = fields.filter(([key]) => !primaryKeys.has(key))
  const voiceMode = sectionId === 'voice' ? byKey.get('voice.voice_chat_mode') : undefined
  const voiceHelp =
    voiceMode && (fieldCopyForSchemaKey(t.settings.fieldDescriptions, voiceMode[0]) ?? voiceMode[1].description)

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
      {(advanced.length > 0 || sectionId === 'voice') && (
        <details className="agent-config-advanced agent-settings-advanced group">
          <summary>
            <span>
              <span className="block">{t.settings.sections.advanced}</span>
              {sectionId === 'voice' && (
                <span className="block text-[13px] leading-[18px] text-(--ui-text-tertiary)">
                  {c.voiceAdvancedDescription}
                </span>
              )}
            </span>
            <span className="flex items-center gap-2 text-(--ui-text-tertiary)">
              {advanced.length > 0 && <span className="text-xs tabular-nums">{advanced.length}</span>}
              <Codicon className="group-open:rotate-180" name="chevron-down" />
            </span>
          </summary>
          <div>
            {voiceHelp && <ListRow description={voiceHelp} title={t.settings.subpages.voiceConversation} />}
            {sectionId === 'voice' && (
              <ListRow
                description={t.settings.config.voiceShortcutHintDesc}
                title={t.settings.config.voiceShortcutHintTitle}
              />
            )}
            {advanced.map(renderField)}
          </div>
        </details>
      )}
    </div>
  )
}
