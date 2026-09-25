interface ConfigSubpageDefinition {
  id: string
  labelKey: string
  fields?: string[]
  prefixes?: string[]
}

// Shared by the page filter and search routing. Match field names, not the
// backend schema: sectionFieldEntries also exposes config-present fields whose
// schema is inferred at runtime. Embedded model and device-local controls use
// these same page ids in ConfigSettings.
const CONFIG_SUBPAGE_DEFINITIONS: Record<string, ConfigSubpageDefinition[]> = {
  model: [
    {
      id: 'main',
      labelKey: 'modelMain',
      fields: ['model', 'model_context_length', 'agent.reasoning_effort', 'agent.service_tier'],
      prefixes: ['model.']
    },
    { id: 'fallbacks', labelKey: 'modelFallbacks', fields: ['fallback_providers'] },
    { id: 'auxiliary', labelKey: 'modelAuxiliary', prefixes: ['auxiliary.'] },
    { id: 'moa', labelKey: 'modelMoa', prefixes: ['moa.'] }
  ],
  chat: [
    {
      id: 'behavior',
      labelKey: 'chatBehavior',
      fields: ['display.personality', 'timezone', 'display.show_reasoning']
    },
    { id: 'attachments', labelKey: 'chatAttachments', fields: ['agent.image_input_mode'] }
  ],
  workspace: [
    {
      id: 'projects',
      labelKey: 'workspaceProjects',
      fields: ['terminal.cwd'],
      prefixes: ['desktop.repo_scan_']
    },
    {
      id: 'shell',
      labelKey: 'workspaceShell',
      fields: ['terminal.persistent_shell', 'terminal.env_passthrough']
    },
    {
      id: 'files',
      labelKey: 'workspaceFiles',
      fields: ['code_execution.mode', 'file_read_max_chars']
    }
  ],
  safety: [
    { id: 'approvals', labelKey: 'safetyApprovals', fields: ['command_allowlist'], prefixes: ['approvals.'] },
    { id: 'privacy', labelKey: 'safetyPrivacy', prefixes: ['security.'] },
    { id: 'checkpoints', labelKey: 'safetyCheckpoints', prefixes: ['checkpoints.'] }
  ],
  browser: [
    { id: 'profile', labelKey: 'browserProfile', fields: ['browser.use_real_profile'] },
    {
      id: 'network',
      labelKey: 'browserNetwork',
      fields: ['browser.allow_private_urls', 'browser.auto_local_for_private_urls']
    }
  ],
  memory: [
    { id: 'persistent', labelKey: 'memoryPersistent', prefixes: ['memory.'] },
    {
      id: 'context',
      labelKey: 'memoryContext',
      prefixes: ['context.', 'compression.', 'auxiliary.compression.']
    }
  ],
  voice: [
    {
      id: 'conversation',
      labelKey: 'voiceConversation',
      fields: ['voice.voice_chat_mode', 'voice.max_recording_seconds', 'voice.client_direct'],
      prefixes: ['voice.gpt_live.']
    },
    { id: 'transcription', labelKey: 'voiceTranscription', prefixes: ['stt.'] },
    { id: 'speech', labelKey: 'voiceSpeech', fields: ['voice.auto_tts'], prefixes: ['tts.'] }
  ],
  advanced: [
    { id: 'desktop', labelKey: 'advancedDesktop', prefixes: ['updates.'] },
    {
      id: 'runtime',
      labelKey: 'advancedRuntime',
      fields: ['agent.max_turns', 'agent.api_max_retries', 'agent.service_tier']
    },
    { id: 'tools', labelKey: 'advancedTools', fields: ['toolsets', 'agent.tool_use_enforcement'] },
    { id: 'terminal', labelKey: 'advancedTerminal', prefixes: ['terminal.'] },
    { id: 'delegation', labelKey: 'advancedDelegation', prefixes: ['delegation.'] },
    { id: 'output', labelKey: 'advancedOutput', prefixes: ['tool_output.', 'checkpoints.'] }
  ]
}

/** Ordered child pages; labelKey indexes t.settings.subpages. */
export const CONFIG_SUBPAGES: Record<string, { id: string; labelKey: string }[]> = Object.fromEntries(
  Object.entries(CONFIG_SUBPAGE_DEFINITIONS).map(([sectionId, pages]) => [
    sectionId,
    pages.map(({ id, labelKey }) => ({ id, labelKey }))
  ])
)

/** Resolve a schema/config field to the same child page that renders it. */
export function configSubpageForField(sectionId: string, field: string): string | undefined {
  return CONFIG_SUBPAGE_DEFINITIONS[sectionId]?.find(
    page => page.fields?.includes(field) || page.prefixes?.some(prefix => field.startsWith(prefix))
  )?.id
}
