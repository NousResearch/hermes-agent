import {
  Brain,
  type IconComponent,
  Lock,
  MessageCircle,
  Mic,
  Monitor,
  Moon,
  Palette,
  Sparkles,
  Sun,
  Wrench
} from '@/lib/icons'
import type { ThemeMode } from '@/themes/context'

import type { DesktopConfigSection } from './types'
import { defineFieldCopy } from './field-copy'

interface ProviderPrefix {
  prefix: string
  name: string
  priority: number
}

export const EMPTY_SELECT_VALUE = '__hermes_empty__'
export const CONTROL_TEXT = 'text-[0.8125rem]'

export const PROVIDER_GROUPS: ProviderPrefix[] = [
  { prefix: 'NOUS_', name: 'Nous Portal', priority: 0 },
  { prefix: 'ANTHROPIC_', name: 'Anthropic', priority: 1 },
  { prefix: 'DASHSCOPE_', name: 'DashScope (Qwen)', priority: 2 },
  { prefix: 'HERMES_QWEN_', name: 'DashScope (Qwen)', priority: 2 },
  { prefix: 'DEEPSEEK_', name: 'DeepSeek', priority: 3 },
  { prefix: 'GOOGLE_', name: 'Gemini', priority: 4 },
  { prefix: 'GEMINI_', name: 'Gemini', priority: 4 },
  { prefix: 'GLM_', name: 'GLM / Z.AI', priority: 5 },
  { prefix: 'ZAI_', name: 'GLM / Z.AI', priority: 5 },
  { prefix: 'Z_AI_', name: 'GLM / Z.AI', priority: 5 },
  { prefix: 'HF_', name: 'Hugging Face', priority: 6 },
  { prefix: 'KIMI_', name: 'Kimi / Moonshot', priority: 7 },
  { prefix: 'MINIMAX_', name: 'MiniMax', priority: 8 },
  { prefix: 'MINIMAX_CN_', name: 'MiniMax (China)', priority: 9 },
  { prefix: 'OPENCODE_GO_', name: 'OpenCode Go', priority: 10 },
  { prefix: 'OPENCODE_ZEN_', name: 'OpenCode Zen', priority: 11 },
  { prefix: 'OPENROUTER_', name: 'OpenRouter', priority: 12 },
  { prefix: 'XIAOMI_', name: 'Xiaomi MiMo', priority: 13 }
]

export const BUILTIN_PERSONALITIES = [
  'helpful',
  'concise',
  'technical',
  'creative',
  'teacher',
  'kawaii',
  'catgirl',
  'pirate',
  'shakespeare',
  'surfer',
  'noir',
  'uwu',
  'philosopher',
  'hype'
]

// Schema-side select overrides for desktop-relevant enum fields whose
// backend schema only declares a string type.
export const ENUM_OPTIONS: Record<string, string[]> = {
  'agent.image_input_mode': ['auto', 'native', 'text'],
  'approvals.mode': ['manual', 'smart', 'off'],
  'code_execution.mode': ['project', 'strict'],
  'context.engine': ['compressor', 'default', 'custom'],
  'delegation.reasoning_effort': ['', 'minimal', 'low', 'medium', 'high', 'xhigh'],
  'memory.provider': ['', 'builtin', 'honcho'],
  'stt.elevenlabs.model_id': ['scribe_v2', 'scribe_v1'],
  'stt.local.model': ['tiny', 'base', 'small', 'medium', 'large-v3'],
  'tts.openai.voice': ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
  'updates.non_interactive_local_changes': ['stash', 'discard']
}

export const FIELD_LABELS: Record<string, string> = {
  model: 'config.label.defaultModel',
  model_context_length: 'config.label.contextWindow',
  fallback_providers: 'config.label.fallbackModels',
  toolsets: 'config.label.enabledToolsets',
  timezone: 'config.label.timezone',
  'display.personality': 'config.label.personality',
  'display.show_reasoning': 'config.label.reasoningBlocks',
  'agent.max_turns': 'config.label.maxAgentSteps',
  'agent.image_input_mode': 'config.label.imageAttachments',
  'terminal.cwd': 'config.label.workingDirectory',
  'terminal.backend': 'config.label.executionBackend',
  'terminal.timeout': 'config.label.commandTimeout',
  'terminal.persistent_shell': 'config.label.persistentShell',
  'terminal.env_passthrough': 'config.label.envPassthrough',
  file_read_max_chars: 'config.label.fileReadLimit',
  'tool_output.max_bytes': 'config.label.terminalOutputLimit',
  'tool_output.max_lines': 'config.label.filePageLimit',
  'tool_output.max_line_length': 'config.label.lineLengthLimit',
  'code_execution.mode': 'config.label.codeExecutionMode',
  'approvals.mode': 'config.label.approvalMode',
  'approvals.timeout': 'config.label.approvalTimeout',
  'approvals.mcp_reload_confirm': 'config.label.confirmMcpReloads',
  command_allowlist: 'config.label.commandAllowlist',
  'security.redact_secrets': 'config.label.redactSecrets',
  'security.allow_private_urls': 'config.label.allowPrivateUrls',
  'browser.allow_private_urls': 'config.label.browserPrivateUrls',
  'browser.auto_local_for_private_urls': 'config.label.localBrowserForPrivateUrls',
  'checkpoints.enabled': 'config.label.fileCheckpoints',
  'checkpoints.max_snapshots': 'config.label.checkpointLimit',
  'voice.record_key': 'config.label.voiceShortcut',
  'voice.max_recording_seconds': 'config.label.maxRecordingLength',
  'voice.auto_tts': 'config.label.readResponsesAloud',
  'stt.enabled': 'config.label.speechToText',
  'stt.provider': 'config.label.speechToTextProvider',
  'stt.local.model': 'config.label.localTranscriptionModel',
  'stt.local.language': 'config.label.transcriptionLanguage',
  'stt.elevenlabs.model_id': 'config.label.elevenlabsSttModel',
  'stt.elevenlabs.language_code': 'config.label.elevenlabsLanguage',
  'stt.elevenlabs.tag_audio_events': 'config.label.tagAudioEvents',
  'stt.elevenlabs.diarize': 'config.label.speakerDiarization',
  'tts.provider': 'config.label.textToSpeechProvider',
  'tts.edge.voice': 'config.label.edgeVoice',
  'tts.openai.model': 'config.label.openaiTtsModel',
  'tts.openai.voice': 'config.label.openaiVoice',
  'tts.elevenlabs.voice_id': 'config.label.elevenlabsVoice',
  'tts.elevenlabs.model_id': 'config.label.elevenlabsModel',
  'memory.memory_enabled': 'config.label.persistentMemory',
  'memory.user_profile_enabled': 'config.label.userProfile',
  'memory.memory_char_limit': 'config.label.memoryBudget',
  'memory.user_char_limit': 'config.label.profileBudget',
  'memory.provider': 'config.label.memoryProvider',
  'context.engine': 'config.label.contextEngine',
  'compression.enabled': 'config.label.autoCompression',
  'compression.threshold': 'config.label.compressionThreshold',
  'compression.target_ratio': 'config.label.compressionTarget',
  'compression.protect_last_n': 'config.label.protectedRecentMessages',
  'agent.api_max_retries': 'config.label.apiRetries',
  'agent.service_tier': 'config.label.serviceTier',
  'agent.tool_use_enforcement': 'config.label.toolUseEnforcement',
  'delegation.model': 'config.label.subagentModel',
  'delegation.provider': 'config.label.subagentProvider',
  'delegation.max_iterations': 'config.label.subagentTurnLimit',
  'delegation.max_concurrent_children': 'config.label.parallelSubagents',
  'delegation.child_timeout_seconds': 'config.label.subagentTimeout',
  'delegation.reasoning_effort': 'config.label.subagentReasoningEffort'
}

export const FIELD_DESCRIPTIONS: Record<string, string> = {
  model: 'config.desc.defaultModel',
  model_context_length: 'config.desc.contextWindow',
  fallback_providers: 'config.desc.fallbackModels',
  'display.personality': 'config.desc.personality',
  timezone: 'config.desc.timezone',
  'display.show_reasoning': 'config.desc.reasoningBlocks',
  'agent.image_input_mode': 'config.desc.imageAttachments',
  'terminal.cwd': 'config.desc.workingDirectory',
  'code_execution.mode': 'config.desc.codeExecutionMode',
  'terminal.persistent_shell': 'config.desc.persistentShell',
  'terminal.env_passthrough': 'config.desc.envPassthrough',
  file_read_max_chars: 'config.desc.fileReadLimit',
  'approvals.mode': 'config.desc.approvalMode',
  'approvals.timeout': 'config.desc.approvalTimeout',
  'security.redact_secrets': 'config.desc.redactSecrets',
  'checkpoints.enabled': 'config.desc.fileCheckpoints',
  'memory.memory_enabled': 'config.desc.persistentMemory',
  'memory.user_profile_enabled': 'config.desc.userProfile',
  'context.engine': 'config.desc.contextEngine',
  'compression.enabled': 'config.desc.autoCompression',
  'voice.auto_tts': 'config.desc.readResponsesAloud',
  'stt.enabled': 'config.desc.speechToText',
  'stt.elevenlabs.language_code': 'config.desc.elevenlabsLanguage',
  'agent.max_turns': 'config.desc.maxAgentSteps'
}

// Curated desktop config surface: only fields a user might tune from the app.
export const SECTIONS: DesktopConfigSection[] = [
  {
    id: 'model',
    label: 'Model',
    labelKey: 'settings.model',
    icon: Sparkles,
    keys: ['model_context_length', 'fallback_providers']
  },
  {
    id: 'chat',
    label: 'Chat',
    labelKey: 'settings.section.chat',
    icon: MessageCircle,
    keys: ['display.personality', 'timezone', 'display.show_reasoning', 'agent.image_input_mode']
  },
  {
    id: 'appearance',
    label: 'Appearance',
    labelKey: 'settings.appearance',
    icon: Palette,
    keys: []
  },
  {
    id: 'workspace',
    label: 'Workspace',
    labelKey: 'settings.section.workspace',
    icon: Monitor,
    keys: [
      'terminal.cwd',
      'code_execution.mode',
      'terminal.persistent_shell',
      'terminal.env_passthrough',
      'file_read_max_chars'
    ]
  },
  {
    id: 'safety',
    label: 'Safety',
    labelKey: 'settings.section.safety',
    icon: Lock,
    keys: [
      'approvals.mode',
      'approvals.timeout',
      'approvals.mcp_reload_confirm',
      'command_allowlist',
      'security.redact_secrets',
      'security.allow_private_urls',
      'browser.allow_private_urls',
      'browser.auto_local_for_private_urls',
      'checkpoints.enabled'
    ]
  },
  {
    id: 'memory',
    label: 'Memory & Context',
    labelKey: 'settings.section.memory',
    icon: Brain,
    keys: [
      'memory.memory_enabled',
      'memory.user_profile_enabled',
      'memory.memory_char_limit',
      'memory.user_char_limit',
      'memory.provider',
      'context.engine',
      'compression.enabled',
      'compression.threshold',
      'compression.target_ratio',
      'compression.protect_last_n'
    ]
  },
  {
    id: 'voice',
    label: 'Voice',
    labelKey: 'settings.section.voice',
    icon: Mic,
    keys: [
      'tts.provider',
      'stt.enabled',
      'stt.provider',
      'voice.auto_tts',
      'tts.edge.voice',
      'tts.openai.model',
      'tts.openai.voice',
      'tts.elevenlabs.voice_id',
      'tts.elevenlabs.model_id',
      'stt.local.model',
      'stt.local.language',
      'stt.elevenlabs.model_id',
      'stt.elevenlabs.language_code',
      'stt.elevenlabs.tag_audio_events',
      'stt.elevenlabs.diarize',
      'voice.record_key',
      'voice.max_recording_seconds'
    ]
  },
  {
    id: 'advanced',
    label: 'Advanced',
    labelKey: 'settings.section.advanced',
    icon: Wrench,
    keys: [
      'toolsets',
      'terminal.backend',
      'terminal.timeout',
      'tool_output.max_bytes',
      'tool_output.max_lines',
      'tool_output.max_line_length',
      'checkpoints.max_snapshots',
      'agent.max_turns',
      'agent.api_max_retries',
      'agent.service_tier',
      'agent.tool_use_enforcement',
      'delegation.model',
      'delegation.provider',
      'delegation.max_iterations',
      'delegation.max_concurrent_children',
      'delegation.child_timeout_seconds',
      'delegation.reasoning_effort',
      'updates.non_interactive_local_changes'
    ]
  }
]

export interface ModeOption {
  id: ThemeMode
  label: string
  description: string
  icon: IconComponent
}

export const MODE_OPTIONS: ModeOption[] = [
  { id: 'light', label: 'appearance.mode.light', description: 'appearance.mode.lightDesc', icon: Sun },
  { id: 'dark', label: 'appearance.mode.dark', description: 'appearance.mode.darkDesc', icon: Moon },
  { id: 'system', label: 'appearance.mode.system', description: 'appearance.mode.systemDesc', icon: Monitor }
]

export const SEARCH_PLACEHOLDER: Record<'about' | 'config' | 'gateway' | 'keys' | 'mcp' | 'sessions', string> = {
  about: 'About Hermes Desktop',
  config: 'Search settings...',
  gateway: 'Gateway connection...',
  keys: 'Search API keys...',
  mcp: 'Search MCP servers...',
  sessions: 'Search archived sessions...'
}
