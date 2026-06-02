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
  'tts.openai.voice': ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
}

export const FIELD_LABELS: Record<string, string> = {
  model: 'settings:field.model',
  model_context_length: 'settings:field.model_context_length',
  fallback_providers: 'settings:field.fallback_providers',
  toolsets: 'settings:field.toolsets',
  timezone: 'settings:field.timezone',
  'display.personality': 'settings:field.display.personality',
  'display.show_reasoning': 'settings:field.display.show_reasoning',
  'agent.max_turns': 'settings:field.agent.max_turns',
  'agent.image_input_mode': 'settings:field.agent.image_input_mode',
  'terminal.cwd': 'settings:field.terminal.cwd',
  'terminal.backend': 'settings:field.terminal.backend',
  'terminal.timeout': 'settings:field.terminal.timeout',
  'terminal.persistent_shell': 'settings:field.terminal.persistent_shell',
  'terminal.env_passthrough': 'settings:field.terminal.env_passthrough',
  file_read_max_chars: 'settings:field.file_read_max_chars',
  'tool_output.max_bytes': 'settings:field.tool_output.max_bytes',
  'tool_output.max_lines': 'settings:field.tool_output.max_lines',
  'tool_output.max_line_length': 'settings:field.tool_output.max_line_length',
  'code_execution.mode': 'settings:field.code_execution.mode',
  'approvals.mode': 'settings:field.approvals.mode',
  'approvals.timeout': 'settings:field.approvals.timeout',
  'approvals.mcp_reload_confirm': 'settings:field.approvals.mcp_reload_confirm',
  command_allowlist: 'settings:field.command_allowlist',
  'security.redact_secrets': 'settings:field.security.redact_secrets',
  'security.allow_private_urls': 'settings:field.security.allow_private_urls',
  'browser.allow_private_urls': 'settings:field.browser.allow_private_urls',
  'browser.auto_local_for_private_urls': 'settings:field.browser.auto_local_for_private_urls',
  'checkpoints.enabled': 'settings:field.checkpoints.enabled',
  'checkpoints.max_snapshots': 'settings:field.checkpoints.max_snapshots',
  'voice.record_key': 'settings:field.voice.record_key',
  'voice.max_recording_seconds': 'settings:field.voice.max_recording_seconds',
  'voice.auto_tts': 'settings:field.voice.auto_tts',
  'stt.enabled': 'settings:field.stt.enabled',
  'stt.provider': 'settings:field.stt.provider',
  'stt.local.model': 'settings:field.stt.local.model',
  'stt.local.language': 'settings:field.stt.local.language',
  'stt.elevenlabs.model_id': 'settings:field.stt.elevenlabs.model_id',
  'stt.elevenlabs.language_code': 'settings:field.stt.elevenlabs.language_code',
  'stt.elevenlabs.tag_audio_events': 'settings:field.stt.elevenlabs.tag_audio_events',
  'stt.elevenlabs.diarize': 'settings:field.stt.elevenlabs.diarize',
  'tts.provider': 'settings:field.tts.provider',
  'tts.edge.voice': 'settings:field.tts.edge.voice',
  'tts.openai.model': 'settings:field.tts.openai.model',
  'tts.openai.voice': 'settings:field.tts.openai.voice',
  'tts.elevenlabs.voice_id': 'settings:field.tts.elevenlabs.voice_id',
  'tts.elevenlabs.model_id': 'settings:field.tts.elevenlabs.model_id',
  'memory.memory_enabled': 'settings:field.memory.memory_enabled',
  'memory.user_profile_enabled': 'settings:field.memory.user_profile_enabled',
  'memory.memory_char_limit': 'settings:field.memory.memory_char_limit',
  'memory.user_char_limit': 'settings:field.memory.user_char_limit',
  'memory.provider': 'settings:field.memory.provider',
  'context.engine': 'settings:field.context.engine',
  'compression.enabled': 'settings:field.compression.enabled',
  'compression.threshold': 'settings:field.compression.threshold',
  'compression.target_ratio': 'settings:field.compression.target_ratio',
  'compression.protect_last_n': 'settings:field.compression.protect_last_n',
  'agent.api_max_retries': 'settings:field.agent.api_max_retries',
  'agent.service_tier': 'settings:field.agent.service_tier',
  'agent.tool_use_enforcement': 'settings:field.agent.tool_use_enforcement',
  'delegation.model': 'settings:field.delegation.model',
  'delegation.provider': 'settings:field.delegation.provider',
  'delegation.max_iterations': 'settings:field.delegation.max_iterations',
  'delegation.max_concurrent_children': 'settings:field.delegation.max_concurrent_children',
  'delegation.child_timeout_seconds': 'settings:field.delegation.child_timeout_seconds',
  'delegation.reasoning_effort': 'settings:field.delegation.reasoning_effort'
}

export const FIELD_DESCRIPTIONS: Record<string, string> = {
  model: 'settings:fieldDesc.model',
  model_context_length: 'settings:fieldDesc.model_context_length',
  fallback_providers: 'settings:fieldDesc.fallback_providers',
  'display.personality': 'settings:fieldDesc.display.personality',
  timezone: 'settings:fieldDesc.timezone',
  'display.show_reasoning': 'settings:fieldDesc.display.show_reasoning',
  'agent.image_input_mode': 'settings:fieldDesc.agent.image_input_mode',
  'terminal.cwd': 'settings:fieldDesc.terminal.cwd',
  'code_execution.mode': 'settings:fieldDesc.code_execution.mode',
  'terminal.persistent_shell': 'settings:fieldDesc.terminal.persistent_shell',
  'terminal.env_passthrough': 'settings:fieldDesc.terminal.env_passthrough',
  file_read_max_chars: 'settings:fieldDesc.file_read_max_chars',
  'approvals.mode': 'settings:fieldDesc.approvals.mode',
  'approvals.timeout': 'settings:fieldDesc.approvals.timeout',
  'security.redact_secrets': 'settings:fieldDesc.security.redact_secrets',
  'checkpoints.enabled': 'settings:fieldDesc.checkpoints.enabled',
  'memory.memory_enabled': 'settings:fieldDesc.memory.memory_enabled',
  'memory.user_profile_enabled': 'settings:fieldDesc.memory.user_profile_enabled',
  'context.engine': 'settings:fieldDesc.context.engine',
  'compression.enabled': 'settings:fieldDesc.compression.enabled',
  'voice.auto_tts': 'settings:fieldDesc.voice.auto_tts',
  'stt.enabled': 'settings:fieldDesc.stt.enabled',
  'stt.elevenlabs.language_code': 'settings:fieldDesc.stt.elevenlabs.language_code',
  'agent.max_turns': 'settings:fieldDesc.agent.max_turns'
}

// Curated desktop config surface: only fields a user might tune from the app.
export const SECTIONS: DesktopConfigSection[] = [
  {
    id: 'model',
    label: 'settings:section.model',
    icon: Sparkles,
    keys: ['model_context_length', 'fallback_providers']
  },
  {
    id: 'chat',
    label: 'settings:section.chat',
    icon: MessageCircle,
    keys: ['display.personality', 'timezone', 'display.show_reasoning', 'agent.image_input_mode']
  },
  {
    id: 'appearance',
    label: 'settings:section.appearance',
    icon: Palette,
    keys: []
  },
  {
    id: 'workspace',
    label: 'settings:section.workspace',
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
    label: 'settings:section.safety',
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
    label: 'settings:section.memory',
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
    label: 'settings:section.voice',
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
    label: 'settings:section.advanced',
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
      'delegation.reasoning_effort'
    ]
  }
]

export interface ModeOption {
  id: ThemeMode
  /** i18n key under the `settings` namespace. */
  labelKey: string
  /** i18n key under the `settings` namespace. */
  descriptionKey: string
  icon: IconComponent
}

export const MODE_OPTIONS: ModeOption[] = [
  {
    id: 'light',
    labelKey: 'settings:appearance.mode.light',
    descriptionKey: 'settings:appearance.mode.lightDescription',
    icon: Sun
  },
  {
    id: 'dark',
    labelKey: 'settings:appearance.mode.dark',
    descriptionKey: 'settings:appearance.mode.darkDescription',
    icon: Moon
  },
  {
    id: 'system',
    labelKey: 'settings:appearance.mode.system',
    descriptionKey: 'settings:appearance.mode.systemDescription',
    icon: Monitor
  }
]

export const SEARCH_PLACEHOLDER: Record<'about' | 'config' | 'gateway' | 'keys' | 'mcp' | 'sessions' | 'tools',string> = {
  about: 'settings:search.about',
  config: 'settings:search.placeholder',
  gateway: 'settings:search.gateway',
  keys: 'settings:search.keys',
  mcp: 'settings:search.mcp',
  sessions: 'settings:search.sessions'
}
