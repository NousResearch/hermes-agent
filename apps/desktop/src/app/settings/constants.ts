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
export const CONTROL_TEXT = 'text-xs'

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
  model: 'Default Model',
  model_context_length: 'Context Window',
  fallback_providers: 'Fallback Models',
  toolsets: 'Enabled Toolsets',
  timezone: 'Timezone',
  'display.personality': 'Personality',
  'display.show_reasoning': 'Reasoning Blocks',
  'agent.max_turns': 'Max Agent Steps',
  'agent.image_input_mode': 'Image Attachments',
  'terminal.cwd': 'Working Directory',
  'terminal.backend': 'Execution Backend',
  'terminal.timeout': 'Command Timeout',
  'terminal.persistent_shell': 'Persistent Shell',
  'terminal.env_passthrough': 'Environment Passthrough',
  file_read_max_chars: 'File Read Limit',
  'tool_output.max_bytes': 'Terminal Output Limit',
  'tool_output.max_lines': 'File Page Limit',
  'tool_output.max_line_length': 'Line Length Limit',
  'code_execution.mode': 'Code Execution Mode',
  'approvals.mode': 'Approval Mode',
  'approvals.timeout': 'Approval Timeout',
  'approvals.mcp_reload_confirm': 'Confirm MCP Reloads',
  command_allowlist: 'Command Allowlist',
  'security.redact_secrets': 'Redact Secrets',
  'security.allow_private_urls': 'Allow Private URLs',
  'browser.allow_private_urls': 'Browser Private URLs',
  'browser.auto_local_for_private_urls': 'Local Browser For Private URLs',
  'checkpoints.enabled': 'File Checkpoints',
  'checkpoints.max_snapshots': 'Checkpoint Limit',
  'voice.record_key': 'Voice Shortcut',
  'voice.max_recording_seconds': 'Max Recording Length',
  'voice.auto_tts': 'Read Responses Aloud',
  'stt.enabled': 'Speech To Text',
  'stt.provider': 'Speech-To-Text Provider',
  'stt.local.model': 'Local Transcription Model',
  'stt.local.language': 'Transcription Language',
  'stt.elevenlabs.model_id': 'ElevenLabs STT Model',
  'stt.elevenlabs.language_code': 'ElevenLabs Language',
  'stt.elevenlabs.tag_audio_events': 'Tag Audio Events',
  'stt.elevenlabs.diarize': 'Speaker Diarization',
  'tts.provider': 'Text-To-Speech Provider',
  'tts.edge.voice': 'Edge Voice',
  'tts.openai.model': 'OpenAI TTS Model',
  'tts.openai.voice': 'OpenAI Voice',
  'tts.elevenlabs.voice_id': 'ElevenLabs Voice',
  'tts.elevenlabs.model_id': 'ElevenLabs Model',
  'memory.memory_enabled': 'Persistent Memory',
  'memory.user_profile_enabled': 'User Profile',
  'memory.memory_char_limit': 'Memory Budget',
  'memory.user_char_limit': 'Profile Budget',
  'memory.provider': 'Memory Provider',
  'context.engine': 'Context Engine',
  'compression.enabled': 'Auto-Compression',
  'compression.threshold': 'Compression Threshold',
  'compression.target_ratio': 'Compression Target',
  'compression.protect_last_n': 'Protected Recent Messages',
  'agent.api_max_retries': 'API Retries',
  'agent.service_tier': 'Service Tier',
  'agent.tool_use_enforcement': 'Tool-Use Enforcement',
  'delegation.model': 'Subagent Model',
  'delegation.provider': 'Subagent Provider',
  'delegation.max_iterations': 'Subagent Turn Limit',
  'delegation.max_concurrent_children': 'Parallel Subagents',
  'delegation.child_timeout_seconds': 'Subagent Timeout',
  'delegation.reasoning_effort': 'Subagent Reasoning Effort'
}

export const FIELD_DESCRIPTIONS: Record<string, string> = {
  model: 'Used for new chats unless you pick a different model in the composer.',
  model_context_length: "Leave at 0 to use the selected model's detected context window.",
  fallback_providers: 'Backup provider:model entries to try if the default model fails.',
  'display.personality': 'Default assistant style for new sessions.',
  timezone: 'Used when Hermes needs local time context. Blank uses the system timezone.',
  'display.show_reasoning': 'Show reasoning sections when the backend provides them.',
  'agent.image_input_mode': 'Controls how image attachments are sent to the model.',
  'terminal.cwd': 'Default project folder for tool and terminal work.',
  'code_execution.mode': 'How strictly code execution is scoped to the current project.',
  'terminal.persistent_shell': 'Keep shell state between commands when the backend supports it.',
  'terminal.env_passthrough': 'Environment variables to pass into tool execution.',
  file_read_max_chars: 'Maximum characters Hermes can read from one file request.',
  'approvals.mode': 'How Hermes handles commands that need explicit approval.',
  'approvals.timeout': 'How long approval prompts wait before timing out.',
  'security.redact_secrets': 'Hide detected secrets from model-visible content when possible.',
  'checkpoints.enabled': 'Create rollback snapshots before file edits.',
  'memory.memory_enabled': 'Save durable memories that can help future sessions.',
  'memory.user_profile_enabled': 'Maintain a compact profile of user preferences.',
  'context.engine': 'Strategy for managing long conversations near the context limit.',
  'compression.enabled': 'Summarize older context when conversations get large.',
  'voice.auto_tts': 'Automatically speak assistant responses.',
  'stt.enabled': 'Enable local or provider-backed speech transcription.',
  'stt.elevenlabs.language_code': 'Optional ISO-639-3 language code. Blank lets ElevenLabs auto-detect.',
  'agent.max_turns': 'Upper bound for tool-calling turns before Hermes stops a run.'
}

// Curated desktop config surface: only fields a user might tune from the app.
export const SECTIONS: DesktopConfigSection[] = [
  {
    id: 'model',
    label: 'Model',
    icon: Sparkles,
    keys: ['model_context_length', 'fallback_providers']
  },
  {
    id: 'chat',
    label: 'Chat',
    icon: MessageCircle,
    keys: ['display.personality', 'timezone', 'display.show_reasoning', 'agent.image_input_mode']
  },
  {
    id: 'appearance',
    label: 'Appearance',
    icon: Palette,
    keys: []
  },
  {
    id: 'workspace',
    label: 'Workspace',
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
  label: string
  description: string
  icon: IconComponent
}

export const MODE_OPTIONS: ModeOption[] = [
  { id: 'light', label: 'Light', description: 'Bright desktop surfaces', icon: Sun },
  { id: 'dark', label: 'Dark', description: 'Low-glare workspace', icon: Moon },
  { id: 'system', label: 'System', description: 'Follow OS appearance', icon: Monitor }
]

export const SEARCH_PLACEHOLDER: Record<'about' | 'config' | 'gateway' | 'keys' | 'mcp' | 'sessions', string> = {
  about: 'About Hermes Desktop',
  config: 'Search settings...',
  gateway: 'Gateway connection...',
  keys: 'Search API keys...',
  mcp: 'Search MCP servers...',
  sessions: 'Search archived sessions...'
}
