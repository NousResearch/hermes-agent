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

// Provider group definitions used to fold raw env-var names like
// ``XAI_API_KEY`` into a single "xAI" card with a friendly label, short
// description, and signup URL. Membership is determined by longest
// prefix match (see ``providerGroup`` in helpers.ts) so more specific
// prefixes (``MINIMAX_CN_``) correctly beat their general parents
// (``MINIMAX_``). New providers should be added here so they get their
// own card in Settings → Keys instead of being lumped into "Other".
interface ProviderPrefix {
  prefix: string
  name: string
  /** Optional one-line tagline shown beneath the group name. */
  description?: string
  /** Optional canonical signup/console URL surfaced from the card header. */
  docsUrl?: string
  /** Lower numbers float to the top of the providers list. */
  priority: number
}

export const EMPTY_SELECT_VALUE = '__hermes_empty__'
export const CONTROL_TEXT = 'text-xs'

export const PROVIDER_GROUPS: ProviderPrefix[] = [
  {
    prefix: 'NOUS_',
    name: 'Nous Portal',
    description: 'Hosted Hermes & Nous-trained models',
    docsUrl: 'https://portal.nousresearch.com',
    priority: 0
  },
  {
    prefix: 'OPENROUTER_',
    name: 'OpenRouter',
    description: 'Aggregator for hundreds of frontier models',
    docsUrl: 'https://openrouter.ai/keys',
    priority: 1
  },
  {
    prefix: 'ANTHROPIC_',
    name: 'Anthropic',
    description: 'Claude API access (Sonnet, Opus, Haiku)',
    docsUrl: 'https://console.anthropic.com/settings/keys',
    priority: 2
  },
  {
    prefix: 'XAI_',
    name: 'xAI',
    description: 'Grok models (use OAuth for SuperGrok / Premium+)',
    docsUrl: 'https://console.x.ai/',
    priority: 3
  },
  {
    prefix: 'GOOGLE_',
    name: 'Gemini',
    description: 'Google AI Studio (Gemini 1.5 / 2.0 / 2.5)',
    docsUrl: 'https://aistudio.google.com/app/apikey',
    priority: 4
  },
  { prefix: 'GEMINI_', name: 'Gemini', priority: 4 },
  { prefix: 'HERMES_GEMINI_', name: 'Gemini', priority: 4 },
  {
    prefix: 'DEEPSEEK_',
    name: 'DeepSeek',
    description: 'Direct DeepSeek API (V3.x, R1)',
    docsUrl: 'https://platform.deepseek.com/api_keys',
    priority: 5
  },
  {
    prefix: 'DASHSCOPE_',
    name: 'DashScope (Qwen)',
    description: 'Alibaba Cloud DashScope — Qwen and multi-vendor models',
    docsUrl: 'https://modelstudio.console.alibabacloud.com/',
    priority: 6
  },
  { prefix: 'HERMES_QWEN_', name: 'DashScope (Qwen)', priority: 6 },
  {
    prefix: 'GLM_',
    name: 'GLM / Z.AI',
    description: 'Zhipu GLM-4.6 and Z.AI hosted endpoints',
    docsUrl: 'https://z.ai/',
    priority: 7
  },
  { prefix: 'ZAI_', name: 'GLM / Z.AI', priority: 7 },
  { prefix: 'Z_AI_', name: 'GLM / Z.AI', priority: 7 },
  {
    prefix: 'KIMI_',
    name: 'Kimi / Moonshot',
    description: 'Moonshot Kimi K2 / coding endpoints',
    docsUrl: 'https://platform.moonshot.cn/',
    priority: 8
  },
  {
    prefix: 'KIMI_CN_',
    name: 'Kimi (China)',
    description: 'Moonshot China endpoint',
    docsUrl: 'https://platform.moonshot.cn/',
    priority: 9
  },
  {
    prefix: 'MINIMAX_',
    name: 'MiniMax',
    description: 'MiniMax-M2 and Hailuo international endpoints',
    docsUrl: 'https://www.minimax.io/',
    priority: 10
  },
  {
    prefix: 'MINIMAX_CN_',
    name: 'MiniMax (China)',
    description: 'MiniMax mainland China endpoint',
    docsUrl: 'https://www.minimaxi.com/',
    priority: 11
  },
  {
    prefix: 'HF_',
    name: 'Hugging Face',
    description: 'Inference Providers — 20+ open models via router.huggingface.co',
    docsUrl: 'https://huggingface.co/settings/tokens',
    priority: 12
  },
  {
    prefix: 'OPENCODE_ZEN_',
    name: 'OpenCode Zen',
    description: 'Pay-as-you-go access to curated coding models',
    docsUrl: 'https://opencode.ai/auth',
    priority: 13
  },
  {
    prefix: 'OPENCODE_GO_',
    name: 'OpenCode Go',
    description: '$10/month subscription for open coding models',
    docsUrl: 'https://opencode.ai/auth',
    priority: 14
  },
  {
    prefix: 'NVIDIA_',
    name: 'NVIDIA NIM',
    description: 'build.nvidia.com or your own local NIM endpoint',
    docsUrl: 'https://build.nvidia.com/',
    priority: 15
  },
  {
    prefix: 'OLLAMA_',
    name: 'Ollama Cloud',
    description: 'Cloud-hosted open models from ollama.com',
    docsUrl: 'https://ollama.com/settings',
    priority: 16
  },
  {
    prefix: 'LM_',
    name: 'LM Studio',
    description: 'Local LM Studio server (OpenAI-compatible)',
    docsUrl: 'https://lmstudio.ai/docs/local-server',
    priority: 17
  },
  {
    prefix: 'STEPFUN_',
    name: 'StepFun',
    description: 'StepFun Step Plan coding models',
    docsUrl: 'https://platform.stepfun.com/',
    priority: 18
  },
  {
    prefix: 'XIAOMI_',
    name: 'Xiaomi MiMo',
    description: 'MiMo-V2.5 and Xiaomi proprietary models',
    docsUrl: 'https://platform.xiaomimimo.com',
    priority: 19
  },
  {
    prefix: 'ARCEEAI_',
    name: 'Arcee AI',
    description: 'Arcee-hosted small + medium models',
    docsUrl: 'https://chat.arcee.ai/',
    priority: 20
  },
  { prefix: 'ARCEE_', name: 'Arcee AI', priority: 20 },
  {
    prefix: 'GMI_',
    name: 'GMI Cloud',
    description: 'GMI Cloud GPU + model serving',
    docsUrl: 'https://www.gmicloud.ai/',
    priority: 21
  },
  {
    prefix: 'AZURE_FOUNDRY_',
    name: 'Azure Foundry',
    description: 'Azure AI Foundry custom endpoints (OpenAI / Anthropic-compatible)',
    docsUrl: 'https://ai.azure.com/',
    priority: 22
  },
  {
    prefix: 'AWS_',
    name: 'AWS Bedrock',
    description: 'Authenticate via AWS profile + region',
    docsUrl: 'https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html',
    priority: 23
  }
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

export const FIELD_LABELS: Record<string, string> = defineFieldCopy({
  model: 'Default Model',
  modelContextLength: 'Context Window',
  fallbackProviders: 'Fallback Models',
  toolsets: 'Enabled Toolsets',
  timezone: 'Timezone',
  display: {
    personality: 'Personality',
    showReasoning: 'Reasoning Blocks'
  },
  agent: {
    maxTurns: 'Max Agent Steps',
    imageInputMode: 'Image Attachments',
    apiMaxRetries: 'API Retries',
    serviceTier: 'Service Tier',
    toolUseEnforcement: 'Tool-Use Enforcement'
  },
  terminal: {
    cwd: 'Working Directory',
    backend: 'Execution Backend',
    timeout: 'Command Timeout',
    persistentShell: 'Persistent Shell',
    envPassthrough: 'Environment Passthrough'
  },
  fileReadMaxChars: 'File Read Limit',
  toolOutput: {
    maxBytes: 'Terminal Output Limit',
    maxLines: 'File Page Limit',
    maxLineLength: 'Line Length Limit'
  },
  codeExecution: {
    mode: 'Code Execution Mode'
  },
  approvals: {
    mode: 'Approval Mode',
    timeout: 'Approval Timeout',
    mcpReloadConfirm: 'Confirm MCP Reloads'
  },
  commandAllowlist: 'Command Allowlist',
  security: {
    redactSecrets: 'Redact Secrets',
    allowPrivateUrls: 'Allow Private URLs'
  },
  browser: {
    allowPrivateUrls: 'Browser Private URLs',
    autoLocalForPrivateUrls: 'Local Browser For Private URLs'
  },
  checkpoints: {
    enabled: 'File Checkpoints',
    maxSnapshots: 'Checkpoint Limit'
  },
  voice: {
    recordKey: 'Voice Shortcut',
    maxRecordingSeconds: 'Max Recording Length',
    autoTts: 'Read Responses Aloud'
  },
  stt: {
    enabled: 'Speech To Text',
    provider: 'Speech-To-Text Provider',
    local: {
      model: 'Local Transcription Model',
      language: 'Transcription Language'
    },
    elevenlabs: {
      modelId: 'ElevenLabs STT Model',
      languageCode: 'ElevenLabs Language',
      tagAudioEvents: 'Tag Audio Events',
      diarize: 'Speaker Diarization'
    }
  },
  tts: {
    provider: 'Text-To-Speech Provider',
    edge: {
      voice: 'Edge Voice'
    },
    openai: {
      model: 'OpenAI TTS Model',
      voice: 'OpenAI Voice'
    },
    elevenlabs: {
      voiceId: 'ElevenLabs Voice',
      modelId: 'ElevenLabs Model'
    }
  },
  memory: {
    memoryEnabled: 'Persistent Memory',
    userProfileEnabled: 'User Profile',
    memoryCharLimit: 'Memory Budget',
    userCharLimit: 'Profile Budget',
    provider: 'Memory Provider'
  },
  context: {
    engine: 'Context Engine'
  },
  compression: {
    enabled: 'Auto-Compression',
    threshold: 'Compression Threshold',
    targetRatio: 'Compression Target',
    protectLastN: 'Protected Recent Messages'
  },
  delegation: {
    model: 'Subagent Model',
    provider: 'Subagent Provider',
    maxIterations: 'Subagent Turn Limit',
    maxConcurrentChildren: 'Parallel Subagents',
    childTimeoutSeconds: 'Subagent Timeout',
    reasoningEffort: 'Subagent Reasoning Effort'
  },
  updates: {
    nonInteractiveLocalChanges: 'In-App Update Local Changes'
  }
})

export const FIELD_DESCRIPTIONS: Record<string, string> = defineFieldCopy({
  model: 'Used for new chats unless you pick a different model in the composer.',
  modelContextLength: "Leave at 0 to use the selected model's detected context window.",
  fallbackProviders: 'Backup provider:model entries to try if the default model fails.',
  display: {
    personality: 'Default assistant style for new sessions.',
    showReasoning: 'Show reasoning sections when the backend provides them.'
  },
  timezone: 'Used when Hermes needs local time context. Blank uses the system timezone.',
  agent: {
    imageInputMode: 'Controls how image attachments are sent to the model.',
    maxTurns: 'Upper bound for tool-calling turns before Hermes stops a run.'
  },
  terminal: {
    cwd: 'Default project folder for tool and terminal work.',
    persistentShell: 'Keep shell state between commands when the backend supports it.',
    envPassthrough: 'Environment variables to pass into tool execution.'
  },
  codeExecution: {
    mode: 'How strictly code execution is scoped to the current project.'
  },
  fileReadMaxChars: 'Maximum characters Hermes can read from one file request.',
  approvals: {
    mode: 'How Hermes handles commands that need explicit approval.',
    timeout: 'How long approval prompts wait before timing out.'
  },
  security: {
    redactSecrets: 'Hide detected secrets from model-visible content when possible.'
  },
  checkpoints: {
    enabled: 'Create rollback snapshots before file edits.'
  },
  memory: {
    memoryEnabled: 'Save durable memories that can help future sessions.',
    userProfileEnabled: 'Maintain a compact profile of user preferences.'
  },
  context: {
    engine: 'Strategy for managing long conversations near the context limit.'
  },
  compression: {
    enabled: 'Summarize older context when conversations get large.'
  },
  voice: {
    autoTts: 'Automatically speak assistant responses.'
  },
  stt: {
    enabled: 'Enable local or provider-backed speech transcription.',
    elevenlabs: {
      languageCode: 'Optional ISO-639-3 language code. Blank lets ElevenLabs auto-detect.'
    }
  },
  updates: {
    nonInteractiveLocalChanges:
      'When Hermes updates itself from the app (no terminal prompt), keep local source edits (stash) or throw them away (discard). Terminal updates always ask.'
  }
})

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
      'delegation.reasoning_effort',
      'updates.non_interactive_local_changes'
    ]
  }
]

export interface ModeOption {
  id: ThemeMode
  label: string
  icon: IconComponent
}

export const MODE_OPTIONS: ModeOption[] = [
  { id: 'light', label: 'Light', icon: Sun },
  { id: 'dark', label: 'Dark', icon: Moon },
  { id: 'system', label: 'System', icon: Monitor }
]
