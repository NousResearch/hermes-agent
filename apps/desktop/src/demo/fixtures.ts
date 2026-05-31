// Canned data for demo mode. Plain objects — the REST shim returns them as the
// generic api<T> result and consumers cast, so we don't need to import every
// runtime type here. Keep this small and representative.

const now = Date.now()
const min = 60_000
const hr = 60 * min

const DEFAULT_PROVIDER = 'anthropic'
const DEFAULT_MODEL = 'claude-sonnet-4.5'

function session(
  id: string,
  title: string,
  preview: string,
  message_count: number,
  ago: number,
  model: string,
  tool_call_count: number,
  cwd = '~/code/hermes-agent'
) {
  return {
    id,
    title,
    preview,
    message_count,
    started_at: now - ago - 20 * min,
    last_active: now - ago,
    ended_at: now - ago,
    is_active: false,
    model,
    input_tokens: 1800 + message_count * 420,
    output_tokens: 900 + message_count * 260,
    tool_call_count,
    cwd,
    source: 'desktop'
  }
}

export const SESSIONS = [
  session(
    's-release',
    'Cut the desktop release notes',
    'Drafted the changelog and opened the PR.',
    18,
    2 * min,
    DEFAULT_MODEL,
    8
  ),
  session(
    's-ci',
    'Debug failing CI on arm64',
    'The wheel build needed a platform tag bump.',
    24,
    38 * min,
    'gpt-5',
    12
  ),
  session(
    's-auth',
    'Refactor auth middleware',
    'Extracted the token guard into its own module.',
    31,
    3 * hr,
    DEFAULT_MODEL,
    15,
    '~/work/api'
  ),
  session(
    's-rfc',
    'Summarize the 42-page sync RFC',
    'Three options; I recommend the event-log design.',
    9,
    6 * hr,
    'gpt-5',
    2,
    '~/work/specs'
  )
]

const message = (role: string, content: string, ago: number) => ({ role, content, timestamp: now - ago })

export const MESSAGES: Record<string, Array<Record<string, unknown>>> = {
  's-release': [
    message(
      'user',
      'We just merged the desktop GUI. Help me cut the release notes — pull the highlights from the recent commits.',
      40 * min
    ),
    message('assistant', 'On it. Let me look at what landed since the last release.', 39 * min),
    {
      role: 'assistant',
      timestamp: now - 39 * min,
      content: '',
      tool_calls: [
        {
          id: 'h1',
          type: 'function',
          function: { name: 'run_command', arguments: JSON.stringify({ command: 'git log --oneline --since=2.weeks' }) }
        }
      ]
    },
    {
      role: 'tool',
      tool_call_id: 'h1',
      tool_name: 'run_command',
      timestamp: now - 38 * min,
      content:
        '4cfc8b7 feat(desktop): native Electron shell\n9a21f0e feat(desktop): live gateway streaming\n1d7e220 feat(themes): theme engine\n…and 137 more'
    },
    message(
      'assistant',
      "Here's the shape of it — a native desktop app with live streaming chat, inline tool cards, a theme engine, a command palette, and one-click updates. Want me to draft the full changelog and open a PR?",
      38 * min
    )
  ]
}

// A streamable turn for window.__demo.playTurn(). The gateway replays these as
// server→client events (reasoning → tool → reply), the same shapes the live
// backend emits.
export const DEMO_TURN = {
  prompt: 'Write a punchy launch tweet for the desktop app.',
  reasoning:
    'They want a launch tweet for the new desktop GUI. Keep it hook-first, concrete, end with a call to action.',
  tool: {
    name: 'read_file',
    tool_id: 'demo-1',
    args: { path: 'RELEASE_NOTES.md' },
    result: {
      output:
        '# Hermes Desktop\n\n## Highlights\n- Native desktop app\n- Streaming chat + inline tool cards\n- Theme engine\n…',
      exit_code: 0
    }
  },
  reply: [
    'Your agent has a desktop now. ',
    'Native app, **streaming chat**, inline tool calls, a theme engine, ',
    'and one-click updates.\n\nSame Hermes. Now with a window. → Download today.'
  ]
}

export const CONFIG = {
  agent: { reasoning_effort: 'medium', service_tier: 'default', personalities: { hermes: {} } },
  display: { personality: 'hermes', skin: 'nous' },
  terminal: { cwd: '~/code/hermes-agent' },
  stt: { enabled: true },
  voice: { max_recording_seconds: 120 }
}

export const MODEL_OPTIONS = {
  provider: DEFAULT_PROVIDER,
  model: DEFAULT_MODEL,
  providers: [
    {
      name: 'Anthropic',
      slug: 'anthropic',
      is_current: true,
      total_models: 2,
      models: ['claude-sonnet-4.5', 'claude-opus-4.1']
    },
    { name: 'OpenAI', slug: 'openai', total_models: 2, models: ['gpt-5', 'gpt-5-mini'] }
  ]
}

export const MODEL_INFO = { provider: DEFAULT_PROVIDER, model: DEFAULT_MODEL, effective_context_length: 200000 }

export const STATUS = {
  active_sessions: 1,
  config_path: '~/.hermes/config.yaml',
  config_version: 14,
  gateway_pid: 4242,
  gateway_running: true,
  gateway_state: 'running',
  gateway_updated_at: new Date().toISOString(),
  hermes_home: '~/.hermes',
  latest_config_version: 14,
  version: ''
}

const skill = (name: string, category: string, description: string, enabled = true) => ({
  name,
  category,
  description,
  enabled
})

export const SKILLS = [
  skill('frontend-design', 'engineering', 'Production-grade UI with high design quality.'),
  skill('mcp-builder', 'engineering', 'Build Model Context Protocol servers for any API.'),
  skill('pdf', 'documents', 'Extract, fill, merge and generate PDF documents.'),
  skill('canvas-design', 'creative', 'Design posters and visual art as PNG / PDF.'),
  skill('web-research', 'research', 'Deep web research with citations and synthesis.'),
  skill('internal-comms', 'writing', 'Status reports, FAQs and incident write-ups.', false)
]

const toolset = (name: string, label: string, description: string, tools: string[], enabled = true) => ({
  name,
  label,
  description,
  tools,
  enabled,
  configured: true
})

export const TOOLSETS = [
  toolset('files', 'Files', 'Read, write and search the workspace.', [
    'read_file',
    'write_file',
    'edit',
    'glob',
    'grep'
  ]),
  toolset('shell', 'Shell', 'Run commands in a sandboxed terminal.', ['run_command', 'kill', 'background']),
  toolset('web', 'Web', 'Search the web and fetch URLs.', ['web_search', 'fetch']),
  toolset('image', 'Image', 'Generate and edit images.', ['generate_image', 'edit_image'], false)
]

const profile = (name: string, provider: string, model: string, skill_count: number, is_default = false) => ({
  name,
  provider,
  model,
  skill_count,
  is_default,
  has_env: true,
  path: `~/.hermes/profiles/${name}`
})

export const PROFILES = [
  profile('default', DEFAULT_PROVIDER, DEFAULT_MODEL, 18, true),
  profile('research', 'openai', 'gpt-5', 9),
  profile('ops', 'anthropic', 'claude-opus-4.1', 24)
]

const dir = (name: string) => ({ name, isDirectory: true })
const file = (name: string) => ({ name, isDirectory: false })
export const FS_TREE: Record<string, Array<{ name: string; isDirectory: boolean }>> = {
  '~/code/hermes-agent': [
    dir('agent'),
    dir('apps'),
    dir('gateway'),
    dir('tools'),
    file('AGENTS.md'),
    file('README.md'),
    file('pyproject.toml')
  ],
  '~/code/hermes-agent/agent': [file('loop.py'), file('tools.py'), file('memory.py')],
  '~/code/hermes-agent/apps': [dir('desktop'), dir('shared')]
}
