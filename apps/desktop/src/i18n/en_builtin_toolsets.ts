// Display copy keyed by the built-in IDs from hermes_cli/tools_config.py.
export const enBuiltinToolsets = {
  web: { label: 'Web Search & Scraping', description: 'web_search, web_extract' },
  browser: { label: 'Browser Automation', description: 'navigate, click, type, scroll' },
  terminal: { label: 'Terminal & Processes', description: 'terminal, process' },
  file: { label: 'File Operations', description: 'read, write, patch, search' },
  code_execution: { label: 'Code Execution', description: 'execute_code' },
  vision: { label: 'Vision / Image Analysis', description: 'vision_analyze' },
  video: { label: 'Video Analysis', description: 'video_analyze (requires video-capable model)' },
  image_gen: { label: 'Image Generation', description: 'image_generate' },
  video_gen: { label: 'Video Generation', description: 'video_generate (text/image/reference)' },
  x_search: { label: 'X (Twitter) Search', description: 'x_search (requires xAI OAuth or XAI_API_KEY)' },
  tts: { label: 'Text-to-Speech', description: 'text_to_speech' },
  stt: { label: 'Speech-to-Text', description: 'voice transcription (gateway voice messages + voice mode)' },
  skills: { label: 'Skills', description: 'list, view, manage' },
  todo: { label: 'Task Planning', description: 'todo_list' },
  kanban: { label: 'Kanban', description: 'opt-in task board tools for this platform' },
  memory: { label: 'Memory', description: 'persistent memory across sessions' },
  context_engine: { label: 'Context Engine', description: 'runtime tools from the active context engine' },
  session_search: { label: 'Session Search', description: 'search past conversations' },
  connections: { label: 'Connections', description: 'remote connector tools and account authorization' },
  clarify: { label: 'Clarifying Questions', description: 'clarify' },
  delegation: { label: 'Task Delegation', description: 'delegate_task' },
  cronjob: { label: 'Cron Jobs', description: 'create/list/update/pause/resume/run, with optional attached skills' },
  discord: { label: 'Discord (read/participate)', description: 'fetch messages, search members, create thread' },
  discord_admin: { label: 'Discord Server Admin', description: 'list channels/roles, pin, assign roles' },
  computer_use: {
    label: 'Computer Use (macOS/Windows/Linux)',
    description: 'background desktop control via cua-driver'
  }
}
