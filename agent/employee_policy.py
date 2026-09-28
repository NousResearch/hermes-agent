"""Fixed employee surface; operational preferences remain native configuration."""
TOOLS = frozenset({
    'web_search', 'web_extract', 'terminal', 'process_manage',
    'read_file', 'write_file', 'patch', 'search_files',
    'vision_analyze', 'image_generate', 'video_analyze', 'browser_exec',
    'todo_list', 'memory', 'recall', 'session_search', 'execute_code',
    'delegate_task', 'send_message',
})


def select_tools(requested):
    return {name for name in requested if name in TOOLS or name.startswith('mcp_')}

# Keep upstream implementations available for merges; these entry points are not
# part of the employee product, regardless of a profile's old configuration.
SKILLS_ENABLED = False
EXCLUDED_COMMANDS = frozenset({'skills', 'reload-skills', 'learn', 'bundles', 'curator', 'kanban', 'cron', 'blueprint', 'suggestions', 'personality'})
KANBAN_ENABLED = False
MEMORY_PROVIDER = "hindsight"
LEGACY_PERSONALITY_ENABLED = False
NATIVE_CRON_AUTHORING_ENABLED = False
CRON_AUTHORING_COMMANDS = frozenset({'create', 'add', 'edit', 'remove', 'rm', 'delete'})
