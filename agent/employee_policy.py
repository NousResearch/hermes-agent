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
