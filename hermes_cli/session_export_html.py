"""
HTML Export generator for Hermes sessions.
Generates a standalone, beautiful HTML file with all messages embedded.
Supports single and multi-session exports with a professional sidebar.
No remote dependencies.
Enhanced with UI-UX-PRO-MAX design intelligence.
"""

import json
import datetime
from typing import Any, Dict, List

# --- Icons (Lucide-style SVGs) ---
ICON_USER = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-user"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
ICON_BOT = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-bot"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>'
ICON_TERMINAL = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-terminal"><polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/></svg>'
ICON_WRENCH = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-wrench"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>'
ICON_SPARKLES = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-sparkles"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>'
ICON_CHEVRON_RIGHT = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-chevron-right"><path d="m9 18 6-6-6-6"/></svg>'
ICON_SEARCH = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-search"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>'
ICON_HERMES = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFD700" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>'

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #F8FAFC;
            --text-color: #0F172A;
            --secondary-text: #475569;
            --user-bg: #FFFFFF;
            --assistant-bg: #F1F5F9;
            --border-color: #E2E8F0;
            --accent-color: #CD7F32;
            --accent-foreground: #FFFFFF;
            --code-bg: #1E293B;
            --code-text: #F8FAFC;
            --reasoning-bg: #FFFBEB;
            --reasoning-border: #FEF3C7;
            --tool-bg: #F0F9FF;
            --tool-border: #E0F2FE;
            --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
            --sidebar-bg: #FFFFFF;
            --sidebar-width: 320px;
        }}

        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: #000101;
                --text-color: #FFF8DC;
                --secondary-text: #94A3B8;
                --user-bg: #041c1c;
                --assistant-bg: #0c1a1a;
                --border-color: #CD7F32;
                --accent-color: #FFD700;
                --code-bg: #000000;
                --reasoning-bg: #1a1a1a;
                --reasoning-border: #CD7F32;
                --tool-bg: #0c4a6e;
                --tool-border: #075985;
                --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.3), 0 2px 4px -2px rgb(0 0 0 / 0.3);
                --sidebar-bg: #041c1c;
            }}
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--bg-color);
            -webkit-font-smoothing: antialiased;
            overflow-x: hidden;
        }}

        .layout {{
            display: flex;
            min-height: 100vh;
        }}

        /* Sidebar */
        .sidebar {{
            width: var(--sidebar-width);
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            position: fixed;
            height: 100vh;
            z-index: 100;
            transition: transform 0.3s ease;
        }}

        @media (max-width: 768px) {{
            .sidebar {{
                transform: translateX(-100%);
            }}
            .sidebar.open {{
                transform: translateX(0);
            }}
            .main-content {{
                margin-left: 0 !important;
            }}
        }}

        .sidebar-header {{
            padding: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }}

        .sidebar-brand {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-weight: 700;
            font-size: 1.25rem;
            color: var(--accent-color);
            margin-bottom: 1rem;
        }}

        .search-container {{
            position: relative;
        }}

        .search-container input {{
            width: 100%;
            padding: 0.5rem 1rem 0.5rem 2.25rem;
            border-radius: 0.5rem;
            border: 1px solid var(--border-color);
            background-color: var(--bg-color);
            color: var(--text-color);
            font-size: 0.875rem;
            outline: none;
        }}

        .search-container svg {{
            position: absolute;
            left: 0.75rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--secondary-text);
        }}

        .session-list {{
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
        }}

        .session-item {{
            display: block;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
            color: inherit;
            border: 1px solid transparent;
        }}

        .session-item:hover {{
            background-color: rgba(0, 0, 0, 0.05);
        }}

        @media (prefers-color-scheme: dark) {{
            .session-item:hover {{
                background-color: rgba(255, 255, 255, 0.05);
            }}
        }}

        .session-item.active {{
            background-color: var(--user-bg);
            border-color: var(--accent-color);
            box-shadow: var(--shadow);
        }}

        .session-item-title {{
            font-weight: 600;
            font-size: 0.875rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 0.25rem;
        }}

        .session-item-meta {{
            font-size: 0.75rem;
            color: var(--secondary-text);
            display: flex;
            justify-content: space-between;
        }}

        /* Main Content */
        .main-content {{
            flex: 1;
            margin-left: {main_margin};
            padding: 3rem 2rem;
            max-width: 100%;
            transition: margin-left 0.3s ease;
        }}

        .session-view {{
            display: none;
            width: 100%;
            margin: 0 auto;
        }}

        .layout-single .session-view {{
            max-width: 90%;
            margin: 0 auto;
        }}

        .layout-multi .session-view {{
            max-width: 100%;
            margin: 0;
        }}

        .layout-multi .main-content {{
            width: 0;
        }}
        .session-view.active {{
            display: block;
        }}

        header {{
            margin-bottom: 3rem;
            text-align: center;
        }}

        .meta {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 1.5rem;
            margin-top: 1rem;
            font-size: 0.875rem;
            color: var(--secondary-text);
         }}

        .meta-item strong {{
            color: var(--text-color);
        }}

        /* Messages */
        .message {{
            margin-bottom: 1.5rem;
            border-radius: 0.75rem;
            background-color: var(--user-bg);
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow);
            overflow: hidden;
        }}

        .message-user {{
            background-color: #f0fdf4;
        }}

        .message-assistant {{
            background-color: var(--assistant-bg);
            border-left: 4px solid var(--accent-color);
        }}

        .message-tool {{
            background-color: #f8fafc;
            border-style: dotted;
        }}

        @media (prefers-color-scheme: dark) {{
            .message-user {{
                background-color: #0c2121;
            }}

            .message-assistant {{
                background-color: #041c1c;
            }}

            .message-tool {{
                background-color: #0f172a;
            }}
        }}

        .message-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.5rem;
            cursor: pointer;
            user-select: none;
            transition: background-color 0.2s ease;
        }}

        .message-header:hover {{
            background-color: rgba(0,0,0,0.02);
        }}

        .message-header svg.chevron {{
            transition: transform 0.2s ease;
            color: var(--secondary-text);
        }}

        .message.active svg.chevron {{
            transform: rotate(90deg);
        }}

        .role-badge {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--secondary-text);
        }}

        .role-badge svg {{
            color: var(--accent-color);
        }}

        .timestamp {{
            font-size: 0.75rem;
            color: var(--secondary-text);
            font-variant-numeric: tabular-nums;
        }}

        .message-body {{
            display: none;
            padding: 0 1.5rem 1.5rem 1.5rem;
            border-top: 1px solid var(--border-color);
            padding-top: 1.5rem;
        }}

        .message.active .message-body {{
            display: block;
        }}

        .content {{
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 1rem;
        }}

        /* Code Blocks */
        code {{
            font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
            background-color: rgba(0,0,0,0.05);
            padding: 0.2rem 0.4rem;
            border-radius: 0.375rem;
            font-size: 0.9em;
        }}

        pre {{
            background-color: var(--code-bg);
            color: var(--code-text);
            padding: 1.25rem;
            border-radius: 0.75rem;
            overflow-x: auto;
            margin: 1.25rem 0;
            font-size: 0.9rem;
            line-height: 1.5;
            box-shadow: inset 0 2px 4px 0 rgb(0 0 0 / 0.1);
        }}

        pre code {{
            background-color: transparent;
            padding: 0;
            border-radius: 0;
            color: inherit;
        }}

        /* Tool Calls */
        .tool-call {{
            margin: 1rem 0;
            border-radius: 0.5rem;
            overflow: hidden;
            border: 1px solid var(--tool-border);
            background-color: var(--tool-bg);
        }}

        .tool-call-header {{
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
            user-select: none;
            font-size: 0.875rem;
            font-weight: 600;
        }}

        .tool-call-header:hover {{
            background-color: rgba(0,0,0,0.03);
        }}

        .tool-call-header svg.chevron {{
            transition: transform 0.2s ease;
            color: var(--secondary-text);
        }}

        .tool-call.active svg.chevron {{
            transform: rotate(90deg);
        }}

        .tool-call-content {{
            display: none;
            padding: 0 1rem 1rem 1rem;
        }}

        .tool-call.active .tool-call-content {{
            display: block;
        }}

        /* Reasoning */
        .reasoning {{
            margin-top: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid var(--reasoning-border);
            background-color: var(--reasoning-bg);
            overflow: hidden;
        }}

        .reasoning-header {{
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
            user-select: none;
            font-size: 0.875rem;
            font-weight: 600;
        }}

        .reasoning-header:hover {{
            background-color: rgba(0,0,0,0.02);
        }}

        .reasoning-header svg.chevron {{
            transition: transform 0.2s ease;
            color: var(--secondary-text);
        }}

        .reasoning.active svg.chevron {{
            transform: rotate(90deg);
        }}

        .reasoning-content {{
            display: none;
            padding: 0 1rem 1rem 1rem;
            font-size: 0.925rem;
            color: var(--secondary-text);
            border-top: 1px solid var(--reasoning-border);
            padding-top: 1rem;
        }}

        .reasoning.active .reasoning-content {{
            display: block;
        }}

        footer {{
            margin-top: 5rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border-color);
            text-align: center;
            font-size: 0.875rem;
            color: var(--secondary-text);
        }}

        /* Animation */
        .fade-in {{
            animation: fadeIn 0.4s ease-out backwards;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Utilities */
        .hidden {{ display: none !important; }}
    </style>
</head>
<body>
    <div class="layout {layout_class}">
        {sidebar_html}
        
        <div class="main-content">
            {sessions_html}
            
            <footer>
                Built with ☤ Hermes Agent • Generated on {generated_at}
            </footer>
        </div>
    </div>

    <script>
        // Session Switching
        function showSession(id) {{
            // Update Sidebar
            document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
            const activeItem = document.querySelector(`.session-item[data-id="${{id}}"]`);
            if (activeItem) activeItem.classList.add('active');

            // Update View
            document.querySelectorAll('.session-view').forEach(el => {{
                el.classList.remove('active');
            }});
            const activeView = document.getElementById(`view-${{id}}`);
            if (activeView) activeView.classList.add('active');

            // Store in URL hash
            window.location.hash = id;
        }}

        // Search Filter
        const searchInput = document.getElementById('session-search');
        if (searchInput) {{
            searchInput.addEventListener('input', (e) => {{
                const term = e.target.value.toLowerCase();
                document.querySelectorAll('.session-item').forEach(item => {{
                    const text = item.innerText.toLowerCase();
                    if (text.includes(term)) {{
                        item.classList.remove('hidden');
                    }} else {{
                        item.classList.add('hidden');
                    }}
                }});
            }});
        }}

        // Card Toggles
        document.addEventListener('click', function(e) {{
            const header = e.target.closest('.message-header, .tool-call-header, .reasoning-header');
            if (header) {{
                header.parentElement.classList.toggle('active');
            }}
        }});

        // Initialization
        window.addEventListener('load', () => {{
            const hash = window.location.hash.slice(1);
            if (hash) {{
                showSession(hash);
            }} else {{
                // Show first session by default if none selected
                const first = document.querySelector('.session-item');
                if (first) showSession(first.getAttribute('data-id'));
            }}
        }});

        // Intersection Observer for scroll animations
        if ('IntersectionObserver' in window) {{
            var observer = new IntersectionObserver(function(entries) {{
                entries.forEach(function(entry) {{
                    if (entry.isIntersecting) {{
                        entry.target.classList.add('fade-in');
                        observer.unobserve(entry.target);
                    }}
                }});
            }}, {{ threshold: 0.1 }});

            document.querySelectorAll('.message').forEach(function(m) {{ observer.observe(m); }});
        }}
    </script>
</body>
</html>
"""

def _escape_html(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

def _format_timestamp(ts: float) -> str:
    if not ts: return "N/A"
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def _generate_messages_html(messages: List[Dict[str, Any]]) -> str:
    html_list = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""
        timestamp = _format_timestamp(msg.get("timestamp", 0))
        
        # Icon selection
        role_icon = ICON_TERMINAL
        if role == "user":
            role_icon = ICON_USER
        elif role == "assistant":
            role_icon = ICON_BOT

        # Handle multimodal or complex content
        if isinstance(content, list):
            content_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        content_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        content_parts.append("[Image Attachment]")
                else:
                    content_parts.append(str(part))
            content = "\n".join(content_parts)

        # Build message HTML
        msg_class = f"message message-{role} active"
        # Delay animation for initial items
        delay_style = f' style="animation-delay: {min(i * 0.05, 1.0)}s"' if i < 10 else ""
        
        chevron_html = ICON_CHEVRON_RIGHT.replace('class="', 'class="chevron ')
        
        html = f'<div class="{msg_class}"{delay_style}>'
        html += f'  <div class="message-header">'
        html += f'    <div class="role-badge">{chevron_html} {role_icon} {role}</div>'
        html += f'    <div class="timestamp">{timestamp}</div>'
        html += '  </div>'
        html += '  <div class="message-body">'
        
        # Tool Calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "unknown")
                args = tc.get("function", {}).get("arguments", "{}")
                html += f'''
                <div class="tool-call">
                    <div class="tool-call-header">
                        {ICON_CHEVRON_RIGHT.replace('class="', 'class="chevron ')}
                        {ICON_WRENCH} Tool Call: {fn_name}
                    </div>
                    <div class="tool-call-content">
                        <pre><code>{_escape_html(args)}</code></pre>
                    </div>
                </div>
                '''

        # Content
        if content:
            if role == "tool":
                html += f'  <div class="content"><pre><code>{_escape_html(content)}</code></pre></div>'
            else:
                html += f'  <div class="content">{_escape_html(content)}</div>'
        
        # Reasoning
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        if reasoning:
            html += f'''
            <div class="reasoning">
                <div class="reasoning-header">
                    {ICON_CHEVRON_RIGHT.replace('class="', 'class="chevron ')}
                    {ICON_SPARKLES} Reasoning
                </div>
                <div class="reasoning-content">
                    <div class="content">{_escape_html(reasoning)}</div>
                </div>
            </div>
            '''
            
        html += '  </div>'
        html += '</div>'
        html_list.append(html)
    return "\n".join(html_list)

def generate_multi_session_html_export(sessions: List[Dict[str, Any]]) -> str:
    if not sessions:
        return "<html><body><h1>No sessions to export.</h1></body></html>"

    is_multi = len(sessions) > 1
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Sidebar
    sidebar_html = ""
    if is_multi:
        sidebar_items = []
        for s in sessions:
            sid = s.get("id", "N/A")
            title = s.get("title") or s.get("preview") or "Untitled Session"
            if len(title) > 50: title = title[:47] + "..."
            date = _format_timestamp(s.get("started_at", 0)).split(" ")[0]
            
            item = f'''
            <a class="session-item" data-id="{sid}" onclick="showSession('{sid}')">
                <div class="session-item-title">{_escape_html(title)}</div>
                <div class="session-item-meta">
                    <span>{sid[:8]}</span>
                    <span>{date}</span>
                </div>
            </a>
            '''
            sidebar_items.append(item)
        
        sidebar_html = f'''
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-brand">
                    {ICON_HERMES} Hermes History
                </div>
                <div class="search-container">
                    {ICON_SEARCH}
                    <input type="text" id="session-search" placeholder="Search sessions...">
                </div>
            </div>
            <div class="session-list">
                {"".join(sidebar_items)}
            </div>
        </aside>
        '''

    # Main Content
    sessions_html_list = []
    for s in sessions:
        sid = s.get("id", "N/A")
        title = s.get("title") or "Hermes Session"
        model = s.get("model", "Unknown")
        started_at = _format_timestamp(s.get("started_at", 0))
        messages = s.get("messages", [])
        messages_html = _generate_messages_html(messages)
        
        view_class = "session-view"
        if not is_multi: view_class += " active"
        
        session_html = f'''
        <div class="{view_class}" id="view-{sid}">
            <header class="fade-in">
                <h1>{_escape_html(title)}</h1>
                <div class="meta">
                    <div class="meta-item"><strong>ID:</strong> {sid}</div>
                    <div class="meta-item"><strong>Model:</strong> {_escape_html(model)}</div>
                    <div class="meta-item"><strong>Started:</strong> {started_at}</div>
                </div>
            </header>
            <main>
                {messages_html}
            </main>
        </div>
        '''
        sessions_html_list.append(session_html)

    return HTML_TEMPLATE.format(
        page_title="Hermes Session Export" if is_multi else _escape_html(sessions[0].get("title", "Hermes Session")),
        sidebar_html=sidebar_html,
        sessions_html="\n".join(sessions_html_list),
        main_margin="var(--sidebar-width)" if is_multi else "0",
        layout_class="layout-multi" if is_multi else "layout-single",
        generated_at=generated_at
    )

def generate_html_export(session_data: Dict[str, Any]) -> str:
    """Legacy wrapper for single session export."""
    return generate_multi_session_html_export([session_data])
