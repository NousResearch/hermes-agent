#!/usr/bin/env python3
"""Simple i18n patch for cli.py - string replacement on full file content."""

def main():
    cli_path = "/home/gzsiang/hermes-agent/cli.py"
    
    with open(cli_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Only replace exact user-facing strings
    REPLACEMENTS = [
        # Compression
        ('🗜️  Compressing ', '🗜️ 正在压缩 '),
        ('Compressed: ', '已压缩: '),
        ('Rough transcript estimate:', '粗略转录估算:'),
        
        # Session token usage
        ('📊 Session Token Usage', '📊 会话 Token 使用'),
        
        # MCP reload
        ('🔄 Reloading MCP servers...', '🔄 正在重新加载 MCP 服务器...'),
        
        # Snapshot
        ('No state snapshots yet.', '暂无状态快照。'),
        
        # Voice mode
        ('Voice mode enabled', '语音模式已启用'),
        ('Voice mode disabled.', '语音模式已禁用。'),
        ('Voice mode is already enabled.', '语音模式已启用。'),
        ('Voice mode unavailable in this environment:', '语音模式在此环境中不可用：'),
        ('Voice mode requirements not met:', '语音模式要求未满足：'),
        ('Voice TTS ', '语音 TTS '),
        ('Voice Mode Status', '语音模式状态'),
        ('Mode:      ', '模式:      '),
        ('TTS:       ', 'TTS:       '),
        ('Recording: ', '录音: '),
        ('Record key: ', '录音键: '),
        ('Requirements:', '要求：'),
        
        # Clarify timeout
        ('(clarify timed out after ', '(澄清超时（'),
        (' — agent will decide)', '）— 代理将自行决定)'),
        
        # Approval timeout
        ('⏱ Timeout — denying command', '⏱ 超时 — 拒绝命令'),
        
        # Approval choices - use full context to avoid replacing code logic
        ('"once", "session", "always", "deny"] if allow_permanent else ["once", "session", "deny"]',
         '"仅一次", "本次会话", "始终", "拒绝"] if allow_permanent else ["仅一次", "本次会话", "拒绝"'),
        ('"once": "Allow once",', '"仅一次": "允许仅一次",'),
    ]
    
    changed = 0
    for old, new in REPLACEMENTS:
        count = content.count(old)
        if count > 0:
            content = content.replace(old, new)
            print(f"Replaced ({count}x): {old!r} -> {new!r}")
            changed += count
    
    with open(cli_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\nTotal replacements: {changed}")

if __name__ == "__main__":
    main()
