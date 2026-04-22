#!/usr/bin/env python3
"""Add format_zh() calls to cli.py for user-facing strings."""

def main():
    cli_path = "/home/gzsiang/hermes-agent/cli.py"
    
    with open(cli_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # List of (old_string, new_string) replacements
    # Using format_zh() to wrap user-facing strings
    replacements = [
        # Compression feedback
        ('print(f"🗜️  Compressing {original_count} messages',
         'print(f"🗜️  {format_zh(\'Compressing {original_count} messages\', original_count=original_count)}'),
        
        ('print("  📊 Session Token Usage")',
         'print("  📊 " + format_zh("Session Token Usage"))'),
        
        # MCP reload
        ('print("🔄 Reloading MCP servers...")',
         'print("🔄 " + format_zh("Reloading MCP servers..."))'),
        
        # Snapshot
        ('print("  No state snapshots yet.")',
         'print("  " + format_zh("No state snapshots yet."))'),
        
        # Voice mode
        ('_cprint(f"\\n{_ACCENT}Voice mode enabled{tts_status}{_RST}")',
         '_cprint(f"\\n{_ACCENT}" + format_zh("Voice mode enabled") + tts_status + "{_RST}")'),
        
        ('_cprint(f"\\n{_DIM}Voice mode disabled.{_RST}")',
         '_cprint(f"\\n{_DIM}" + format_zh("Voice mode disabled.") + "{_RST}")'),
        
        ('_cprint(f"{_DIM}Voice mode is already enabled.{_RST}")',
         '_cprint(f"{_DIM}" + format_zh("Voice mode is already enabled.") + "{_RST}")'),
        
        ('_cprint(f"\\n{_ACCENT}Voice mode unavailable in this environment:{_RST}")',
         '_cprint(f"\\n{_ACCENT}" + format_zh("Voice mode unavailable in this environment:") + "{_RST}")'),
        
        ('_cprint(f"\\n{_ACCENT}Voice mode requirements not met:{_RST}")',
         '_cprint(f"\\n{_ACCENT}" + format_zh("Voice mode requirements not met:") + "{_RST}")'),
        
        ('_cprint(f"{_ACCENT}Voice TTS {status}.{_RST}")',
         '_cprint(f"{_ACCENT}" + format_zh("Voice TTS {status}.", status=status) + "{_RST}")'),
        
        ('_cprint(f"\\n{_BOLD}Voice Mode Status{_RST}")',
         '_cprint(f"\\n{_BOLD}" + format_zh("Voice Mode Status") + "{_RST}")'),
        
        ('_cprint(f"  Mode:      {\'ON\' if self._voice_mode else \'OFF\'}")',
         '_cprint(f"  " + format_zh("Mode:") + f"      {\'ON\' if self._voice_mode else \'OFF\'}")'),
        
        ('_cprint(f"  TTS:       {\'ON\' if self._voice_tts else \'OFF\'}")',
         '_cprint(f"  " + format_zh("TTS:") + f"       {\'ON\' if self._voice_tts else \'OFF\'}")'),
        
        ('_cprint(f"  Recording: {\'YES\' if self._voice_recording else \'no\'}")',
         '_cprint(f"  " + format_zh("Recording:") + f" {\'YES\' if self._voice_recording else \'no\'}")'),
        
        ('_cprint(f"  Record key: {_display_key}")',
         '_cprint(f"  " + format_zh("Record key:") + f" {_display_key}")'),
        
        ('_cprint(f"\\n  {_BOLD}Requirements:{_RST}")',
         '_cprint(f"\\n  {_BOLD}" + format_zh("Requirements:") + "{_RST}")'),
        
        # Clarify timeout
        ('_cprint(f"\\n{_DIM}(clarify timed out after {timeout}s — agent will decide){_RST}")',
         '_cprint(f"\\n{_DIM}" + format_zh("(clarify timed out after {timeout}s — agent will decide)", timeout=timeout) + "{_RST}")'),
        
        # Approval timeout
        ('_cprint(f"\\n{_DIM}  ⏱ Timeout — denying command{_RST}")',
         '_cprint(f"\\n{_DIM}  ⏱ " + format_zh("Timeout — denying command") + "{_RST}")'),
        
        # Approval choices
        ('choices = ["once", "session", "always", "deny"] if allow_permanent else ["once", "session", "deny"]',
         'choices = [format_zh("once"), format_zh("session"), format_zh("always"), format_zh("deny")] if allow_permanent else [format_zh("once"), format_zh("session"), format_zh("deny")]'),
        
        ('"once": "Allow once",',
         '"once": format_zh("Allow once"),'),
    ]
    
    changed = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new, 1)
            print(f"Replaced: {old[:60]}...")
            changed += 1
        else:
            print(f"NOT FOUND: {old[:60]}...")
    
    with open(cli_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\nTotal replacements: {changed}")

if __name__ == "__main__":
    main()
