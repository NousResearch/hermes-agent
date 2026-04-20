import os, sys, time, threading, uuid
from datetime import datetime
from pathlib import Path
from rich.panel import Panel
from rich import box as rich_box

class CLICommandHandlersMixin:
        def _handle_rollback_command(self, command: str):
            """Handle /rollback — list, diff, or restore filesystem checkpoints.
    
            Syntax:
                /rollback                 — list checkpoints
                /rollback <N>             — restore checkpoint N (also undoes last chat turn)
                /rollback diff <N>        — preview changes since checkpoint N
                /rollback <N> <file>      — restore a single file from checkpoint N
            """
            from tools.checkpoint_manager import format_checkpoint_list
    
            if not hasattr(self, 'agent') or not self.agent:
                print("  No active agent session.")
                return
    
            mgr = self.agent._checkpoint_mgr
            if not mgr.enabled:
                print("  Checkpoints are not enabled.")
                print("  Enable with: hermes --checkpoints")
                print("  Or in config.yaml: checkpoints: { enabled: true }")
                return
    
            cwd = os.getenv("TERMINAL_CWD", os.getcwd())
            parts = command.split()
            args = parts[1:] if len(parts) > 1 else []
    
            if not args:
                # List checkpoints
                checkpoints = mgr.list_checkpoints(cwd)
                print(format_checkpoint_list(checkpoints, cwd))
                return
    
            # Handle /rollback diff <N>
            if args[0].lower() == "diff":
                if len(args) < 2:
                    print("  Usage: /rollback diff <N>")
                    return
                checkpoints = mgr.list_checkpoints(cwd)
                if not checkpoints:
                    print(f"  No checkpoints found for {cwd}")
                    return
                target_hash = self._resolve_checkpoint_ref(args[1], checkpoints)
                if not target_hash:
                    return
                result = mgr.diff(cwd, target_hash)
                if result["success"]:
                    stat = result.get("stat", "")
                    diff = result.get("diff", "")
                    if not stat and not diff:
                        print("  No changes since this checkpoint.")
                    else:
                        if stat:
                            print(f"\n{stat}")
                        if diff:
                            # Limit diff output to avoid terminal flood
                            diff_lines = diff.splitlines()
                            if len(diff_lines) > 80:
                                print("\n".join(diff_lines[:80]))
                                print(f"\n  ... ({len(diff_lines) - 80} more lines, showing first 80)")
                            else:
                                print(f"\n{diff}")
                else:
                    print(f"  ❌ {result['error']}")
                return
    
            # Resolve checkpoint reference (number or hash)
            checkpoints = mgr.list_checkpoints(cwd)
            if not checkpoints:
                print(f"  No checkpoints found for {cwd}")
                return
    
            target_hash = self._resolve_checkpoint_ref(args[0], checkpoints)
            if not target_hash:
                return
    
            # Check for file-level restore: /rollback <N> <file>
            file_path = args[1] if len(args) > 1 else None
    
            result = mgr.restore(cwd, target_hash, file_path=file_path)
            if result["success"]:
                if file_path:
                    print(f"  ✅ Restored {file_path} from checkpoint {result['restored_to']}: {result['reason']}")
                else:
                    print(f"  ✅ Restored to checkpoint {result['restored_to']}: {result['reason']}")
                print("  A pre-rollback snapshot was saved automatically.")
    
                # Also undo the last conversation turn so the agent's context
                # matches the restored filesystem state
                if self.conversation_history:
                    self.undo_last()
                    print("  Chat turn undone to match restored file state.")
            else:
                print(f"  ❌ {result['error']}")

        def _handle_stop_command(self):
            """Handle /stop — kill all running background processes.
    
            Inspired by OpenAI Codex's separation of interrupt (stop current turn)
            from /stop (clean up background processes). See openai/codex#14602.
            """
            from tools.process_registry import process_registry
    
            processes = process_registry.list_sessions()
            running = [p for p in processes if p.get("status") == "running"]
    
            if not running:
                print("  No running background processes.")
                return
    
            print(f"  Stopping {len(running)} background process(es)...")
            killed = process_registry.kill_all()
            print(f"  ✅ Stopped {killed} process(es).")

        def _handle_paste_command(self):
            """Handle /paste — explicitly check clipboard for an image.
    
            This is the reliable fallback for terminals where BracketedPaste
            doesn't fire for image-only clipboard content (e.g., VSCode terminal,
            Windows Terminal with WSL2).
            """
            from hermes_cli.clipboard import has_clipboard_image
            if has_clipboard_image():
                if self._try_attach_clipboard_image():
                    n = len(self._attached_images)
                    _cprint(f"  📎 Image #{n} attached from clipboard")
                else:
                    _cprint(f"  {_DIM}(>_<) Clipboard has an image but extraction failed{_RST}")
            else:
                _cprint(f"  {_DIM}(._.) No image found in clipboard{_RST}")

        def _handle_tools_command(self, cmd: str):
            """Handle /tools [list|disable|enable] slash commands.
    
            /tools (no args) shows the tool list.
            /tools list shows enabled/disabled status per toolset.
            /tools disable/enable saves the change to config and resets
            the session so the new tool set takes effect cleanly (no
            prompt-cache breakage mid-conversation).
            """
            import shlex
            from argparse import Namespace
            from hermes_cli.tools_config import tools_disable_enable_command
    
            try:
                parts = shlex.split(cmd)
            except ValueError:
                parts = cmd.split()
    
            subcommand = parts[1] if len(parts) > 1 else ""
            if subcommand not in ("list", "disable", "enable"):
                self.show_tools()
                return
    
            if subcommand == "list":
                tools_disable_enable_command(
                    Namespace(tools_action="list", platform="cli"))
                return
    
            names = parts[2:]
            if not names:
                print(f"(._.) Usage: /tools {subcommand} <name> [name ...]")
                print(f"  Built-in toolset:  /tools {subcommand} web")
                print(f"  MCP tool:          /tools {subcommand} github:create_issue")
                return
    
            # Apply the change directly — the user typing the command is implicit
            # consent.  Do NOT use input() here; it hangs inside prompt_toolkit's
            # TUI event loop (known pitfall).
            verb = "Disabling" if subcommand == "disable" else "Enabling"
            label = ", ".join(names)
            _cprint(f"{_GOLD}{verb} {label}...{_RST}")
    
            tools_disable_enable_command(
                Namespace(tools_action=subcommand, names=names, platform="cli"))
    
            # Reset session so the new tool config is picked up from a clean state
            from hermes_cli.tools_config import _get_platform_tools
            from hermes_cli.config import load_config
            self.enabled_toolsets = _get_platform_tools(load_config(), "cli")
            self.new_session()
            _cprint(f"{_DIM}Session reset. New tool configuration is active.{_RST}")

        def _handle_profile_command(self):
            """Display active profile name and home directory."""
            from hermes_constants import get_hermes_home, display_hermes_home
    
            home = get_hermes_home()
            display = display_hermes_home()
    
            profiles_parent = Path.home() / ".hermes" / "profiles"
            try:
                rel = home.relative_to(profiles_parent)
                profile_name = str(rel).split("/")[0]
            except ValueError:
                profile_name = None
    
            print()
            if profile_name:
                print(f"  Profile: {profile_name}")
            else:
                print("  Profile: default")
            print(f"  Home:    {display}")
            print()

        def _handle_resume_command(self, cmd_original: str) -> None:
            """Handle /resume <session_id_or_title> — switch to a previous session mid-conversation."""
            parts = cmd_original.split(None, 1)
            target = parts[1].strip() if len(parts) > 1 else ""
    
            if not target:
                _cprint("  Usage: /resume <session_id_or_title>")
                _cprint("  Tip:   Use /history or `hermes sessions list` to find sessions.")
                return
    
            if not self._session_db:
                _cprint("  Session database not available.")
                return
    
            # Resolve title or ID
            from hermes_cli.main import _resolve_session_by_name_or_id
            resolved = _resolve_session_by_name_or_id(target)
            target_id = resolved or target
    
            session_meta = self._session_db.get_session(target_id)
            if not session_meta:
                _cprint(f"  Session not found: {target}")
                _cprint("  Use /history or `hermes sessions list` to see available sessions.")
                return
    
            if target_id == self.session_id:
                _cprint("  Already on that session.")
                return
    
            # End current session
            try:
                self._session_db.end_session(self.session_id, "resumed_other")
            except Exception:
                pass
    
            # Switch to the target session
            self.session_id = target_id
            self._resumed = True
            self._pending_title = None
    
            # Load conversation history
            restored = self._session_db.get_messages_as_conversation(target_id)
            self.conversation_history = restored or []
    
            # Re-open the target session so it's not marked as ended
            try:
                self._session_db.reopen_session(target_id)
            except Exception:
                pass
    
            # Sync the agent if already initialised
            if self.agent:
                self.agent.prepare_for_session_switch(
                    target_id,
                    flushed_db_idx=len(self.conversation_history),
                )
    
            title_part = f" \"{session_meta['title']}\"" if session_meta.get("title") else ""
            msg_count = len([m for m in self.conversation_history if m.get("role") == "user"])
            if self.conversation_history:
                _cprint(
                    f"  ↻ Resumed session {target_id}{title_part}"
                    f" ({msg_count} user message{'s' if msg_count != 1 else ''},"
                    f" {len(self.conversation_history)} total)"
                )
            else:
                _cprint(f"  ↻ Resumed session {target_id}{title_part} — no messages, starting fresh.")

        def _handle_prompt_command(self, cmd: str):
            """Handle the /prompt command to view or set system prompt."""
            parts = cmd.split(maxsplit=1)
            
            if len(parts) > 1:
                # Set new prompt
                new_prompt = parts[1].strip()
                
                if new_prompt.lower() == "clear":
                    self.system_prompt = ""
                    self.agent = None  # Force re-init
                    if save_config_value("agent.system_prompt", ""):
                        print("(^_^)b System prompt cleared (saved to config)")
                    else:
                        print("(^_^) System prompt cleared (session only)")
                else:
                    self.system_prompt = new_prompt
                    self.agent = None  # Force re-init
                    if save_config_value("agent.system_prompt", new_prompt):
                        print("(^_^)b System prompt set (saved to config)")
                    else:
                        print("(^_^) System prompt set (session only)")
                    print(f"  \"{new_prompt[:60]}{'...' if len(new_prompt) > 60 else ''}\"")
            else:
                # Show current prompt
                print()
                print("+" + "-" * 50 + "+")
                print("|" + " " * 15 + "(^_^) System Prompt" + " " * 15 + "|")
                print("+" + "-" * 50 + "+")
                print()
                if self.system_prompt:
                    # Word wrap the prompt for display
                    words = self.system_prompt.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) + 1 <= 50:
                            current_line += (" " if current_line else "") + word
                        else:
                            lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    for line in lines:
                        print(f"  {line}")
                else:
                    print("  (no custom prompt set - using default)")
                print()
                print("  Usage:")
                print("    /prompt <text>  - Set a custom system prompt")
                print("    /prompt clear   - Remove custom prompt")
                print("    /personality    - Use a predefined personality")
                print()

        def _handle_personality_command(self, cmd: str):
            """Handle the /personality command to set predefined personalities."""
            parts = cmd.split(maxsplit=1)
            
            if len(parts) > 1:
                # Set personality
                personality_name = parts[1].strip().lower()
                
                if personality_name in ("none", "default", "neutral"):
                    self.system_prompt = ""
                    self.agent = None  # Force re-init
                    if save_config_value("agent.system_prompt", ""):
                        print("(^_^)b Personality cleared (saved to config)")
                    else:
                        print("(^_^) Personality cleared (session only)")
                    print("  No personality overlay — using base agent behavior.")
                elif personality_name in self.personalities:
                    self.system_prompt = self._resolve_personality_prompt(self.personalities[personality_name])
                    self.agent = None  # Force re-init
                    if save_config_value("agent.system_prompt", self.system_prompt):
                        print(f"(^_^)b Personality set to '{personality_name}' (saved to config)")
                    else:
                        print(f"(^_^) Personality set to '{personality_name}' (session only)")
                    print(f"  \"{self.system_prompt[:60]}{'...' if len(self.system_prompt) > 60 else ''}\"")
                else:
                    print(f"(._.) Unknown personality: {personality_name}")
                    print(f"  Available: none, {', '.join(self.personalities.keys())}")
            else:
                # Show available personalities
                print()
                print("+" + "-" * 50 + "+")
                print("|" + " " * 12 + "(^o^)/ Personalities" + " " * 15 + "|")
                print("+" + "-" * 50 + "+")
                print()
                print(f"  {'none':<12} - (no personality overlay)")
                for name, prompt in self.personalities.items():
                    if isinstance(prompt, dict):
                        preview = prompt.get("description") or prompt.get("system_prompt", "")[:50]
                    else:
                        preview = str(prompt)[:50]
                    print(f"  {name:<12} - {preview}")
                print()
                print("  Usage: /personality <name>")
                print()

        def _handle_cron_command(self, cmd: str):
            """Handle the /cron command to manage scheduled tasks."""
            import shlex
            from tools.cronjob_tools import cronjob as cronjob_tool
    
            def _cron_api(**kwargs):
                return json.loads(cronjob_tool(**kwargs))
    
            def _normalize_skills(values):
                normalized = []
                for value in values:
                    text = str(value or "").strip()
                    if text and text not in normalized:
                        normalized.append(text)
                return normalized
    
            def _parse_flags(tokens):
                opts = {
                    "name": None,
                    "deliver": None,
                    "repeat": None,
                    "skills": [],
                    "add_skills": [],
                    "remove_skills": [],
                    "clear_skills": False,
                    "all": False,
                    "prompt": None,
                    "schedule": None,
                    "positionals": [],
                }
                i = 0
                while i < len(tokens):
                    token = tokens[i]
                    if token == "--name" and i + 1 < len(tokens):
                        opts["name"] = tokens[i + 1]
                        i += 2
                    elif token == "--deliver" and i + 1 < len(tokens):
                        opts["deliver"] = tokens[i + 1]
                        i += 2
                    elif token == "--repeat" and i + 1 < len(tokens):
                        try:
                            opts["repeat"] = int(tokens[i + 1])
                        except ValueError:
                            print("(._.) --repeat must be an integer")
                            return None
                        i += 2
                    elif token == "--skill" and i + 1 < len(tokens):
                        opts["skills"].append(tokens[i + 1])
                        i += 2
                    elif token == "--add-skill" and i + 1 < len(tokens):
                        opts["add_skills"].append(tokens[i + 1])
                        i += 2
                    elif token == "--remove-skill" and i + 1 < len(tokens):
                        opts["remove_skills"].append(tokens[i + 1])
                        i += 2
                    elif token == "--clear-skills":
                        opts["clear_skills"] = True
                        i += 1
                    elif token == "--all":
                        opts["all"] = True
                        i += 1
                    elif token == "--prompt" and i + 1 < len(tokens):
                        opts["prompt"] = tokens[i + 1]
                        i += 2
                    elif token == "--schedule" and i + 1 < len(tokens):
                        opts["schedule"] = tokens[i + 1]
                        i += 2
                    else:
                        opts["positionals"].append(token)
                        i += 1
                return opts
    
            tokens = shlex.split(cmd)
    
            if len(tokens) == 1:
                print()
                print("+" + "-" * 68 + "+")
                print("|" + " " * 22 + "(^_^) Scheduled Tasks" + " " * 23 + "|")
                print("+" + "-" * 68 + "+")
                print()
                print("  Commands:")
                print("    /cron list")
                print('    /cron add "every 2h" "Check server status" [--skill blogwatcher]')
                print('    /cron edit <job_id> --schedule "every 4h" --prompt "New task"')
                print("    /cron edit <job_id> --skill blogwatcher --skill find-nearby")
                print("    /cron edit <job_id> --remove-skill blogwatcher")
                print("    /cron edit <job_id> --clear-skills")
                print("    /cron pause <job_id>")
                print("    /cron resume <job_id>")
                print("    /cron run <job_id>")
                print("    /cron remove <job_id>")
                print()
                result = _cron_api(action="list")
                jobs = result.get("jobs", []) if result.get("success") else []
                if jobs:
                    print("  Current Jobs:")
                    print("  " + "-" * 63)
                    for job in jobs:
                        repeat_str = job.get("repeat", "?")
                        print(f"    {job['job_id'][:12]:<12} | {job['schedule']:<15} | {repeat_str:<8}")
                        if job.get("skills"):
                            print(f"      Skills: {', '.join(job['skills'])}")
                        print(f"      {job.get('prompt_preview', '')}")
                        if job.get("next_run_at"):
                            print(f"      Next: {job['next_run_at']}")
                        print()
                else:
                    print("  No scheduled jobs. Use '/cron add' to create one.")
                print()
                return
    
            subcommand = tokens[1].lower()
            opts = _parse_flags(tokens[2:])
            if opts is None:
                return
    
            if subcommand == "list":
                result = _cron_api(action="list", include_disabled=opts["all"])
                jobs = result.get("jobs", []) if result.get("success") else []
                if not jobs:
                    print("(._.) No scheduled jobs.")
                    return
    
                print()
                print("Scheduled Jobs:")
                print("-" * 80)
                for job in jobs:
                    print(f"  ID: {job['job_id']}")
                    print(f"  Name: {job['name']}")
                    print(f"  State: {job.get('state', '?')}")
                    print(f"  Schedule: {job['schedule']} ({job.get('repeat', '?')})")
                    print(f"  Next run: {job.get('next_run_at', 'N/A')}")
                    if job.get("skills"):
                        print(f"  Skills: {', '.join(job['skills'])}")
                    print(f"  Prompt: {job.get('prompt_preview', '')}")
                    if job.get("last_run_at"):
                        print(f"  Last run: {job['last_run_at']} ({job.get('last_status', '?')})")
                    print()
                return
    
            if subcommand in {"add", "create"}:
                positionals = opts["positionals"]
                if not positionals:
                    print("(._.) Usage: /cron add <schedule> <prompt>")
                    return
                schedule = opts["schedule"] or positionals[0]
                prompt = opts["prompt"] or " ".join(positionals[1:])
                skills = _normalize_skills(opts["skills"])
                if not prompt and not skills:
                    print("(._.) Please provide a prompt or at least one skill")
                    return
                result = _cron_api(
                    action="create",
                    schedule=schedule,
                    prompt=prompt or None,
                    name=opts["name"],
                    deliver=opts["deliver"],
                    repeat=opts["repeat"],
                    skills=skills or None,
                )
                if result.get("success"):
                    print(f"(^_^)b Created job: {result['job_id']}")
                    print(f"  Schedule: {result['schedule']}")
                    if result.get("skills"):
                        print(f"  Skills: {', '.join(result['skills'])}")
                    print(f"  Next run: {result['next_run_at']}")
                else:
                    print(f"(x_x) Failed to create job: {result.get('error')}")
                return
    
            if subcommand == "edit":
                positionals = opts["positionals"]
                if not positionals:
                    print("(._.) Usage: /cron edit <job_id> [--schedule ...] [--prompt ...] [--skill ...]")
                    return
                job_id = positionals[0]
                existing = get_job(job_id)
                if not existing:
                    print(f"(._.) Job not found: {job_id}")
                    return
    
                final_skills = None
                replacement_skills = _normalize_skills(opts["skills"])
                add_skills = _normalize_skills(opts["add_skills"])
                remove_skills = set(_normalize_skills(opts["remove_skills"]))
                existing_skills = list(existing.get("skills") or ([] if not existing.get("skill") else [existing.get("skill")]))
                if opts["clear_skills"]:
                    final_skills = []
                elif replacement_skills:
                    final_skills = replacement_skills
                elif add_skills or remove_skills:
                    final_skills = [skill for skill in existing_skills if skill not in remove_skills]
                    for skill in add_skills:
                        if skill not in final_skills:
                            final_skills.append(skill)
    
                result = _cron_api(
                    action="update",
                    job_id=job_id,
                    schedule=opts["schedule"],
                    prompt=opts["prompt"],
                    name=opts["name"],
                    deliver=opts["deliver"],
                    repeat=opts["repeat"],
                    skills=final_skills,
                )
                if result.get("success"):
                    job = result["job"]
                    print(f"(^_^)b Updated job: {job['job_id']}")
                    print(f"  Schedule: {job['schedule']}")
                    if job.get("skills"):
                        print(f"  Skills: {', '.join(job['skills'])}")
                    else:
                        print("  Skills: none")
                else:
                    print(f"(x_x) Failed to update job: {result.get('error')}")
                return
    
            if subcommand in {"pause", "resume", "run", "remove", "rm", "delete"}:
                positionals = opts["positionals"]
                if not positionals:
                    print(f"(._.) Usage: /cron {subcommand} <job_id>")
                    return
                job_id = positionals[0]
                action = "remove" if subcommand in {"remove", "rm", "delete"} else subcommand
                result = _cron_api(action=action, job_id=job_id, reason="paused from /cron" if action == "pause" else None)
                if not result.get("success"):
                    print(f"(x_x) Failed to {action} job: {result.get('error')}")
                    return
                if action == "pause":
                    print(f"(^_^)b Paused job: {result['job']['name']} ({job_id})")
                elif action == "resume":
                    print(f"(^_^)b Resumed job: {result['job']['name']} ({job_id})")
                    print(f"  Next run: {result['job'].get('next_run_at')}")
                elif action == "run":
                    print(f"(^_^)b Triggered job: {result['job']['name']} ({job_id})")
                    print("  It will run on the next scheduler tick.")
                else:
                    removed = result.get("removed_job", {})
                    print(f"(^_^)b Removed job: {removed.get('name', job_id)} ({job_id})")
                return
    
            print(f"(._.) Unknown cron command: {subcommand}")
            print("  Available: list, add, edit, pause, resume, run, remove")

        def _handle_notebook_command(self, cmd: str):
            """Handle the /notebook command for research and study."""
            from rich.table import Table
            from agent.display import ChatConsole
            
            cc = ChatConsole()
            parts = cmd.split()
            subcommand = parts[1].lower() if len(parts) > 1 else None
            
            if not subcommand:
                # Show the NotebookLM Dashboard
                table = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
                table.add_row("[bold cyan]discover[/]", "Search arXiv & Web for high-quality sources")
                table.add_row("[bold cyan]brief[/]", "Generate AI study guides & flashcards from sources")
                table.add_row("[bold cyan]podcast[/]", "Create a 2-speaker 'Deep Dive' audio overview (.mp3)")
                table.add_row("[bold cyan]map[/]", "Visualize connections in your Context Graph (Mermaid)")
                table.add_row("[bold cyan]status[/]", "Check active research folder and source counts")
                table.add_row("[bold cyan]mode[/]", "Toggle Strict Grounded RAG mode (Notebook Mode)")
    
                panel = Panel(
                    table,
                    title="[bold blue]☤ Hermes NotebookLM Research Lab[/]",
                    subtitle="[dim]Usage: /notebook <command> [args][/]",
                    border_style="blue",
                    padding=(1, 2)
                )
                cc.print(panel)
                return
    
            if subcommand == "discover":
                topic = " ".join(parts[2:])
                if not topic:
                    cc.print("  [bold red]Usage: /notebook discover <topic>[/]")
                    return
                cc.print(f"\n⚡ [bold yellow]Discovery initiated for:[/] {topic}")
                msg = f"Use the source-discovery skill to find high-quality sources on '{topic}' and save them to my research folder."
                if hasattr(self, "_pending_input"):
                    self._pending_input.put(msg)
                    
            elif subcommand == "brief":
                cc.print("\n🎙️ [bold yellow]Manually triggering Obsidian Studio Briefings...[/]")
                import subprocess
                try:
                    script_path = str(PROJECT_ROOT / "hermes-agent/cron/obsidian_ingest.py")
                    subprocess.run([sys.executable, script_path], check=True)
                    cc.print("✅ Briefing generation check complete. Check your Obsidian vault!")
                except Exception as e:
                    cc.print(f"❌ Error triggering briefing: {e}")
    
            elif subcommand == "podcast":
                source = " ".join(parts[2:])
                if not source:
                    cc.print("  [bold red]Usage: /notebook podcast <source_file_or_folder> [--style deep_dive|quick_brief|debate][/]")
                    return
                # Parse optional --style flag
                style = "deep_dive"
                if "--style" in source:
                    style_parts = source.split("--style")
                    source = style_parts[0].strip()
                    style = style_parts[1].strip().split()[0] if style_parts[1].strip() else "deep_dive"
                cc.print(f"\n🎧 [bold yellow]Generating Alex & Sam Podcast from:[/] {source}")
                cc.print(f"   [dim]Style: {style} | This may take a few minutes...[/]")
                msg = f"Use the podcast_generate tool with source='{source}' and style='{style}'."
                if hasattr(self, "_pending_input"):
                    self._pending_input.put(msg)
    
            elif subcommand == "map":
                topic = " ".join(parts[2:])
                cc.print(f"\n🧠 [bold yellow]Mapping Context Graph for:[/] {topic or 'General Context'}")
                msg = f"Use the mind-map skill to visualize my knowledge graph related to '{topic or 'current context'}'."
                if hasattr(self, "_pending_input"):
                    self._pending_input.put(msg)
    
            elif subcommand == "mode":
                cc.print("\n🔐 [bold yellow]Activating Notebook Mode (Strict Grounded RAG)...[/]")
                msg = "Activate notebook-mode. Remind me of the strict rules and ask for the source directory."
                if hasattr(self, "_pending_input"):
                    self._pending_input.put(msg)
    
            elif subcommand == "status":
                from agent.wiki_paths import resolve_llm_wiki_path, resolve_obsidian_vault_path
    
                vault = resolve_obsidian_vault_path()
                wiki_path = resolve_llm_wiki_path()
                vault_path = str(vault) if vault else "Not set"
                cc.print(f"\n📊 [bold blue]Notebook Status[/]")
                cc.print(f"  [bold]Vault Path:[/] {vault_path}")
                cc.print(f"  [bold]Wiki Path:[/] {wiki_path}")
                try:
                    res_dir = Path(os.path.expanduser(vault_path)) / "Hermes" / "Research"
                    if res_dir.exists():
                        docs = list(res_dir.glob("*.md"))
                        briefs = [d for d in docs if d.name.endswith("_Briefing.md")]
                        sources = [d for d in docs if not d.name.endswith("_Briefing.md")]
                        cc.print(f"  [bold]Research Sources:[/] {len(sources)}")
                        cc.print(f"  [bold]Studio Briefings:[/] {len(briefs)}")
                    else:
                        cc.print("  [dim]Research folder not found in vault.[/]")
                except Exception:
                    cc.print("  [dim]Could not access research folder.[/]")
            else:
                cc.print(f"  [bold red]Unknown notebook subcommand:[/] {subcommand}")
                cc.print("  Available: discover, brief, podcast, map, mode, status")

        def _handle_wiki_command(self, cmd: str):
            """Handle /wiki [init|status|lint|ingest|review|map|file-query|compare|entity|concept] [domain|source]."""
            from agent.wiki_command import run_wiki_command
    
            parts = cmd.split(maxsplit=2)
            subcommand = parts[1].lower() if len(parts) > 1 else "status"
            argument = parts[2] if len(parts) > 2 else ""
            _cprint(run_wiki_command(subcommand, argument))

        def _handle_skills_command(self, cmd: str):
            """Handle /skills slash command — delegates to hermes_cli.skills_hub."""
            from hermes_cli.skills_hub import handle_skills_slash
            handle_skills_slash(cmd, ChatConsole())

        def _handle_plan_command(self, cmd: str):
            """Handle /plan [request] — load the bundled plan skill."""
            parts = cmd.strip().split(maxsplit=1)
            user_instruction = parts[1].strip() if len(parts) > 1 else ""
    
            plan_path = build_plan_path(user_instruction)
            msg = build_skill_invocation_message(
                "/plan",
                user_instruction,
                task_id=self.session_id,
                runtime_note=(
                    "Save the markdown plan with write_file to this exact relative path "
                    f"inside the active workspace/backend cwd: {plan_path}"
                ),
            )
    
            if not msg:
                self.console.print("[bold red]Failed to load the bundled /plan skill[/]")
                return
    
            _cprint(f"  📝 Plan mode queued via skill. Markdown plan target: {plan_path}")
            if hasattr(self, '_pending_input'):
                self._pending_input.put(msg)
            else:
                self.console.print("[bold red]Plan mode unavailable: input queue not initialized[/]")

        def _handle_background_command(self, cmd: str):
            """Handle /background <prompt> — run a prompt in a separate background session.
    
            Spawns a new AIAgent in a background thread with its own session.
            When it completes, prints the result to the CLI without modifying
            the active session's conversation history.
            """
            parts = cmd.strip().split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                _cprint("  Usage: /background <prompt>")
                _cprint("  Example: /background Summarize the top HN stories today")
                _cprint("  The task runs in a separate session and results display here when done.")
                return
    
            prompt = parts[1].strip()
            self._background_task_counter += 1
            task_num = self._background_task_counter
            task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
            # Make sure we have valid credentials
            if not self._ensure_runtime_credentials():
                _cprint("  (>_<) Cannot start background task: no valid credentials.")
                return
    
            _cprint(f"  🔄 Background task #{task_num} started: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
            _cprint(f"  Task ID: {task_id}")
            _cprint("  You can continue chatting — results will appear when done.\n")
    
            turn_route = self._resolve_turn_agent_config(prompt)
    
            def run_background():
                try:
                    bg_agent = AIAgent(
                        model=turn_route["model"],
                        api_key=turn_route["runtime"].get("api_key"),
                        base_url=turn_route["runtime"].get("base_url"),
                        provider=turn_route["runtime"].get("provider"),
                        api_mode=turn_route["runtime"].get("api_mode"),
                        acp_command=turn_route["runtime"].get("command"),
                        acp_args=turn_route["runtime"].get("args"),
                        max_iterations=self.max_turns,
                        enabled_toolsets=self.enabled_toolsets,
                        quiet_mode=True,
                        verbose_logging=False,
                        session_id=task_id,
                        platform="cli",
                        session_db=self._session_db,
                        reasoning_config=self.reasoning_config,
                        providers_allowed=self._providers_only,
                        providers_ignored=self._providers_ignore,
                        providers_order=self._providers_order,
                        provider_sort=self._provider_sort,
                        provider_require_parameters=self._provider_require_params,
                        provider_data_collection=self._provider_data_collection,
                        fallback_model=self._fallback_model,
                    )
                    # Silence raw spinner; route thinking through TUI widget when no foreground agent is active.
                    bg_agent._print_fn = lambda *_a, **_kw: None
    
                    def _bg_thinking(text: str) -> None:
                        # Concurrent bg tasks may race on _spinner_text; acceptable for best-effort UI.
                        if not self._agent_running:
                            self._spinner_text = text
                            if self._app:
                                self._app.invalidate()
    
                    bg_agent.thinking_callback = _bg_thinking
    
                    result = bg_agent.run_conversation(
                        user_message=prompt,
                        task_id=task_id,
                    )
    
                    response = result.get("final_response", "") if result else ""
                    if not response and result and result.get("error"):
                        response = f"Error: {result['error']}"
    
                    # Display result in the CLI (thread-safe via patch_stdout).
                    # Force a TUI refresh first so spinner/status bar don't overlap
                    # with the output (fixes #2718).
                    if self._app:
                        self._app.invalidate()
                        import time as _tmod
                        _tmod.sleep(0.05)  # brief pause for refresh
                    print()
                    ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
                    _cprint(f"  ✅ Background task #{task_num} complete")
                    _cprint(f"  Prompt: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
                    ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
                    if response:
                        try:
                            from hermes_cli.skin_engine import get_active_skin
                            _skin = get_active_skin()
                            label = _skin.get_branding("response_label", "⚕ Hermes")
                            _resp_color = _skin.get_color("response_border", "#CD7F32")
                            _resp_text = _skin.get_color("banner_text", "#FFF8DC")
                        except Exception:
                            label = "⚕ Hermes"
                            _resp_color = "#CD7F32"
                            _resp_text = "#FFF8DC"
    
                        _chat_console = ChatConsole()
                        _chat_console.print(Panel(
                            _rich_text_from_ansi(response),
                            title=f"[{_resp_color} bold]{label} (background #{task_num})[/]",
                            title_align="left",
                            border_style=_resp_color,
                            style=_resp_text,
                            box=rich_box.HORIZONTALS,
                            padding=(1, 2),
                        ))
                    else:
                        _cprint("  (No response generated)")
    
                    # Play bell if enabled
                    if self.bell_on_complete:
                        sys.stdout.write("\a")
                        sys.stdout.flush()
    
                except Exception as e:
                    # Same TUI refresh pattern as success path (#2718)
                    if self._app:
                        self._app.invalidate()
                        import time as _tmod
                        _tmod.sleep(0.05)
                    print()
                    _cprint(f"  ❌ Background task #{task_num} failed: {e}")
                finally:
                    self._background_tasks.pop(task_id, None)
                    # Clear spinner only if no foreground agent owns it
                    if not self._agent_running:
                        self._spinner_text = ""
                    if self._app:
                        self._invalidate(min_interval=0)
    
            thread = threading.Thread(target=run_background, daemon=True, name=f"bg-task-{task_id}")
            self._background_tasks[task_id] = thread
            thread.start()

        def _handle_teach_command(self, cmd: str):
            """Handle /teach <topic> — Professor Emeritus Feynman-style explanation.

            Loads the canonical teach_professor_emeritus.md prompt and prepends
            it to the user's topic as a directive block, then queues the
            combined prompt for the main agent loop. The result is persisted
            like any normal turn (unlike /btw which is ephemeral).
            """
            parts = cmd.strip().split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                _cprint("  Usage: /teach <topic>")
                _cprint("  Example: /teach entropy")
                _cprint("  Asks a Professor Emeritus to explain the topic from first principles.")
                return

            import os
            topic = parts[1].strip()
            # handlers.py is at hermes-agent/hermes_cli/cmd_handlers/handlers.py
            # → three dirname()s to reach hermes-agent/, then agent/prompts.
            prompt_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                ),
                "agent",
                "prompts",
                "teach_professor_emeritus.md",
            )
            try:
                with open(prompt_path, encoding="utf-8") as fh:
                    system_prompt = fh.read().strip()
            except OSError as exc:
                _cprint(f"  ❌ /teach: failed to load prompt at {prompt_path}: {exc}")
                return

            composed = (
                "[TEACH MODE — Professor Emeritus. Follow the contract below exactly.]\n\n"
                f"{system_prompt}\n\n"
                f"TOPIC: {topic}"
            )

            preview = topic[:60] + ("..." if len(topic) > 60 else "")
            _cprint(f'  🎓 /teach: "{preview}"')

            if hasattr(self, "_pending_input"):
                self._pending_input.put(composed)
            else:
                _cprint("  ❌ /teach unavailable: input queue not initialized")

        def _handle_btw_command(self, cmd: str):
            """Handle /btw <question> — ephemeral side question using session context.
    
            Snapshots the current conversation history, spawns a no-tools agent in
            a background thread, and prints the answer without persisting anything
            to the main session.
            """
            parts = cmd.strip().split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                _cprint("  Usage: /btw <question>")
                _cprint("  Example: /btw what module owns session title sanitization?")
                _cprint("  Answers using session context. No tools, not persisted.")
                return
    
            question = parts[1].strip()
            task_id = f"btw_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
            if not self._ensure_runtime_credentials():
                _cprint("  (>_<) Cannot start /btw: no valid credentials.")
                return
    
            turn_route = self._resolve_turn_agent_config(question)
            history_snapshot = list(self.conversation_history)
    
            preview = question[:60] + ("..." if len(question) > 60 else "")
            _cprint(f'  💬 /btw: "{preview}"')
    
            def run_btw():
                try:
                    btw_agent = AIAgent(
                        model=turn_route["model"],
                        api_key=turn_route["runtime"].get("api_key"),
                        base_url=turn_route["runtime"].get("base_url"),
                        provider=turn_route["runtime"].get("provider"),
                        api_mode=turn_route["runtime"].get("api_mode"),
                        acp_command=turn_route["runtime"].get("command"),
                        acp_args=turn_route["runtime"].get("args"),
                        max_iterations=8,
                        enabled_toolsets=[],
                        quiet_mode=True,
                        verbose_logging=False,
                        session_id=task_id,
                        platform="cli",
                        reasoning_config=self.reasoning_config,
                        providers_allowed=self._providers_only,
                        providers_ignored=self._providers_ignore,
                        providers_order=self._providers_order,
                        provider_sort=self._provider_sort,
                        provider_require_parameters=self._provider_require_params,
                        provider_data_collection=self._provider_data_collection,
                        fallback_model=self._fallback_model,
                        session_db=None,
                        skip_memory=True,
                        skip_context_files=True,
                        persist_session=False,
                    )
    
                    btw_prompt = (
                        "[Ephemeral /btw side question. Answer using the conversation "
                        "context. No tools available. Be direct and concise.]\n\n"
                        + question
                    )
                    result = btw_agent.run_conversation(
                        user_message=btw_prompt,
                        conversation_history=history_snapshot,
                        task_id=task_id,
                        sync_honcho=False,
                    )
    
                    response = (result.get("final_response") or "") if result else ""
                    if not response and result and result.get("error"):
                        response = f"Error: {result['error']}"
    
                    # TUI refresh before printing
                    if self._app:
                        self._app.invalidate()
                        time.sleep(0.05)
                    print()
    
                    if response:
                        try:
                            from hermes_cli.skin_engine import get_active_skin
                            _skin = get_active_skin()
                            _resp_color = _skin.get_color("response_border", "#4F6D4A")
                        except Exception:
                            _resp_color = "#4F6D4A"
    
                        ChatConsole().print(Panel(
                            _rich_text_from_ansi(response),
                            title=f"[{_resp_color} bold]⚕ /btw[/]",
                            title_align="left",
                            border_style=_resp_color,
                            box=rich_box.HORIZONTALS,
                            padding=(1, 2),
                        ))
                    else:
                        _cprint("  💬 /btw: (no response)")
    
                    if self.bell_on_complete:
                        sys.stdout.write("\a")
                        sys.stdout.flush()
    
                except Exception as e:
                    if self._app:
                        self._app.invalidate()
                        time.sleep(0.05)
                    print()
                    _cprint(f"  ❌ /btw failed: {e}")
                finally:
                    if self._app:
                        self._invalidate(min_interval=0)
    
            thread = threading.Thread(target=run_btw, daemon=True, name=f"btw-{task_id}")
            thread.start()

        @staticmethod
        def _try_launch_chrome_debug(port: int, system: str) -> bool:
            """Try to launch Chrome/Chromium with remote debugging enabled.
    
            Returns True if a launch command was executed (doesn't guarantee success).
            """
            import shutil
            import subprocess as _sp
    
            candidates = []
            if system == "Darwin":
                # macOS: try common app bundle locations
                for app in (
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                    "/Applications/Chromium.app/Contents/MacOS/Chromium",
                    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
                    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                ):
                    if os.path.isfile(app):
                        candidates.append(app)
            else:
                # Linux: try common binary names
                for name in ("google-chrome", "google-chrome-stable", "chromium-browser",
                             "chromium", "brave-browser", "microsoft-edge"):
                    path = shutil.which(name)
                    if path:
                        candidates.append(path)
    
            if not candidates:
                return False
    
            chrome = candidates[0]
            try:
                _sp.Popen(
                    [chrome, f"--remote-debugging-port={port}"],
                    stdout=_sp.DEVNULL,
                    stderr=_sp.DEVNULL,
                    start_new_session=True,  # detach from terminal
                )
                return True
            except Exception:
                return False

        def _handle_browser_command(self, cmd: str):
            """Handle /browser connect|disconnect|status — manage live Chrome CDP connection."""
            import platform as _plat
    
            parts = cmd.strip().split(None, 1)
            sub = parts[1].lower().strip() if len(parts) > 1 else "status"
    
            _DEFAULT_CDP = "http://localhost:9222"
            current = os.environ.get("BROWSER_CDP_URL", "").strip()
    
            if sub.startswith("connect"):
                # Optionally accept a custom CDP URL: /browser connect ws://host:port
                connect_parts = cmd.strip().split(None, 2)  # ["/browser", "connect", "ws://..."]
                cdp_url = connect_parts[2].strip() if len(connect_parts) > 2 else _DEFAULT_CDP
    
                # Clear any existing browser sessions so the next tool call uses the new backend
                try:
                    from tools.browser_tool import cleanup_all_browsers
                    cleanup_all_browsers()
                except Exception:
                    pass
    
                print()
    
                # Extract port for connectivity checks
                _port = 9222
                try:
                    _port = int(cdp_url.rsplit(":", 1)[-1].split("/")[0])
                except (ValueError, IndexError):
                    pass
    
                # Check if Chrome is already listening on the debug port
                import socket
                _already_open = False
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect(("127.0.0.1", _port))
                    s.close()
                    _already_open = True
                except (OSError, socket.timeout):
                    pass
    
                if _already_open:
                    print(f"   ✓ Chrome is already listening on port {_port}")
                elif cdp_url == _DEFAULT_CDP:
                    # Try to auto-launch Chrome with remote debugging
                    print("   Chrome isn't running with remote debugging — attempting to launch...")
                    _launched = self._try_launch_chrome_debug(_port, _plat.system())
                    if _launched:
                        # Wait for the port to come up
                        import time as _time
                        for _wait in range(10):
                            try:
                                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                s.settimeout(1)
                                s.connect(("127.0.0.1", _port))
                                s.close()
                                _already_open = True
                                break
                            except (OSError, socket.timeout):
                                _time.sleep(0.5)
                        if _already_open:
                            print(f"   ✓ Chrome launched and listening on port {_port}")
                        else:
                            print(f"   ⚠ Chrome launched but port {_port} isn't responding yet")
                            print("     You may need to close existing Chrome windows first and retry")
                    else:
                        print("   ⚠ Could not auto-launch Chrome")
                        # Show manual instructions as fallback
                        sys_name = _plat.system()
                        if sys_name == "Darwin":
                            chrome_cmd = 'open -a "Google Chrome" --args --remote-debugging-port=9222'
                        elif sys_name == "Windows":
                            chrome_cmd = 'chrome.exe --remote-debugging-port=9222'
                        else:
                            chrome_cmd = "google-chrome --remote-debugging-port=9222"
                        print(f"     Launch Chrome manually: {chrome_cmd}")
                else:
                    print(f"   ⚠ Port {_port} is not reachable at {cdp_url}")
    
                os.environ["BROWSER_CDP_URL"] = cdp_url
                print()
                print("🌐 Browser connected to live Chrome via CDP")
                print(f"   Endpoint: {cdp_url}")
                print()
    
                # Inject context message so the model knows
                if hasattr(self, '_pending_input'):
                    self._pending_input.put(
                        "[System note: The user has connected your browser tools to their live Chrome browser "
                        "via Chrome DevTools Protocol. Your browser_navigate, browser_snapshot, browser_click, "
                        "and other browser tools now control their real browser — including any pages they have "
                        "open, logged-in sessions, and cookies. They likely opened specific sites or logged into "
                        "services before connecting. Please await their instruction before attempting to operate "
                        "the browser. When you do act, be mindful that your actions affect their real browser — "
                        "don't close tabs or navigate away from pages without asking.]"
                    )
    
            elif sub == "disconnect":
                if current:
                    os.environ.pop("BROWSER_CDP_URL", None)
                    try:
                        from tools.browser_tool import cleanup_all_browsers
                        cleanup_all_browsers()
                    except Exception:
                        pass
                    print()
                    print("🌐 Browser disconnected from live Chrome")
                    print("   Browser tools reverted to default mode (local headless or Browserbase)")
                    print()
    
                    if hasattr(self, '_pending_input'):
                        self._pending_input.put(
                            "[System note: The user has disconnected the browser tools from their live Chrome. "
                            "Browser tools are back to default mode (headless local browser or Browserbase cloud).]"
                        )
                else:
                    print()
                    print("Browser is not connected to live Chrome (already using default mode)")
                    print()
    
            elif sub == "status":
                print()
                if current:
                    print("🌐 Browser: connected to live Chrome via CDP")
                    print(f"   Endpoint: {current}")
    
                    _port = 9222
                    try:
                        _port = int(current.rsplit(":", 1)[-1].split("/")[0])
                    except (ValueError, IndexError):
                        pass
                    try:
                        import socket
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.settimeout(1)
                        s.connect(("127.0.0.1", _port))
                        s.close()
                        print("   Status: ✓ reachable")
                    except (OSError, Exception):
                        print("   Status: ⚠ not reachable (Chrome may not be running)")
                elif os.environ.get("BROWSERBASE_API_KEY"):
                    print("🌐 Browser: Browserbase (cloud)")
                else:
                    print("🌐 Browser: local headless Chromium (agent-browser)")
                print()
                print("   /browser connect      — connect to your live Chrome")
                print("   /browser disconnect   — revert to default")
                print()
    
            else:
                print()
                print("Usage: /browser connect|disconnect|status")
                print()
                print("   connect      Connect browser tools to your live Chrome session")
                print("   disconnect   Revert to default browser backend")
                print("   status       Show current browser mode")
                print()

        def _handle_skin_command(self, cmd: str):
            """Handle /skin [name] — show or change the display skin."""
            try:
                from hermes_cli.skin_engine import list_skins, set_active_skin, get_active_skin_name
            except ImportError:
                print("Skin engine not available.")
                return
    
            parts = cmd.strip().split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                # Show current skin and list available
                current = get_active_skin_name()
                skins = list_skins()
                print(f"\n  Current skin: {current}")
                print("  Available skins:")
                for s in skins:
                    marker = " ●" if s["name"] == current else "  "
                    source = f" ({s['source']})" if s["source"] == "user" else ""
                    print(f"   {marker} {s['name']}{source} — {s['description']}")
                print("\n  Usage: /skin <name>")
                print(f"  Custom skins: drop a YAML file in {display_hermes_home()}/skins/\n")
                return
    
            new_skin = parts[1].strip().lower()
            available = {s["name"] for s in list_skins()}
            if new_skin not in available:
                print(f"  Unknown skin: {new_skin}")
                print(f"  Available: {', '.join(sorted(available))}")
                return
    
            set_active_skin(new_skin)
            if save_config_value("display.skin", new_skin):
                print(f"  Skin set to: {new_skin} (saved)")
            else:
                print(f"  Skin set to: {new_skin}")
            print("  Note: banner colors will update on next session start.")
            if self._apply_tui_skin_style():
                print("  Prompt + TUI colors updated.")

        def _handle_reasoning_command(self, cmd: str):
            """Handle /reasoning — manage effort level and display toggle.
    
            Usage:
                /reasoning              Show current effort level and display state
                /reasoning <level>      Set reasoning effort (none, low, medium, high, xhigh)
                /reasoning show|on      Show model thinking/reasoning in output
                /reasoning hide|off     Hide model thinking/reasoning from output
            """
            parts = cmd.strip().split(maxsplit=1)
    
            if len(parts) < 2:
                # Show current state
                rc = self.reasoning_config
                if rc is None:
                    level = "medium (default)"
                elif rc.get("enabled") is False:
                    level = "none (disabled)"
                else:
                    level = rc.get("effort", "medium")
                display_state = "on ✓" if self.show_reasoning else "off"
                _cprint(f"  {_GOLD}Reasoning effort:  {level}{_RST}")
                _cprint(f"  {_GOLD}Reasoning display: {display_state}{_RST}")
                _cprint(f"  {_DIM}Usage: /reasoning <none|low|medium|high|xhigh|show|hide>{_RST}")
                return
    
            arg = parts[1].strip().lower()
    
            # Display toggle
            if arg in ("show", "on"):
                self.show_reasoning = True
                if self.agent:
                    self.agent.reasoning_callback = self._current_reasoning_callback()
                save_config_value("display.show_reasoning", True)
                _cprint(f"  {_GOLD}✓ Reasoning display: ON (saved){_RST}")
                _cprint(f"  {_DIM}  Model thinking will be shown during and after each response.{_RST}")
                return
            if arg in ("hide", "off"):
                self.show_reasoning = False
                if self.agent:
                    self.agent.reasoning_callback = self._current_reasoning_callback()
                save_config_value("display.show_reasoning", False)
                _cprint(f"  {_GOLD}✓ Reasoning display: OFF (saved){_RST}")
                return
    
            # Effort level change
            parsed = _parse_reasoning_config(arg)
            if parsed is None:
                _cprint(f"  {_DIM}(._.) Unknown argument: {arg}{_RST}")
                _cprint(f"  {_DIM}Valid levels: none, low, minimal, medium, high, xhigh{_RST}")
                _cprint(f"  {_DIM}Display:      show, hide{_RST}")
                return
    
            self.reasoning_config = parsed
            self.agent = None  # Force agent re-init with new reasoning config
    
            if save_config_value("agent.reasoning_effort", arg):
                _cprint(f"  {_GOLD}✓ Reasoning effort set to '{arg}' (saved to config){_RST}")
            else:
                _cprint(f"  {_GOLD}✓ Reasoning effort set to '{arg}' (session only){_RST}")

        def _handle_voice_command(self, command: str):
            """Handle /voice [on|off|tts|status] command."""
            parts = command.strip().split(maxsplit=1)
            subcommand = parts[1].lower().strip() if len(parts) > 1 else ""
    
            if subcommand == "on":
                self._enable_voice_mode()
            elif subcommand == "off":
                self._disable_voice_mode()
            elif subcommand == "tts":
                self._toggle_voice_tts()
            elif subcommand == "status":
                self._show_voice_status()
            elif subcommand == "":
                # Toggle
                if self._voice_mode:
                    self._disable_voice_mode()
                else:
                    self._enable_voice_mode()
            else:
                _cprint(f"Unknown voice subcommand: {subcommand}")
                _cprint("Usage: /voice [on|off|tts|status]")

        def _handle_approval_selection(self) -> None:
            """Process the currently selected dangerous-command approval choice."""
            state = self._approval_state
            if not state:
                return
    
            selected = state.get("selected", 0)
            choices = state.get("choices") or []
            if not (0 <= selected < len(choices)):
                return
    
            chosen = choices[selected]
            if chosen == "view":
                state["show_full"] = True
                state["choices"] = [choice for choice in choices if choice != "view"]
                if state["selected"] >= len(state["choices"]):
                    state["selected"] = max(0, len(state["choices"]) - 1)
                self._invalidate()
                return
    
            state["response_queue"].put(chosen)
            self._approval_state = None
            self._invalidate()

