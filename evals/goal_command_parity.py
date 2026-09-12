"""CLI /goal semantic oracle, with real storage and no model calls.

Run against base and candidate in fresh processes with distinct temporary homes:
    python evals/goal_command_parity.py ROOT NEW_HOME OUTPUT_JSON
Compare state and queued prompts; output includes presentation differences.
"""
import importlib
import json
import os
from pathlib import Path
import queue
import sys

root, home, out = map(Path, sys.argv[1:4])
home.mkdir(parents=True, exist_ok=False)
os.environ['HERMES_HOME'] = str(home)
os.environ['HERMES_NIX_BUILD'] = '1'
sys.path.insert(0, str(root))
commands = importlib.import_module("hermes_cli.cli_commands_mixin")
GoalManager = importlib.import_module("hermes_cli.goals").GoalManager

manager = GoalManager(session_id='cli-oracle', default_max_turns=17)
cli = commands.CLICommandsMixin()
cli._get_goal_manager = lambda: manager
cli._pending_input = queue.Queue()
cli.conversation_history = []
printed = []
commands._cp = lambda *lines: printed.extend(lines)
rows = []
command_args = ['', 'show', 'pause', 'resume', 'clear', 'draft',
                'Maintain the service\nconstraints: keep data\nstop when: credentials missing',
                'show', 'pause', 'resume', 'wait nope', 'wait', 'wait 0',
                f'wait {os.getpid()} running probe', 'status', 'unwait', 'unwait',
                'gate list', 'gate remove nope', 'gate clear', 'done', 'status']
for arg in command_args:
    printed.clear()
    cli._handle_goal_command('/goal' + (' ' + arg if arg else ''))
    state = manager.state
    prompts = []
    while not cli._pending_input.empty():
        prompts.append(cli._pending_input.get_nowait())
    contract = state.contract if state else None
    rows.append({'command': arg.replace(str(os.getpid()), '<PID>'),
                 'active': manager.has_goal(),
                 'goal': state.goal if state else None,
                 'status': state.status if state else None,
                 'contract': contract.to_dict() if contract else None,
                 'prompt_count': len(prompts), 'prompts': prompts,
                 'output': '\n'.join(printed).replace(str(os.getpid()), '<PID>')})
out.write_text(json.dumps(rows, indent=2, default=str))
print(json.dumps(rows, indent=2, default=str))
