"""The classic CLI's real startup and command dispatch, in a fresh process, recording every plugin
discovery / execution entry point it reaches (#134889 PLG-7).

``argv[1]`` is the launch: ``oneshot`` (``_prepare_agent_startup``, as ``hermes chat -q`` runs it) or
``interactive`` (main's fast path: discovery deferred to the first turn). The rest of argv are the
chat flags (``-t terminal``, ``-s <skill>``). The gateway for ``HERMES_HOME`` must already run, with the probe plugin
(a ``pre_command`` hook, a command, a skill and a toolset) loaded. Prints one JSON line.
"""
import io
import json
import os
import queue
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

from agent import skill_commands
from hermes_cli import plugins

calls = []


def _spy(module, name):
    real = getattr(module, name)

    def spy(*args, **kwargs):
        caller = next((f for f in traceback.extract_stack()[-2::-1] if not f.filename.endswith('plugins.py')), None)
        calls.append(f'{name} <- {Path(caller.filename).name}:{caller.lineno}' if caller else name)
        return real(*args, **kwargs)
    setattr(module, name, spy)


for _name in ('discover_plugins', 'start_background_plugin_discovery', 'fire_pre_command_hook', 'invoke_hook',
              'get_plugin_manager', 'get_plugin_toolset_keys_nowait', 'get_portable_mcp_server_names_nowait',
              'get_plugin_toolsets', 'get_plugin_commands'):
    _spy(plugins, _name)
_spy(skill_commands, 'get_plugin_skill_commands')


def _add_skill(name):
    path = Path(os.environ['HERMES_HOME'], 'skills', name)
    path.mkdir(parents=True)
    (path / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: Added mid-run\n---\nSay {name.upper()}_BODY.\n')


def _added_here(_cli):  # installed in the profile, then /reload-skills here
    _add_skill('fresh-skill')


def _added_elsewhere(cli):  # installed, then another client of this gateway rescans its session
    _add_skill('other-skill')
    cli._session_db.connection.call('skills.reload', session_id=cli.session_id)


# Typed in this order after startup: the original probe, then the paths that used to discover
# plugins in this process (help, a plugin skill, /tools and its toolset validation). ``/repo-skill`` is
# the trusted repo's own skill in this process's cwd, which the gateway does not run from. A callable
# step changes the profile's skills between commands; a repeated command's output is keyed ``<cmd> #2``.
COMMANDS = ('/plugins', '/history', '/busy queue', '/help', '/help skills', '/probe-hook:probe check it',
            '/repo-skill from the repo', _added_here, '/reload-skills', '/help skills', _added_elsewhere,
            '/other-skill go now', '/tools', '/tools disable probe_tools')
COMPLETE = ('/probe', '/tools enable ', '/tools disable ')


def main():
    mode, flags = sys.argv[1], sys.argv[2:]
    from hermes_cli import main as hermes_main
    args = hermes_main._light_chat_parser().parse_args(['chat', *flags])
    if mode == 'interactive':
        os.environ['HERMES_DEFER_AGENT_STARTUP'] = '1'
    else:
        hermes_main._prepare_agent_startup(args)
    import cli
    from prompt_toolkit.document import Document
    outputs, errors, completions = {}, [], {}
    with redirect_stdout(io.StringIO()) as out:
        c = cli._build_cli_from_args(None, args.toolsets, None, None, None, None, None, None, None, True, None,
                                     False, False, False, args.skills)
        c._ensure_gateway_session()
        agent = c._init_agent()
        c._pending_input = queue.Queue()
        for command in COMMANDS:
            if callable(command):
                command(c)
                continue
            start = out.tell()
            try:
                c.process_command(command)
            except Exception:
                errors.append(f'{command}: {traceback.format_exc()}')
            outputs[command if command not in outputs else f'{command} #2'] = out.getvalue()[start:]
        try:
            from hermes_cli.commands_completion import cli_slash_completer
            completer = cli_slash_completer(c)  # the prompt's own completer
            for text in COMPLETE:
                completions[text] = sorted({x.text for x in completer.get_completions(Document(text), None)})
        except Exception:
            errors.append(f'completion: {traceback.format_exc()}')
    queued = []
    while not c._pending_input.empty():
        queued.append(str(c._pending_input.get()))
    print(json.dumps({'calls': calls, 'agent': agent, 'session_id': c.session_id, 'output': out.getvalue(),
                      'outputs': outputs, 'errors': errors, 'completions': completions, 'queued': queued,
                      'model_tools': 'model_tools' in sys.modules,
                      'plugin_modules': sorted(m for m in sys.modules if m.startswith('hermes_plugins'))}),
          file=sys.__stdout__, flush=True)
    os._exit(0)


main()
