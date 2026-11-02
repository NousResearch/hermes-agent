"""Run a workflow step's bash against a loopback object store.

The step's python/python3 calls are routed through a driver that points
scripts.releases.r2 at the test server, so staging and rendering steps run
exactly as written in the workflow.
"""
import os
from pathlib import Path
import shlex
import subprocess
import sys

from tests.ci.test_desktop_release_tag_admission import _BASH, _child_env, _workflow

ROOT = Path(__file__).resolve().parents[2]


def step_script(job, name):
    return next(step['run'] for step in _workflow()['jobs'][job]['steps'] if step.get('name') == name)


def shell_step(tmp_path, r2_server, job, name, env, *, script=None):
    helper = tmp_path / 'bin'
    helper.mkdir(exist_ok=True)
    driver = helper / 'python-driver.py'
    driver.write_text(
        'import runpy,sys\n'
        f'sys.path.insert(0, {str(ROOT)!r})\n'
        'from scripts.releases import r2\n'
        f'r2.s3_endpoint=lambda _: "http://127.0.0.1:{r2_server.server_port}"\n'
        'args=sys.argv[1:]\n'
        'assert args[:2] == ["-m", "scripts.releases.handoff"] or args[:2] == ["-m", "scripts.releases.upload_summary"] or '
        'args[:1] in (["scripts/render-builds-table.py"], ["-"]), args\n'
        'if args[:1] == ["-m"]:\n'
        '    sys.argv=args[1:]\n'
        '    runpy.run_module(args[1],run_name="__main__")\n'
        'elif args[:1] == ["-"]:\n'
        '    sys.argv=args\n'
        '    exec(compile(sys.stdin.read(), "workflow-inline", "exec"))\n'
        'else:\n'
        '    sys.argv=args\n'
        f'    runpy.run_path({str(ROOT / "scripts/render-builds-table.py")!r},run_name="__main__")\n',
        encoding='utf-8',
    )
    for name_ in ['python', 'python3']:
        command = helper / name_
        command.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"\n',
                           encoding='utf-8', newline='\n')
        command.chmod(0o755)
    script_file = tmp_path / 'step.sh'
    script_file.write_text(script if script is not None else step_script(job, name), encoding='utf-8', newline='\n')
    environment = _child_env(**env)
    environment['PATH'] = str(helper) + os.pathsep + environment['PATH']
    return subprocess.run([_BASH, '-e', '-o', 'pipefail', str(script_file)], cwd=tmp_path,
                          env=environment, capture_output=True, text=True, encoding='utf-8', timeout=60, check=False)
