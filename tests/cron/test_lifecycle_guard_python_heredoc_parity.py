"""Python stdin and Python files have the same gateway-lifecycle verdict."""

import pytest

from cron.lifecycle_guard import (
    GatewayLifecycleBlocked,
    check_gateway_lifecycle,
    contains_gateway_lifecycle_command_or_referenced_script as guard,
)


def test_read_only_python_heredoc_does_not_execute_diagnostic_log(tmp_path):
    log = tmp_path / "intake.jsonl"
    log.write_text("hermes gateway " + "re" + "start\n")
    body = (
        "import subprocess\nfrom pathlib import Path\n"
        "subprocess.run(['sudo', '-n', 'sqlite3', "
        f"'file:{tmp_path / 'ledger.db'}?mode=ro', 'select count(*) from reviews'])\n"
        f"print(Path('{log}').read_text())\n"
    )
    script = tmp_path / "diagnose.py"
    script.write_text(body)
    assert not guard(f"python3 {script}", cwd=str(tmp_path))
    assert not guard(f"python3 - <<'PY'\n{body}PY", cwd=str(tmp_path))


def test_chained_sql_reads_and_open_loop_match_file_verdict(tmp_path):
    log = tmp_path / "intake.jsonl"
    log.write_text("hermes gateway " + "restart\n")
    body = (
        "import json,subprocess\n"
        "rows=subprocess.run(['sudo','-n','sqlite3',"
        f"'file:{tmp_path / 'reviews.sqlite3'}?mode=ro','select count(*) from reviews;'],"
        "capture_output=True,text=True).stdout.split()\n"
        f"for line in reversed(open('{log}').read().splitlines()):\n"
        "    try: d=json.loads(line)\n"
        "    except: continue\n"
        "    if d.get('type')!='intake': continue\n"
        "    k=f\"{d['repo'].lower()}#{d['pr']}\"\n"
        "    print(d)\n"
    )
    script = tmp_path / "diagnose.py"
    script.write_text(body)
    chain = (
        f"DB={tmp_path / 'reviews.sqlite3'}; echo '== running now'; "
        "echo configured | grep configured; "
        'sudo -n sqlite3 "file:$DB?mode=ro" "select count(*) from reviews;"; '
    )
    assert not guard(chain, cwd=str(tmp_path))
    assert not guard(f"python3 {script}", cwd=str(tmp_path))
    assert not guard(chain + f"python3 - <<'PY'\n{body}PY", cwd=str(tmp_path))


def test_open_loop_executing_lines_stays_blocked(tmp_path):
    log = tmp_path / "intake.jsonl"
    log.write_text("hermes gateway " + "restart\n")
    body = (
        f"for line in reversed(open('{log}').read().splitlines()):\n"
        "    __import__('os').system(line)\n"
    )
    assert guard("echo ok; " + f"python3 - <<'PY'\n{body}PY", cwd=str(tmp_path))


@pytest.mark.parametrize("body", [
    "import subprocess\nsubprocess.run(['launchctl', 'bootout', 'system/ai.hermes.gateway'])\n",
    "import subprocess\nsubprocess.run(['kill', '$(pgrep -f hermes-gateway)'])\n",
    "import subprocess\nsubprocess.run(['systemctl', 'stop', 'hermes-gateway'])\n",
])
def test_python_heredoc_self_lifecycle_stays_blocked(tmp_path, body):
    script = tmp_path / "diagnose.py"
    script.write_text(body)
    assert guard(f"python3 - <<'PY'\n{body}PY", cwd=str(tmp_path))
    with pytest.raises(GatewayLifecycleBlocked):
        check_gateway_lifecycle("diagnose", script=str(script))


def test_python_heredoc_executing_referenced_script_stays_blocked(tmp_path):
    script = tmp_path / "restart.sh"
    script.write_text("hermes gateway " + "re" + "start\n")
    body = f"import os\nos.system('{script}')\n"
    assert guard(f"python3 - <<'PY'\n{body}PY", cwd=str(tmp_path))


def test_shadowed_path_name_cannot_hide_executed_script(tmp_path):
    script = tmp_path / "restart.sh"
    script.write_text("hermes gateway " + "re" + "start\n")
    body = (
        "from pathlib import Path\n"
        "Path = lambda value: __import__('os').system(value)\n"
        f"Path('{script}').read_text()\n"
    )
    assert guard(f"python3 - <<'PY'\n{body}PY", cwd=str(tmp_path))


def test_read_text_piped_to_os_system_is_executable(tmp_path):
    data = tmp_path / "commands.txt"
    data.write_text("hermes gateway " + "re" + "start\n")
    body = f"from pathlib import Path\nimport os\nos.system(Path('{data}').read_text())\n"
    assert guard(f"python3 - <<'PY'\n{body}PY", cwd=str(tmp_path))


def test_open_loop_parser_name_does_not_change_file_parity(tmp_path):
    log = tmp_path / "intake.jsonl"
    log.write_text("hermes gateway " + "restart\n")
    body = (
        "import json\n"
        f"for line in reversed(open('{log}').read().splitlines()):\n"
        "    try: rec=json.loads(line)\n"
        "    except: continue\n"
        "    if rec.get('type')!='intake': continue\n"
        "    print(rec['repo'].lower())\n"
    )
    script = tmp_path / "diagnose.py"
    script.write_text(body)
    assert not guard(f"python3 {script}", cwd=str(tmp_path))
    assert not guard(f"python3 - <<'PY'\n{body}PY", cwd=str(tmp_path))


@pytest.mark.parametrize("prefix,inside", [
    ("json.loads = print\n", "    print(d)\n"),
    ("setattr(json, 'loads', print)\n", "    print(d)\n"),
    ("print = len\n", "    print(line)\n"),
    ("len = print\n", "    len(line)\n"),
    ("reversed = print\n", "    print(d)\n"),
    ("open = print\n", "    print(d)\n"),
    ("seen = type('Reader', (), {'add': print})()\n", "    seen.add(line)\n"),
    ("out = type('Reader', (), {'append': print})()\n", "    out.append(line)\n"),
    ("", "    d = type('Reader', (), {'get': print})()\n    d.get(line)\n"),
])
def test_open_loop_rebound_callable_does_not_hide_log(tmp_path, prefix, inside):
    log = tmp_path / "intake.jsonl"
    log.write_text("hermes gateway " + "restart\n")
    body = (
        "import json\n" + prefix
        + f"for line in reversed(open('{log}').read().splitlines()):\n"
        + "    d=json.loads(line)\n" + inside
    )
    assert guard(f"python3 - <<'PY'\n{body}PY", cwd=str(tmp_path))


@pytest.mark.parametrize("binding,nested", [
    ("match J():\n    case json: pass\n", False),
    ("match [J()]:\n    case [*json]: pass\n", False),
    ("match {'item': J()}:\n    case {**json}: pass\n", False),
    ("try: raise J()\nexcept J as json:\n", True),
    ("async def json(): pass\n", False),
])
def test_open_loop_non_name_binding_does_not_hide_log(tmp_path, binding, nested):
    log = tmp_path / "intake.jsonl"
    log.write_text("hermes gateway " + "restart\n")
    loop = (
        f"for line in reversed(open('{log}').read().splitlines()):\n"
        "    d=json.loads(line)\n"
        "    print(d)\n"
    )
    assert not guard(f"python3 - <<'PY'\nimport json\n{loop}PY", cwd=str(tmp_path))
    unsafe = "import json, os\nclass J(Exception):\n    loads=staticmethod(os.system)\n" + binding
    unsafe += "".join("    " + line for line in loop.splitlines(keepends=True)) if nested else loop
    assert guard(f"python3 - <<'PY'\n{unsafe}PY", cwd=str(tmp_path))


_OPEN_LOOP = (
    "for line in reversed(open('{log}').read().splitlines()):\n"
    "    d=json.loads(line)\n"
    "    print(d)\n"
)


@pytest.mark.parametrize("prefix", [
    # Changing a trusted object in place binds no name. The mask must be
    # granted by a whole-body allowlist, not by a list of binding shapes.
    "import json\njson.__dict__.update(loads=print)\n",
    "import json\nobject.__setattr__(json, 'loads', print)\n",
    "import sys, types\nsys.modules.update(json=types.SimpleNamespace(loads=print))\nimport json\n",
    "import json, builtins\nbuiltins.__dict__.update(print=len)\n",
    "import json\nfrom io import *\n",
    "import json\nimport importlib\n",
    "import json\njson.decoder.scanstring.__class__\n",
    "import json\nvars(json).update(loads=print)\n",
    "import json\ngetattr(json, 'loads')\n",
    "import json\ntype(json)\n",
    "import json\ndef hook(value):\n    return value\n",
    "import json\nhook = lambda value: value\n",
    "import json\nwith open('/dev/null') as handle: pass\n",
    "import json\nimport json as j\n",
    "import json\nfrom pathlib import Path as P\n",
    "import json\njson.JSONDecoder.decode = print\n",
])
def test_open_loop_mask_requires_whole_body_allowlist(tmp_path, prefix):
    log = tmp_path / "intake.jsonl"
    log.write_text("hermes gateway " + "restart\n")
    ordinary = "import json\n" + _OPEN_LOOP.format(log=log)
    unsafe = prefix + _OPEN_LOOP.format(log=log)
    assert not guard(f"echo ok; python3 - <<'PY'\n{ordinary}PY", cwd=str(tmp_path))
    assert guard(f"echo ok; python3 - <<'PY'\n{unsafe}PY", cwd=str(tmp_path))


@pytest.mark.parametrize("call", [
    "subprocess.run(['sqlite3', 'x'], shell=True)",
    "subprocess.run(cmd)",
    "subprocess.Popen(['sqlite3', 'x'])",
    "subprocess.run(['sqlite3', 'x'], env=dict())",
    "json.load(open('/dev/null'))",
])
def test_open_loop_mask_refuses_non_allowlisted_calls(tmp_path, call):
    log = tmp_path / "intake.jsonl"
    log.write_text("hermes gateway " + "restart\n")
    body = "import json, subprocess\ncmd = ['sqlite3']\n" + call + "\n" + _OPEN_LOOP.format(log=log)
    assert guard(f"echo ok; python3 - <<'PY'\n{body}PY", cwd=str(tmp_path))
