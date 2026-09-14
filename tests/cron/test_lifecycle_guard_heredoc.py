"""Python stdin source is executable Python, not shell or a data-file edge."""

import json
import shlex

import pytest

import cron.lifecycle_guard as lifecycle_guard


@pytest.mark.parametrize("opener, terminator, tabs, shell_wrapper", [
    ("python3 - <<'PY'", "PY\n", False, False),
    ('python3 - <<"PY"', "PY\n", False, False),
    (r"python3 - <<\PY", "PY\n", False, False),
    ("python3 - <<-'PY'", "\tPY\n", True, False),
    ("python3.12 - <<'PY' # stdin source", "PY", False, False),
    ("python - <<'PY'", "PY\n", False, False),
    ("python3 - <<'PY'", "PY\n", False, True),
])
def test_python_heredoc_does_not_read_large_json(
    tmp_path, monkeypatch, opener, terminator, tabs, shell_wrapper,
):
    from tools import process_registry
    from tools.terminal_tool_guards import gateway_lifecycle_block

    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    data = tmp_path / "report.json"
    data.write_text(json.dumps([{}] * 20000, indent=2), encoding="utf-8")
    source = (
        "from pathlib import Path\nimport json\n"
        f"data = Path({str(data)!r})\n"
        "print(len(json.loads(data.read_text())))\n"
    )
    reads = []
    original = lifecycle_guard._read_referenced_script

    def recording_read(path, *, max_bytes=None):
        reads.append(path)
        return original(path, max_bytes=max_bytes)

    monkeypatch.setattr(lifecycle_guard, "_read_referenced_script", recording_read)
    if tabs:
        source = "".join("\t" + line for line in source.splitlines(keepends=True))
    command = f"{opener}\n{source}{terminator}"
    if shell_wrapper:
        command = "sh -c " + shlex.quote(command)
    assert gateway_lifecycle_block(
        command=command, env=None, env_type="local", cwd=str(tmp_path),
        workdir=str(tmp_path), session_key="heredoc-regression",
    ) is None
    lifecycle_guard.check_gateway_lifecycle(command)
    assert data not in reads

    # An enabled guard still refuses a real lifecycle action and oversized root
    # source. Proving a body boundary must never discount the original input.
    guard = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script
    assert guard("hermes gateway restart")
    for padding in (
        "\n" * lifecycle_guard._MAX_LIFECYCLE_SCAN_LINES,
        "#" * (lifecycle_guard._MAX_LIFECYCLE_SCAN_LINE_BYTES + 1),
        "# padded source\n" * (lifecycle_guard._MAX_LIFECYCLE_SCAN_BYTES // 10),
    ):
        assert guard(f"{opener}\n{source}{padding}\n{terminator}", cwd=str(tmp_path))


@pytest.mark.parametrize("command", [
    # Literal process operands need Python AST discovery, including quoted argv
    # paths which are not shell source. No fixture commands are ever executed.
    "python3 - <<'PY'\nimport subprocess\nsubprocess.run(['sh', './child helper.sh'])\nPY\n",
    'python3 - <<"PY"\nfrom subprocess import run as launch\nlaunch(args=["./child helper.sh"])\nPY\n',
    "python3 - <<\\PY\nimport os\nos.system(\"sh './child helper.sh'\")\nPY\n",
    "python3 - <<-'PY'\n\timport asyncio\n\tasync def main():\n"
    "\t    await asyncio.create_subprocess_exec('sh', './child helper.sh')\n\tasyncio.run(main())\n\tPY\n",
    # Commands outside the body remain shell source.
    "./child.sh\npython3 - <<'PY'\npass\nPY\n",
    "python3 - <<'PY'\npass\nPY\n./child.sh\n",
    "./child.sh; python3 - <<'PY'\npass\nPY\n",
    "python3 - <<'PY'; ./child.sh\npass\nPY\n",
    # Ambiguous, expanding, malformed and non-Python consumers keep the old walk.
    "bash <<'PY'\n./child.sh\nPY\n",
    "sh <<'PY'\npython3 - <<'INNER'\npass\nINNER\n./child.sh\nPY\n",
    "python3 - <<PY\nx = '$(hermes gateway restart)'\nPY\n",
    "python3 - <<PY\nx = '`hermes gateway restart`'\nPY\n",
    "python3 - <<'PY'\n('./child.sh')\n",
    "python3 - <<''\nimport os\nos.system('./child.sh')\n",
    'python3 - <<""\nimport os\nos.system(\'./child.sh\')\n',
    "python3 - <<'PY'\n('./child.sh')\n PY\n",
    "python3 - <<'PY\n./child.sh\nPY\n",
    "python3 - <<'PY'\n./child.sh\nPY\n",  # invalid Python: shell fallback
    "sudo python3 - <<'PY'\n('./child.sh')\nPY\n",
    "env python3 - <<'PY'\n('./child.sh')\nPY\n",
    "python3 - <<'PY' >$(sh ./child.sh)\npass\nPY\n",
    "python3 - <<'ONE' <<'TWO'\npass\nONE\n('./child.sh')\nTWO\n",
    "./python3 - <<'PY'\npass\nPY\n",
    "./env python3 - <<'PY'\npass\nPY\n",
    "sh -c \"python3 - <<'PY'\nimport os\nos.system('./child.sh')\nPY\"",
])
def test_heredoc_executable_edges_stay_guarded(tmp_path, command):
    guard = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script
    for name in ("python3", "env"):
        (tmp_path / name).write_text("#!/bin/sh\n./child.sh\n", encoding="utf-8")
    for unsafe in (False, True):
        for child in ("child.sh", "child helper.sh"):
            (tmp_path / child).write_text(
                '#!/bin/sh\n' + ('hermes gate"way" re"start"\n' if unsafe else 'exit 0\n'),
                encoding="utf-8",
            )
        # Direct substitutions block independently of the child fixture.
        if unsafe or "hermes gateway restart" not in command:
            assert guard(command, cwd=str(tmp_path)) is unsafe
    # The cron caller uses the same walk with script-relative path resolution.
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(command, encoding="utf-8")
    with pytest.raises(lifecycle_guard.GatewayLifecycleBlocked):
        lifecycle_guard.check_gateway_lifecycle("inspect", str(wrapper))

    for child in ("child.sh", "child helper.sh"):
        (tmp_path / child).write_text(
            "# bounded descendant\n" * lifecycle_guard._MAX_LIFECYCLE_SCAN_LINES,
            encoding="utf-8",
        )
    assert guard(command, cwd=str(tmp_path))
