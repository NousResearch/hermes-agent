"""A file a masked heredoc body merely MENTIONED is walked only when a shell can consume it (#134313).

Hermes writes command-shaped approval descriptions (e.g. ``use 'systemctl --user restart
hermes-gateway'``) into user ``config.yaml`` ``command_allowlist`` entries. The mention walk
raw-opened that YAML and tokenized it as shell, so a benign heredoc that only READ the config was
blocked with a misattributed lifecycle message. Mentions are now gated on shell provenance
(shebang / shell suffix / executable bit); shell-routed mentions and executed references are never
gated.
"""

import os

import pytest

from cron.lifecycle_guard import (
    contains_gateway_lifecycle_command_or_referenced_script as guard,
)

_LIFECYCLE_LINE = "systemctl --user restart hermes-gateway\n"
_ALLOWLIST_CONFIG = (
    "command_allowlist:\n"
    "  - start gateway outside systemd (use 'systemctl --user restart hermes-gateway')\n"
)


def _write(tmp_path, name, text, mode=0o644):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


@pytest.mark.parametrize(
    "body",
    [
        'print(open("{path}").read())',
        "from pathlib import Path\nprint(Path('{path}').stat().st_size)",
        'import os\nt = open("{path}", encoding="utf-8").read()',
    ],
)
def test_inert_heredoc_reading_own_config_is_allowed(tmp_path, body):
    config = _write(tmp_path, "config.yaml", _ALLOWLIST_CONFIG)
    command = f"python3 - <<'PY'\n{body.format(path=config)}\nPY"
    assert guard(command, cwd=str(tmp_path)) is False


def test_mentioned_shebangless_data_file_with_lifecycle_text_is_allowed(tmp_path):
    """A data file (no shebang, no shell suffix, no executable bit) whose TEXT merely looks like a
    lifecycle command is not a verdict: without shell provenance nothing can execute that text."""
    data = _write(tmp_path, "plain.yaml", _LIFECYCLE_LINE)
    command = f"python3 - <<'PY'\nimport os\nos.system(\"{data}\")\nPY"
    assert guard(command, cwd=str(tmp_path)) is False


def test_mentioned_shebang_script_still_blocks(tmp_path):
    script = _write(
        tmp_path, "restart.sh", "#!/bin/sh\nhermes gateway restart\n", 0o755
    )
    command = f"python3 - <<'PY'\nimport os\nos.system('{script}')\nPY"
    assert guard(command, cwd=str(tmp_path)) is True


def test_mentioned_suffixless_shebang_script_still_blocks(tmp_path):
    script = _write(tmp_path, "tool", "#!/bin/sh\n" + _LIFECYCLE_LINE, 0o755)
    command = f"python3 - <<'PY'\nimport os\nos.system('{script}')\nPY"
    assert guard(command, cwd=str(tmp_path)) is True


def test_mentioned_executable_shebangless_file_still_blocks(tmp_path):
    """Direct exec of shebang-less text falls back to the shell only when the executable bit is
    set, so the bit alone is shell provenance for a mentioned file."""
    payload = _write(tmp_path, "payload.yaml", _LIFECYCLE_LINE, 0o755)
    command = f"python3 - <<'PY'\nimport os\nos.system(\"{payload}\")\nPY"
    assert guard(command, cwd=str(tmp_path)) is True


def test_mentioned_shell_routed_file_still_blocks(tmp_path):
    """``bash x`` executes any text it is handed, whatever the file looks like."""
    payload = _write(tmp_path, "payload.yaml", _LIFECYCLE_LINE, 0o644)
    command = f"python3 - <<'PY'\nbash {payload}\nPY"
    assert guard(command, cwd=str(tmp_path)) is True


def test_executed_references_never_gated(tmp_path):
    config = _write(tmp_path, "config.yaml", _ALLOWLIST_CONFIG)
    payload = _write(tmp_path, "payload.yaml", _LIFECYCLE_LINE, 0o644)
    script = _write(
        tmp_path, "restart.sh", "#!/bin/sh\nhermes gateway restart\n", 0o755
    )
    for command in (f"bash {config}", str(config), f"sh {script}", f"source {script}"):
        assert guard(command, cwd=str(tmp_path)) is True, command
    assert guard(f"cat {config}", cwd=str(tmp_path)) is False


def test_unquoted_heredoc_mention_stays_fail_closed(tmp_path):
    """An expansion-capable body is not provably inert, so its mention of the config stays
    conservative on main's behavior (executed view, no provenance gate)."""
    config = _write(tmp_path, "config.yaml", _ALLOWLIST_CONFIG)
    command = f'python3 - <<EOF\nopen("{config}")\nEOF'
    assert guard(command, cwd=str(tmp_path)) is True
