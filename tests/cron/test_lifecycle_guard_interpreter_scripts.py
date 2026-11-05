"""A directly executed non-shell script is not walked as shell code (#125378).

``./demo-tool`` with a ``#!/usr/bin/env python3`` shebang runs under Python, but the walk tokenized
its source with shell rules, promoted the string literal ``"/dev/null"`` to an executed script and
failed closed on the device. Its text is still scanned and the paths it names are still read; only
"could not scan" stops being a block, as for an interpreter heredoc body (#113944).
"""

import os

import pytest

from cron.lifecycle_guard import _runs_outside_posix_shell, scan_gateway_lifecycle

posix_only = pytest.mark.skipif(os.name == "nt", reason="needs a POSIX /dev/null device")

PYTHON_TOOL = '#!/usr/bin/env python3\nimport sys\nprint("/dev/null" in sys.argv[1:])\n'


def _script(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@posix_only
@pytest.mark.parametrize("shebang", ["#!/usr/bin/env python3", "#!/usr/bin/python3", "#!/usr/bin/env -S node"])
def test_direct_interpreter_script_with_device_literal_is_allowed(tmp_path, shebang):
    tool = _script(tmp_path, "demo-tool", PYTHON_TOOL.replace("#!/usr/bin/env python3", shebang))
    assert scan_gateway_lifecycle(f"{tool} go") == (False, None)


@posix_only
def test_same_script_fed_to_a_shell_still_fails_closed(tmp_path):
    # bash ignores the shebang and runs the text as shell, where `/dev/null` is a command.
    tool = _script(tmp_path, "demo-tool", PYTHON_TOOL)
    assert scan_gateway_lifecycle(f"bash {tool}")[0] is True


@posix_only
def test_shell_walk_after_a_lenient_direct_run_is_not_skipped(tmp_path):
    tool = _script(tmp_path, "demo-tool", PYTHON_TOOL)
    assert scan_gateway_lifecycle(f"{tool} go; bash {tool}")[0] is True


@posix_only
@pytest.mark.parametrize("text", ["#!/bin/sh\n/dev/null\n", "/dev/null\n"])
def test_shell_or_shebangless_script_still_fails_closed(tmp_path, text):
    tool = _script(tmp_path, "demo-tool", text)
    assert scan_gateway_lifecycle(f"{tool}")[0] is True


def test_lifecycle_command_in_interpreter_script_is_still_blocked(tmp_path):
    tool = _script(tmp_path, "demo-tool", "#!/usr/bin/env python3\nimport os\nos.system('hermes gateway restart')\n")
    assert scan_gateway_lifecycle(f"{tool}") == (True, None)


def test_script_named_by_interpreter_script_is_still_read(tmp_path):
    restart = _script(tmp_path, "restart.sh", "#!/bin/sh\nhermes gateway restart\n")
    tool = _script(tmp_path, "demo-tool", f"#!/usr/bin/env python3\nimport os\nos.system('{restart}')\n")
    assert scan_gateway_lifecycle(f"{tool}") == (True, None)


@pytest.mark.parametrize(("text", "expected"), [
    ("#!/usr/bin/env python3\n", True),
    ("#!/usr/bin/python3 -u\n", True),
    ("#! /usr/bin/env PYTHONPATH=. -S node --flag\n", True),
    ("#!/bin/bash\n", False),
    ("#!/usr/bin/env sh\n", False),
    ("#!/usr/bin/env\n", False),
    ("#!\n", False),
    ("echo hi\n", False),
    ("", False),
])
def test_runs_outside_posix_shell(text, expected):
    assert _runs_outside_posix_shell(text) is expected
