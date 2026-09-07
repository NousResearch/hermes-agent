"""Offline production-dispatch receipt; never invokes a privileged binary.

Run with the shared Python environment and a source tree argument to compare
base/head. The executable named sudo is an owned temporary marker script.
"""
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile

source = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix="hermes-deny-dispatch-") as directory:
    home = Path(directory)
    os.environ.clear()
    # The terminal's own NOPASSWD probe must also resolve to our inert fixture.
    os.environ.update(PATH=directory + ":/usr/bin:/bin", HOME=directory, HERMES_HOME=directory,
                      HERMES_INTERACTIVE="1", TERMINAL_ENV="local")
    (home / "config.yaml").write_text(
        'approvals:\n  mode: "off"\n  deny: ["sudo *"]\n'
        'security:\n  tirith_enabled: false\n', encoding="utf-8")
    executable = home / "sudo"
    executable.write_text('#!/bin/sh\nprintf "OWNED_EXECUTABLE_REACHED\\n"\n', encoding="utf-8")
    executable.chmod(0o700)
    sys.path.insert(0, str(source))
    from tools import terminal_tool  # noqa: F401
    from tools.registry import registry
    from tools import approval_context

    binary = shlex.quote(str(executable))
    cases = [binary + " -n id -u", "env -i " + binary + " -n id -u",
             "command " + binary + " -n id -u",
             "true && " + binary + " -n id -u",
             "bash -c " + shlex.quote(binary + " -n id -u"),
             "printf '%s\\n' 'sudo -n id -u'"]
    rows = []
    for command in cases:
        result = registry.dispatch("terminal", {"command": command, "workdir": directory, "timeout": 10})
        if isinstance(result, str):
            result = json.loads(result)
        rows.append({"command": command, "result": result})
    print(json.dumps({"source": str(source), "config": approval_context._get_approval_config(),
                      "rows": rows}, indent=2))
