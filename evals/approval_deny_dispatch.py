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
        'approvals:\n  mode: "off"\n  deny: ["sudo *", "printf SAFE", "git status"]\n'
        'security:\n  tirith_enabled: false\n', encoding="utf-8")
    executable = home / "sudo"
    executable.write_text('#!/bin/sh\nprintf "OWNED_EXECUTABLE_REACHED\\n"\n', encoding="utf-8")
    executable.chmod(0o700)
    # Prefer GNU env when the host's default env implements different -S syntax.
    if Path("/usr/bin/gnuenv").exists():
        (home / "env").symlink_to("/usr/bin/gnuenv")
    sys.path.insert(0, str(source))
    from tools import terminal_tool  # noqa: F401
    from tools.registry import registry
    from tools import approval_context

    binary = shlex.quote(str(executable))
    cases = [binary + " -n id -u", "env -i " + binary + " -n id -u",
             "command " + binary + " -n id -u",
             "true && " + binary + " -n id -u",
             "bash -c " + shlex.quote(binary + " -n id -u"),
             "printf '%s\\n' 'sudo -n id -u'",
             "env -S 'printf %s; sudo -n id'", "env -S printf SAFE",
             "env -S git status", "echo ok # ; bash -c " + shlex.quote(binary + " -n id"),
             "env -S 'printf %s' 'sudo -n id'", "env -S printf BENIGN",
             "env -S " + shlex.quote("bash -c " + shlex.quote(binary + " -n id")),
             "echo ok # ignored\n bash -c " + shlex.quote(binary + " -n id")]
    cases += [
        r"env -S 'printf\_SAFE'", r"env -S 'printf\_SAFE\c ignored'",
        "env -S 'printf SAFE # ignored'", "env -S 'printf # ignored' SAFE",
        "env -a marker printf SAFE", "env --argv0 marker printf SAFE",
        "env --argv0=marker printf SAFE", "env -amarker printf SAFE",
        r"env -a marker -S 'printf\_SAFE'",
        "env -S '\"printf\"\\_SAFE'", "env -S \"'printf' SAFE\"",
        r"env -S 'printf\_\_SAFE'", r"env -S 'printf\c ignored' SAFE",
        "env -a sudo printf BENIGN", "env --argv0 sudo printf BENIGN",
        "env -S 'printf %s \"sudo\\_-n\\_id\"'",
        r"env -S 'printf BENIGN\c bash -c sudo'",
        "env -S 'printf BENIGN # bash -c sudo'",
        r"env -S 'printf %s \${IGNORED}'",
    ]
    rows = []
    for command in cases:
        result = registry.dispatch("terminal", {"command": command, "workdir": directory, "timeout": 10})
        if isinstance(result, str):
            result = json.loads(result)
        rows.append({"command": command, "result": result})
    # Optional assertions keep the same harness usable for before/after receipts.
    expected_outputs = {5: 'sudo -n id -u', 6: 'sudo;-n;id;', 9: 'ok',
                        10: 'sudo -n id', 11: 'BENIGN', 27: 'BENIGN', 28: 'BENIGN',
                        29: 'sudo -n id', 30: 'BENIGN', 31: 'BENIGN', 32: '${IGNORED}'}
    if '--verify' in sys.argv:
        for index, row in enumerate(rows):
            result = row['result']
            if index in expected_outputs:
                assert result.get('exit_code') == 0, row
                assert result.get('output', '').strip() == expected_outputs[index], row
            else:
                assert result.get('status') == 'blocked', row
                assert 'user-defined deny rule' in result.get('error', ''), row
                assert not result.get('output'), row
    print(json.dumps({"source": str(source), "config": approval_context._get_approval_config(),
                      "rows": rows}, indent=2))
