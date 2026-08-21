"""Config-path bridge tests for the registry-fed redaction pass.

``security.redact_patterns`` in config.yaml is bridged to the
``HERMES_REDACT_PATTERNS`` env var at the entry points (hermes_cli/main.py,
cli.py, gateway/run.py) — mirroring the existing ``security.redact_secrets``
bridge. These tests drive the REAL hermes_cli.main import in a subprocess
with an isolated HERMES_HOME, so the bridge, the env carrier, and redact.py's
per-call path resolution are exercised end to end.

Precedence (main.py early bridge): env var in .env still wins — config.yaml
is the fallback, the default path is the last resort.
"""

import json
import os
import subprocess
import sys
import textwrap

CHILD = textwrap.dedent(
    """
    import os
    import hermes_cli.main  # the config->env bridge runs at import time
    from agent.redact import redact_sensitive_text

    env = os.environ.get("HERMES_REDACT_PATTERNS", "")
    out = redact_sensitive_text(
        "token CONFIGPATH" + "PATTERNLITERAL and ENVOVERRIDE" + "PATTERNLITERAL here")
    print("ENV=" + env)
    print("CONFIG_MASKED=" + str("CONFIGPATH" + "PATTERNLITERAL" not in out))
    print("ENV_MASKED=" + str("ENVOVERRIDE" + "PATTERNLITERAL" not in out))
    """
)


def _run_child(extra_env):
    env = dict(os.environ)
    env.pop("HERMES_REDACT_PATTERNS", None)
    env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", CHILD],
        capture_output=True, text=True, env=env, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith(("ENV=", "CONFIG_MASKED=", "ENV_MASKED=")):
            k, _, v = line.partition("=")
            out[k] = v
    return out


def _make_home(tmp_path, patterns_a):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "security:\n  redact_patterns: " + json.dumps(str(patterns_a)) + "\n"
    )
    return home


def test_config_path_bridged_and_used(tmp_path):
    patterns_a = tmp_path / "patterns_a.json"
    patterns_a.write_text(json.dumps({"literals": ["CONFIGPATH" + "PATTERNLITERAL"]}))
    home = _make_home(tmp_path, patterns_a)

    res = _run_child({"HERMES_HOME": str(home)})
    assert res["ENV"] == str(patterns_a)          # bridged to the env var
    assert res["CONFIG_MASKED"] == "True"         # config file drives redaction
    assert res["ENV_MASKED"] == "False"           # env file's literal not masked


def test_env_wins_over_config(tmp_path):
    patterns_a = tmp_path / "patterns_a.json"
    patterns_a.write_text(json.dumps({"literals": ["CONFIGPATH" + "PATTERNLITERAL"]}))
    home = _make_home(tmp_path, patterns_a)
    patterns_b = tmp_path / "patterns_b.json"
    patterns_b.write_text(json.dumps({"literals": ["ENVOVERRIDE" + "PATTERNLITERAL"]}))

    res = _run_child({"HERMES_HOME": str(home),
                      "HERMES_REDACT_PATTERNS": str(patterns_b)})
    assert res["ENV"] == str(patterns_b)          # env var wins
    assert res["CONFIG_MASKED"] == "False"        # config file NOT used
    assert res["ENV_MASKED"] == "True"
