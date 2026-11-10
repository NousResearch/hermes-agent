"""Profile terminal-boundary contracts from the #82936 class map.

The former xfails were historical witnesses, not current production defects:
* #81952: interactive loads expose FailedConfigRead; noninteractive admission
  refuses corruption (tests/hermes_cli/test_profile_config_recovery_contract.py).
* #83346: ingress pins RoutingIdentity; supplying two explicit profile arguments
  to build_session_key is not ambient fallback (test_session_identity.py).
* #80318: profiles are independent config islands. Root MoA presets are not
  implicit grants (tests/hermes_cli/test_profile_moa_isolation.py).
* #83197/#83557: current cron ownership must be acquired and delivery is queued;
  the fake exec-1/old _deliver_result witness never reached that boundary
  (tests/cron/test_profile_delivery_scope.py).

These scoped contracts do not certify closure of the entire issue class or a
live multi-tenant deployment. Member 1 is cross-listed in credential EPIC #83565.
"""
from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_multiplex_terminal_env_excludes_other_profiles_undeclared_secret(monkeypatch):
    """The target neither owns nor grants the launch profile's secret."""
    from agent.secret_scope import (
        set_multiplex_active,
        set_secret_scope,
        reset_secret_scope,
    )
    from tools.environments.local import _make_run_env

    monkeypatch.setenv("SERVICE_PASSWORD", "default-profile-secret")
    set_multiplex_active(True)
    token = set_secret_scope({"SECONDARY_ONLY_KEY": "b-value"})
    try:
        run_env = _make_run_env({})
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)
    assert "SERVICE_PASSWORD" not in run_env, (
        "secondary profile's terminal subprocess sees the default profile's "
        "undeclared secret straight from os.environ"
    )


def test_profile_carrier_ownership_survives_terminal_composition(tmp_path, monkeypatch):
    """A transport wrapper cannot hide source ownership or create a target grant."""
    import json
    import subprocess
    import sys
    from agent import secret_scope
    from tools.env_passthrough import clear_env_passthrough
    from tools.environments.local import _make_run_env

    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / ".env").write_text("APPTAINERENV_SOURCE_ONLY=alpha\n", encoding="utf-8")
    (target / ".env").write_text("APPTAINERENV_TARGET_ONLY=beta\n", encoding="utf-8")
    (target / "config.yaml").write_text(
        "terminal:\n  env_passthrough: [APPTAINERENV_TARGET_ONLY]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    names = ["SOURCE_ONLY", "APPTAINERENV_SOURCE_ONLY",
             "SINGULARITYENV_APPTAINERENV_SOURCE_ONLY", "APPTAINERENV_TARGET_ONLY", "TARGET_ONLY"]
    for name in names:
        monkeypatch.setenv(name, "ambient-residue")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    scope_token = secret_scope.set_secret_scope(None)
    home_token = set_hermes_home_override(target)
    clear_env_passthrough()
    try:
        child = subprocess.run(
            [sys.executable, "-c", "import os,json,sys;print(json.dumps({k:os.getenv(k) for k in sys.argv[1:]}))", *names],
            env=_make_run_env({}), capture_output=True, text=True, encoding="utf-8",
            check=True, timeout=10)
    finally:
        clear_env_passthrough()
        reset_hermes_home_override(home_token)
        secret_scope.reset_secret_scope(scope_token)
    expected = dict.fromkeys(names)
    expected["APPTAINERENV_TARGET_ONLY"] = "beta"
    assert json.loads(child.stdout) == expected
