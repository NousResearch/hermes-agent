"""scripts/check_profile_scope_patterns.py flags a known profile-scope hazard and stays silent on clean code.

Advisory lint over the validated pattern set (``scripts/ci/profile_scope_patterns.json``): a child env
built from ``os.environ`` is the shape that leaked the launch profile's secrets into served-profile
children; the same spawn through ``served_profile_child_env`` is the fix and must not be flagged.
"""
import importlib.util
import sys
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_profile_scope_patterns.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_profile_scope_patterns", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules[cls.__module__] (3.11).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_child_env_from_environ_is_flagged_and_the_scoped_builder_is_not():
    mod = _load()
    patterns = mod.load_patterns()
    assert patterns and all(p["scope_hint"] and p["why"] for p in patterns)

    hazard = textwrap.dedent('''
        import os, subprocess

        def delivery_env(author):
            env = dict(os.environ)
            env["HERMES_TURN_AUTHOR"] = author
            return env

        def run(cmd):
            return subprocess.Popen(cmd, env=os.environ.copy())
    ''')
    clean = textwrap.dedent('''
        import subprocess
        from tools.environments.local import served_profile_child_env

        def delivery_env(author, profile_home):
            env = served_profile_child_env(target_home=profile_home, inherit_credentials=True)
            env["HERMES_TURN_AUTHOR"] = author
            return env

        def run(cmd, env):
            return subprocess.Popen(cmd, env=env)
    ''')
    flagged = mod.scan_text("tools/x.py", hazard, patterns)
    assert {(f.line, f.pattern_id) for f in flagged} >= {(5, "P05"), (10, "P05")}
    assert all(f.pattern_class == "C2" and f.why for f in flagged if f.pattern_id == "P05")
    assert mod.scan_text("tools/x.py", clean, patterns) == []

    # Diff mode only reports lines the change added: restricting to the untouched line hides the hit.
    assert mod.scan_text("tools/x.py", hazard, patterns, lines={6}) == []
    assert [f.line for f in mod.scan_text("tools/x.py", hazard, patterns, lines={5})] == [5]


def test_adapter_key_outside_the_seam_is_flagged_only_under_platforms():
    """An adapter that derives a session key with the free ``build_session_key()`` bypasses the
    owner-profile seam (``_source_session_key``); the same call in ``platforms/base.py`` (the seam
    itself) or in the runner is legitimate and must stay silent."""
    mod = _load()
    patterns = mod.load_patterns()
    free_key = textwrap.dedent('''
        from gateway.session import build_session_key

        def _batch_key(self, event):
            return build_session_key(event.source, profile=event.source.profile)
    ''')
    seam = textwrap.dedent('''
        def _batch_key(self, event):
            return self._event_session_key(event)
    ''')
    flagged = mod.scan_text("plugins/platforms/acme/adapter.py", free_key, patterns)
    assert [(f.line, f.pattern_id, f.pattern_class) for f in flagged] == [(5, "P32", "C4")]
    assert [f.pattern_id for f in mod.scan_text("gateway/platforms/acme.py", free_key, patterns)] == ["P32"]
    assert mod.scan_text("plugins/platforms/acme/adapter.py", seam, patterns) == []
    for owner in ("gateway/platforms/base.py", "gateway/run_startup.py", "gateway/session_recovery.py"):
        assert not [f for f in mod.scan_text(owner, free_key, patterns) if f.pattern_id == "P32"], owner


def test_lint_is_advisory_and_exits_zero_with_findings(tmp_path, capsys):
    mod = _load()
    bad = tmp_path / "bad.py"
    bad.write_text("import os\nenv = os.environ.copy()\n", encoding="utf-8")
    assert mod.main(["--files", str(bad)]) == 0
    out = capsys.readouterr().out
    assert "P05/C2" in out and ":2 " in out and "ADVISORY" in out
