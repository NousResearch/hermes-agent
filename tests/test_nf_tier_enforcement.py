"""North Forge tier / pinned-edition access control (CHG-2026-09-07-022, NF-v0.6.0).

An *edition* is a Hermes profile. A deployed drive carries a signed provisioning
record (``<nf-root>/north-forge/provisioning.json`` + ``.nf-key``) written once at
Setup Run. Two tiers, no third:

* **Full**  — the pinned edition is only a default landing profile; every switch
  path stays open.
* **Basic** — the pinned edition is the ONLY reachable one. ``-p``, a hand-edited
  ``active_profile``, ``hermes profile use``, the dashboard and ``/edition`` all
  refuse anything else, and ``HERMES_HOME`` never moves.

DECISION-2026-09-07-003. The enforcement lives at the profile-resolution layer
(``hermes_cli.main._apply_profile_override`` + backstops in
``hermes_cli.profiles``), not merely in the UI — a Basic session that names a
forbidden edition directly must *fail cleanly*, never silently succeed.

Layers (same split as ``test_nf_preflight_readiness.py``):
  * unit — ``hermes_cli.nf_tier`` in-process, every platform.
  * integration — a real ``python -c "import hermes_cli.main"`` subprocess with a
    crafted ``sys.argv`` / ``HERMES_HOME``, so the actual pre-argparse gate runs.
  * ``_WINDOWS_ONLY`` — drive ``scripts/nf-setup.ps1`` and confirm the PS 5.1
    stderr-abort trap does not bite and the passcode gate holds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="drives a .ps1 script")


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def drive(tmp_path):
    """A fake deployed drive: ``<tmp>/data`` is HERMES_HOME, with two editions."""
    data = tmp_path / "data"
    for name in ("penny-pincher", "kyocera", "sales-edition"):
        d = data / "profiles" / name
        d.mkdir(parents=True)
        (d / "config.yaml").write_text("model: test\n", encoding="utf-8")
    return data


def _nf():
    # imported per-call so tests that mutate the on-disk record see a fresh read
    from hermes_cli import nf_tier
    nf_tier.clear_cache()
    return nf_tier


def _provision(data: Path, *, tier: str, pin: str, installed=None, overwrite=True):
    t = _nf()
    return t.write_provisioning(tier=tier, pinned_edition=pin,
                                installed_editions=installed or [], root=data, overwrite=overwrite)


# --------------------------------------------------------------------------- unit


def test_unprovisioned_drive_is_inert(drive):
    t = _nf()
    p = t.load(drive)
    assert p.state == t.STATE_UNPROVISIONED
    assert not p.locked
    assert t.allowed_editions(drive) is None
    assert t.default_edition(drive) is None
    assert t.edition_allowed("anything", drive) is True
    t.assert_edition_allowed("anything", root=drive)  # no raise


def test_basic_locks_to_the_pin(drive):
    t = _nf()
    p = _provision(drive, tier="basic", pin="penny-pincher")
    assert p.state == t.STATE_ACTIVE and p.tier == "basic" and p.locked
    assert p.pinned_edition == "penny-pincher"
    assert p.installed_editions == ("penny-pincher",)  # basic records only its pin
    assert t.allowed_editions(drive) == {"penny-pincher"}
    assert t.default_edition(drive) == "penny-pincher"
    assert t.edition_allowed("penny-pincher", drive) is True
    assert t.edition_allowed("kyocera", drive) is False
    with pytest.raises(t.NfTierError):
        t.assert_edition_allowed("kyocera", root=drive)
    # NfTierError must be a ValueError so existing except-arms catch it
    assert issubclass(t.NfTierError, ValueError)


def test_full_pin_is_only_a_default(drive):
    t = _nf()
    p = _provision(drive, tier="full", pin="kyocera",
                   installed=["kyocera", "sales-edition", "pine-barron-farms"])
    assert p.tier == "full" and not p.locked
    assert t.allowed_editions(drive) is None            # no restriction
    assert t.default_edition(drive) == "kyocera"
    t.assert_edition_allowed("sales-edition", root=drive)   # no raise
    t.assert_edition_allowed("pine-barron-farms", root=drive)


def test_enforce_startup_profile_matrix(drive):
    t = _nf()

    _provision(drive, tier="basic", pin="penny-pincher")
    e = lambda req, flag: t.enforce_startup_profile(req, from_flag=flag, root=drive)  # noqa: E731
    assert e(None, False) == "penny-pincher"            # bare launch -> pin
    assert e("penny-pincher", True) == "penny-pincher"  # -p pin -> allowed
    assert e("kyocera", False) == "penny-pincher"       # stale active_profile -> ignored
    with pytest.raises(t.NfTierError):
        e("kyocera", True)                              # explicit -p other -> hard error

    _provision(drive, tier="full", pin="kyocera", installed=["kyocera", "sales-edition"])
    assert e(None, False) == "kyocera"                  # bare -> pinned default
    assert e("sales-edition", True) == "sales-edition"  # -p other -> honoured
    assert e("sales-edition", False) == "sales-edition"  # sticky honoured

    # basic drive pinned to the plain chassis
    _provision(drive, tier="basic", pin="")
    assert e(None, False) is None
    with pytest.raises(t.NfTierError):
        e("sales-edition", True)


def test_tamper_is_detected_and_fails_closed(drive):
    t = _nf()
    _provision(drive, tier="basic", pin="penny-pincher")
    rp = t.record_path(drive)
    rec = json.loads(rp.read_text())
    rec["tier"] = "full"                                # hand-flip Basic -> Full
    rp.write_text(json.dumps(rec, indent=2))
    t.clear_cache()

    p = t.load(drive)
    assert p.state == t.STATE_TAMPERED
    assert "signature" in p.error
    with pytest.raises(t.NfTierError):
        t.enforce_startup_profile(None, from_flag=False, root=drive)
    with pytest.raises(t.NfTierError):
        t.assert_edition_allowed("penny-pincher", root=drive)


def test_missing_key_is_tamper(drive):
    t = _nf()
    _provision(drive, tier="basic", pin="penny-pincher")
    t.key_path(drive).unlink()
    t.clear_cache()
    assert t.load(drive).state == t.STATE_TAMPERED


def test_admin_passcode_gate(drive):
    t = _nf()
    assert t.verify_admin_passcode("whatever", drive) is True   # none set yet
    t.set_admin_passcode("forge-master-9", drive)
    assert t.admin_passcode_is_set(drive) is True
    assert t.verify_admin_passcode("forge-master-9", drive) is True
    assert t.verify_admin_passcode("wrong", drive) is False
    with pytest.raises(t.NfTierError):
        t.set_admin_passcode("another", drive)                  # no clobber without rotate
    t.set_admin_passcode("second-pass-7", drive, rotate=True)
    assert t.verify_admin_passcode("second-pass-7", drive) is True
    assert t.verify_admin_passcode("forge-master-9", drive) is False


def test_cli_provision_show_verify(drive):
    from hermes_cli import nf_tier
    rc = nf_tier.main(["--root", str(drive), "provision", "--tier", "basic", "--pin", "penny-pincher"])
    assert rc == 0
    nf_tier.clear_cache()
    assert nf_tier.main(["--root", str(drive), "verify"]) == 0
    # tamper -> verify exits 2
    rp = nf_tier.record_path(drive)
    rec = json.loads(rp.read_text()); rec["pinned_edition"] = "kyocera"; rp.write_text(json.dumps(rec))
    nf_tier.clear_cache()
    assert nf_tier.main(["--root", str(drive), "verify"]) == 2


# ------------------------------------------------------------------- integration


def _run_hermes(argv_tail, home, extra_env=None):
    """`python -c "import hermes_cli.main"` with a crafted argv — the real gate runs."""
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO_ROOT)}
    env.pop("HERMES_PROFILE", None)
    env.update(extra_env or {})
    code = textwrap.dedent(f"""
        import sys, os
        sys.argv = ['hermes', *{argv_tail!r}]
        import hermes_cli.main
        print('RESOLVED_HOME=' + os.environ.get('HERMES_HOME', ''))
    """)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def _resolved(res):
    for ln in res.stdout.splitlines():
        if ln.startswith("RESOLVED_HOME="):
            return ln.split("=", 1)[1]
    return ""


def test_integration_basic_blocks_flag_and_forces_pin(drive):
    _provision(drive, tier="basic", pin="penny-pincher")

    r = _run_hermes(["-p", "kyocera", "chat"], drive)
    assert r.returncode == 1, r.stderr
    assert "kyocera" in r.stderr and "penny-pincher" in r.stderr
    assert "kyocera" not in _resolved(r)                       # HERMES_HOME never moved

    r = _run_hermes(["-p", "penny-pincher", "chat"], drive)
    assert r.returncode == 0 and _resolved(r).replace("\\", "/").endswith("profiles/penny-pincher")

    r = _run_hermes(["chat"], drive)                           # bare -> pin
    assert r.returncode == 0 and _resolved(r).replace("\\", "/").endswith("profiles/penny-pincher")


def test_integration_basic_ignores_stale_active_profile(drive):
    _provision(drive, tier="basic", pin="penny-pincher")
    (drive / "active_profile").write_text("kyocera\n", encoding="utf-8")
    r = _run_hermes(["chat"], drive)
    assert r.returncode == 0
    assert _resolved(r).replace("\\", "/").endswith("profiles/penny-pincher")


def test_integration_basic_blocks_hermes_home_pointed_at_other_edition(drive):
    _provision(drive, tier="basic", pin="penny-pincher")
    r = _run_hermes(["chat"], drive / "profiles" / "sales-edition")
    assert r.returncode == 1
    assert "sales-edition" in r.stderr


def test_integration_tampered_record_refuses_to_start(drive):
    _provision(drive, tier="basic", pin="penny-pincher")
    rp = (_nf()).record_path(drive)
    rec = json.loads(rp.read_text()); rec["tier"] = "full"; rp.write_text(json.dumps(rec))
    r = _run_hermes(["chat"], drive)
    assert r.returncode == 1
    assert "invalid" in r.stderr or "modified" in r.stderr


def test_integration_full_defaults_to_pin_but_switches(drive):
    _provision(drive, tier="full", pin="kyocera", installed=["kyocera", "sales-edition"])

    r = _run_hermes(["chat"], drive)
    assert _resolved(r).replace("\\", "/").endswith("profiles/kyocera")

    r = _run_hermes(["-p", "sales-edition", "chat"], drive)
    assert r.returncode == 0 and _resolved(r).replace("\\", "/").endswith("profiles/sales-edition")

    (drive / "active_profile").write_text("sales-edition\n", encoding="utf-8")
    r = _run_hermes(["chat"], drive)
    assert _resolved(r).replace("\\", "/").endswith("profiles/sales-edition")


def test_integration_unprovisioned_drive_unchanged(drive):
    r = _run_hermes(["-p", "kyocera", "chat"], drive)          # no provisioning record
    assert r.returncode == 0
    assert _resolved(r).replace("\\", "/").endswith("profiles/kyocera")


# --------------------------------------------------------------- backstop gates


def _pyrun(snippet, home):
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO_ROOT)}
    env.pop("HERMES_PROFILE", None)
    return subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env)


def test_backstop_resolve_profile_env(drive):
    _provision(drive, tier="basic", pin="penny-pincher")
    r = _pyrun("import hermes_cli.profiles as p; print(p.resolve_profile_env('kyocera'))", drive)
    assert r.returncode != 0 and "Basic" in (r.stderr + r.stdout)
    r = _pyrun("import hermes_cli.profiles as p; print(p.resolve_profile_env('penny-pincher'))", drive)
    assert r.returncode == 0 and r.stdout.strip().replace("\\", "/").endswith("profiles/penny-pincher")


def test_backstop_set_active_profile(drive):
    _provision(drive, tier="basic", pin="penny-pincher")
    r = _pyrun("import hermes_cli.profiles as p; p.set_active_profile('kyocera')", drive)
    assert r.returncode != 0 and "Basic" in (r.stderr + r.stdout)
    r = _pyrun("import hermes_cli.profiles as p; p.set_active_profile('default')", drive)
    assert r.returncode != 0, "Basic drive must not be able to drop to the root profile"
    r = _pyrun("import hermes_cli.profiles as p; p.set_active_profile('penny-pincher'); print('OK')", drive)
    assert r.returncode == 0 and "OK" in r.stdout


def test_backstop_profile_cmd_mutations_blocked_on_basic(drive):
    _provision(drive, tier="basic", pin="penny-pincher")
    for verb in ("create", "delete", "import", "install", "rename", "alias"):
        r = _pyrun(
            "import sys; sys.argv=['hermes']; "
            f"from hermes_cli.profile_cmd import _nf_guard_mutation; _nf_guard_mutation({verb!r})",
            drive)
        assert r.returncode != 0 and "Basic" in r.stdout, verb


def test_edition_executor(drive):
    _provision(drive, tier="basic", pin="penny-pincher")
    r = _pyrun(
        "from hermes_cli.commands import COMMAND_REGISTRY;"
        "from hermes_cli.slash_exec import run_execute, CommandContext;"
        "cd=[c for c in COMMAND_REGISTRY if c.name=='edition'][0];"
        "print(run_execute(cd, CommandContext(args='')).text)", drive)
    assert r.returncode == 0
    assert "Basic" in r.stdout and "penny-pincher" in r.stdout and "no switcher" in r.stdout.lower()


# ------------------------------------------------------------------- nf-setup.ps1


@_WINDOWS_ONLY
def test_nf_setup_ps1_provisions_and_survives_stderr(drive, tmp_path):
    script = REPO_ROOT / "scripts" / "nf-setup.ps1"
    common = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
              "-RepoRoot", str(REPO_ROOT), "-DataDir", str(drive), "-NonInteractive"]

    r = subprocess.run(common + ["-Tier", "basic", "-Pin", "penny-pincher"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((drive / "north-forge" / "provisioning.json").read_text())
    assert rec["tier"] == "basic" and rec["pinned_edition"] == "penny-pincher" and rec["sig"]

    # re-provision without -Force is refused
    r = subprocess.run(common + ["-Tier", "full", "-Pin", "kyocera"], capture_output=True, text=True)
    assert r.returncode == 1

    # set a passcode, then a wrong one is rejected and the right one works
    r = subprocess.run(common + ["-SetPasscode", "-Passcode", "forge-master-9"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run(common + ["-Tier", "full", "-Pin", "kyocera", "-Force", "-Passcode", "nope123"],
                       capture_output=True, text=True)
    assert r.returncode == 1
    r = subprocess.run(common + ["-Tier", "full", "-Pin", "kyocera", "-Installed", "kyocera",
                                 "-Force", "-Passcode", "forge-master-9"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((drive / "north-forge" / "provisioning.json").read_text())["tier"] == "full"
