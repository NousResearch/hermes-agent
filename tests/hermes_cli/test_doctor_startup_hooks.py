"""Doctor startup-hook scan — the LiteLLM 1.107.0 supply-chain IOC class.

Only files Python actually loads are flagged: ``*.pth`` and
``sitecustomize``/``usercustomize`` at the ROOT of a site-packages directory,
usercustomize at the site-packages root.
"""
import pytest

from hermes_cli import doctor_startup_hooks as dsh
from hermes_cli.doctor_startup_hooks import scan_startup_hooks


def _site(tmp_path):
    sd = tmp_path / "site-packages"
    sd.mkdir()
    return sd


def test_clean_site_packages(tmp_path):
    sd = _site(tmp_path)
    (sd / "_virtualenv.pth").write_text("import _virtualenv\n")
    (sd / "distutils-precedence.pth").write_text(
        "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = os.environ.get(var, 'local') == 'local'; "
        "enabled and __import__('_distutils_hack').add_shim(); "
    )
    # nested files inside an installed package are never auto-loaded by Python
    pkg = sd / "opentelemetry" / "instrumentation" / "auto_instrumentation"
    pkg.mkdir(parents=True)
    (pkg / "sitecustomize.py").write_text("x = 1\n")
    assert scan_startup_hooks([str(sd)]) == []


def test_flags_rogue_pth(tmp_path):
    sd = _site(tmp_path)
    (sd / "cook.pth").write_text("import credential_stealer\n")
    out = scan_startup_hooks([str(sd)])
    assert len(out) == 1
    assert out[0][0] == "suspicious .pth"
    assert "cook.pth" in out[0][1]


def test_flags_root_sitecustomize(tmp_path):
    sd = _site(tmp_path)
    (sd / "sitecustomize.py").write_text("import os\n")
    out = scan_startup_hooks([str(sd)])
    assert out and out[0][0] == "startup module"


def test_flags_any_unknown_executable_pth(tmp_path):
    # class behavior: ANY import not in the legitimate bootstrap set is flagged,
    # regardless of which package dropped it
    sd = _site(tmp_path)
    (sd / "innocuous-name.pth").write_text("import totally_unknown_package\n")
    out = scan_startup_hooks([str(sd)])
    assert out and out[0][0] == "suspicious .pth"


def test_flags_root_usercustomize(tmp_path):
    sd = _site(tmp_path)
    (sd / "usercustomize.py").write_text("import os\n")
    out = scan_startup_hooks([str(sd)])
    assert out and out[0][0] == "startup module"


def test_check_registered_in_doctor():
    from hermes_cli.doctor import DOCTOR_CHECKS
    names = [getattr(fn, "__name__", "") for _, fn in DOCTOR_CHECKS]
    assert "_check_startup_hooks" in names


def _rogue_site(tmp_path):
    sd = _site(tmp_path)
    (sd / "cook.pth").write_text("import credential_stealer\n")
    return sd


def test_gate_blocks_update_on_findings(tmp_path, capsys):
    sd = _rogue_site(tmp_path)
    with pytest.raises(SystemExit) as exc:
        dsh.gate_update_on_startup_hooks(site_dirs=[str(sd)], env={})
    assert exc.value.code == 2
    assert "refused" in capsys.readouterr().out.lower()


def test_gate_env_override_allows(tmp_path):
    sd = _rogue_site(tmp_path)
    dsh.gate_update_on_startup_hooks(site_dirs=[str(sd)], env={"HERMES_SKIP_STARTUP_HOOK_SCAN": "1"})


def test_gate_flag_override_allows(tmp_path):
    from types import SimpleNamespace
    sd = _rogue_site(tmp_path)
    dsh.gate_update_on_startup_hooks(
        SimpleNamespace(skip_startup_hook_scan=True), site_dirs=[str(sd)], env={})


def test_gate_clean_env_proceeds(tmp_path):
    sd = _site(tmp_path)
    (sd / "_virtualenv.pth").write_text("import _virtualenv\n")
    dsh.gate_update_on_startup_hooks(site_dirs=[str(sd)], env={})
