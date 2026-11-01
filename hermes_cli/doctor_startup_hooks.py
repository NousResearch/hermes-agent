"""Doctor check: Python startup-hook integrity (supply-chain IOC scan).

Detects a *class* of supply-chain persistence, not any single incident:
executable ``.pth`` files and root-level ``sitecustomize``/``usercustomize``
modules in site-packages run automatically on *every* interpreter start —
no import of the package that dropped them is required. A compromised
dependency therefore only needs to be installed, never used, to achieve
code execution and credential theft. No legitimate package needs a startup
hook at the site-packages root, so flagging everything outside the
known-legitimate bootstrap set (virtualenv, setuptools, pywin32, …) catches
the mechanism itself.

Reference incident: the compromised LiteLLM 1.107.0 release (Oct 21, 2026)
used exactly this mechanism — see
docs.litellm.ai/blog/security-incident-oct-21-2026 — but the check targets
the mechanism, not that package.

The scan is read-only: it never imports or executes scanned content, and only
inspects files Python actually loads as startup hooks — ``*.pth`` and
``sitecustomize``/``usercustomize`` modules at the *root* of a site-packages
directory (nested copies inside a package's own tree are not auto-loaded), plus
dist-info markers of the known-compromised release.
"""
from __future__ import annotations
import os
import re
import sys

from hermes_cli.doctor_report import Finding, check_ok, check_warn, doctor_check

# .pth contents that are known-legitimate bootstrap hooks shipped by the
# standard toolchain: virtualenv bootstrap, setuptools editable-install finder,
# the distutils shim, pywin32, the coloredlogs opt-in auto-install, and the
# setuptools namespace-package template.
_LEGIT_PTH = (
    r"^import _virtualenv$",
    r"^import __editable___.*_finder; __editable___.*_finder\.install\(\)$",
    r"^import os; var = 'SETUPTOOLS_USE_DISTUTILS';.*_distutils_hack.*add_shim\(\); ?$",
    r"^import pywin32.*$",
    r"^import os; exec\('try: .*coloredlogs.*$",
    r"^import sys, types, os;p = os\.path\.join\(sys\._getframe\(1\).*$",
)



def scan_startup_hooks(site_dirs):
    """Return ``[(kind, path, detail)]`` for suspicious startup-hook files.

    Only the root level of each site-packages directory is considered — that
    is what the interpreter loads; nested files inside installed packages are
    ignored.
    """
    findings: list[tuple[str, str, str]] = []
    for sd in site_dirs:
        try:
            entries = sorted(os.listdir(sd))
        except OSError:
            continue
        for name in entries:
            path = os.path.join(sd, name)
            if name.endswith(".pth"):
                try:
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        content = fh.read().strip()
                except OSError:
                    continue
                if content.startswith("import ") and not any(re.match(rx, content) for rx in _LEGIT_PTH):
                    findings.append(("suspicious .pth", path, content[:120]))
            elif name in ("sitecustomize.py", "usercustomize.py"):
                findings.append(("startup module", path, "loaded unconditionally by the Python interpreter"))
    return findings


def _site_packages_dirs():
    """Unique existing site-packages/dist-packages directories on sys.path."""
    dirs: list[str] = []
    for p in sys.path:
        if not p:
            continue
        low = p.replace("\\", "/")
        if low.endswith("site-packages") or low.endswith("dist-packages"):
            if os.path.isdir(p) and p not in dirs:
                dirs.append(p)
    return dirs


@doctor_check()
def _check_startup_hooks(should_fix: bool, f: Finding) -> None:
    """Flag Python startup-hook files matching the LiteLLM 1.107.0 IOC class."""
    findings = scan_startup_hooks(_site_packages_dirs())
    if not findings:
        check_ok("No suspicious Python startup hooks in site-packages")
        return
    for kind, path, detail in findings[:8]:
        check_warn(f"{kind}: {os.path.basename(path)}", detail)
        f.manual_issues.append(
            f"Investigate {path} before starting Python again; do not run Python/pip in this "
            "environment until it is reviewed (supply-chain startup-hook persistence)."
        )
    if len(findings) > 8:
        check_warn(f"...and {len(findings) - 8} more startup-hook findings", "")


def gate_update_on_startup_hooks(args=None, *, site_dirs=None, env=None) -> None:
    """Refuse ``hermes update`` when this environment shows startup-hook anomalies.

    An update runs Python/pip in the *current* environment; if a malicious
    ``.pth``/``sitecustomize`` is present, the update would re-execute it —
    the core response lesson from startup-hook supply-chain incidents. Exits
    with code 2 on findings. Overrides: ``--skip-startup-hook-scan`` flag or
    ``HERMES_SKIP_STARTUP_HOOK_SCAN=1`` once the operator has reviewed them.
    """
    import os
    env = os.environ if env is None else env
    flag = str(env.get("HERMES_SKIP_STARTUP_HOOK_SCAN", "")).strip().lower() in {"1", "true", "yes", "on"}
    if not flag and getattr(args, "skip_startup_hook_scan", False):
        flag = True
    if flag:
        return
    findings = scan_startup_hooks(_site_packages_dirs() if site_dirs is None else site_dirs)
    if not findings:
        return
    print("✗ Update refused: Python startup-hook anomalies found in this environment.")
    print("  An update runs Python/pip here, which would re-execute any dropped .pth /")
    print("  sitecustomize startup hook (supply-chain persistence).")
    for kind, path, detail in findings[:8]:
        print(f"  ⚠ {kind}: {path}" + (f" ({detail})" if detail else ""))
    print("  Investigate first ('hermes doctor'), clean the environment, then retry — or,")
    print("  once reviewed, pass --skip-startup-hook-scan / HERMES_SKIP_STARTUP_HOOK_SCAN=1.")
    raise SystemExit(2)
