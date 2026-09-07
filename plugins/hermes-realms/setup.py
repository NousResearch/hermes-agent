"""Explicit enable-time setup, invoked only by the host's consent lifecycle."""
from pathlib import Path
from dataclasses import asdict
import json
import os
import platform
import runpy


def _load(name):
    return runpy.run_path(str(Path(__file__).resolve().parent / "realms/_binding.py"))["load_runtime"](name)


def _revision(installer):
    return f"{installer.VERSION}:{installer.ARCHIVE_SHA256}:{installer.BINARY_SHA256}"


def _receipt_data(home, target, installer):
    config = _load("config")
    return {
        "profile": str(config.effective_home(home)),
        "revision": _revision(installer),
        "driver": installer._execution_identity(target),
        "config": asdict(config.Config.load(home)),
    }


def describe(hermes_home):
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        return {
            "revision": "unsupported-platform",
            "ready": False,
            "summary": "Realms setup requires Linux x86-64",
            "details": ["No verified Cua release is available for this host. Leave Realms disabled."],
        }
    installer = _load("install_driver")
    target = _load("config").driver_path(hermes_home)
    status = _load("integration").setup_status(driver_executable=target)
    completed = False
    if status["ready"] and not os.access(target.parent, os.W_OK | os.X_OK):
        status = {
            "ready": False,
            "message": "Realms setup cannot update its driver directory. Restore this profile's directory write/search permissions and retry setup.",
        }
    if status["ready"]:
        try:
            installer.profile_target(hermes_home)
            completed = json.loads(target.with_name(".realms-setup.json").read_text(encoding="utf-8")) == _receipt_data(hermes_home, target, installer)
        except (OSError, ValueError):
            completed = False
    return {
        "revision": _revision(installer),
        "ready": completed,
        "summary": f"Install and verify Cua driver {installer.VERSION} for Realms",
        "details": [
            installer.URL,
            f"Destination: {target}",
            f"Archive SHA-256: {installer.ARCHIVE_SHA256}",
            f"Binary SHA-256: {installer.BINARY_SHA256}",
            "Downloads the pinned release if needed, writes only this profile's driver, and executes --version to verify it.",
            "Does not install system packages, change the global Cua driver, start a desktop, or alter approvals.",
            status["message"],
        ],
    }


def run(hermes_home):
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise ValueError("Realms setup requires Linux x86-64; leave this plugin disabled on this host")
    installer = _load("install_driver")
    target = installer.profile_target(hermes_home)
    receipt = target.with_name(".realms-setup.json")
    receipt.unlink(missing_ok=True)
    installer.install(home=hermes_home)
    status = _load("integration").setup_status(driver_executable=target)
    if not status["ready"]:
        raise ValueError(status["message"])
    report = _load("manager").Manager(hermes_home).doctor()
    if not report["ok"]:
        raise ValueError("Realms setup failed: a working systemd user manager is required. Run hermes realms doctor in this profile; host fallback is disabled.")
    installer.write_receipt(receipt, _receipt_data(hermes_home, target, installer))
