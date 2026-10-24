"""Guidance contract for system-scope gateway verbs typed without root.

Regression: ``hermes gateway restart --system`` printed only
``System gateway restart requires root. Re-run with sudo.`` and stopped there. That advice cannot work —
``hermes`` is a per-user install whose shim lives in ``~/.local/bin``, and sudo resets PATH to
``secure_path`` — so the user's next command is ``sudo: hermes: command not found``. The entry point now
prints the remediation the setup wizard already printed: a recipe that runs as pasted.

``install`` is the verb used for the end-to-end cases: it reaches the root guard before touching anything,
whereas ``restart --system`` with no installed unit deliberately falls through to a detached run.
"""

from __future__ import annotations

import argparse

import pytest

from hermes_cli import gateway as gw
from hermes_cli import gateway_sudo_guidance as guidance

RENDERED_BINARY = "/opt/hermes/.local/bin/hermes"


@pytest.fixture
def non_root(monkeypatch):
    """A non-root caller whose CLI resolves to a user-local shim, like a pip/pipx install."""
    monkeypatch.setattr(gw.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(guidance, "_hermes_binary_for_sudo", lambda: RENDERED_BINARY)
    return monkeypatch


def _typed(verb: str) -> argparse.Namespace:
    return argparse.Namespace(gateway_command=verb, system=True)


def _assert_no_bare_sudo_hermes(text: str) -> None:
    assert "sudo hermes " not in text, "a bare `sudo hermes` cannot resolve under sudo's PATH"


def test_typed_system_install_prints_a_recipe_that_runs(non_root, capsys):
    non_root.setattr(gw.sys, "argv", [RENDERED_BINARY, "gateway", "install", "--system"])

    with pytest.raises(SystemExit) as exit_info:
        gw.gateway_command(_typed("install"))

    text = capsys.readouterr().out
    assert exit_info.value.code == 1
    assert f"{RENDERED_BINARY} gateway install --system" in text
    assert "systemctl install" not in text, "systemctl has no install verb"
    _assert_no_bare_sudo_hermes(text)


def test_remediation_reuses_the_typed_tail_after_global_flags(non_root, capsys):
    non_root.setattr(gw.sys, "argv", [RENDERED_BINARY, "--yolo", "gateway", "install", "--system"])

    with pytest.raises(SystemExit):
        gw.gateway_command(_typed("install"))

    text = capsys.readouterr().out
    assert f"{RENDERED_BINARY} gateway install --system" in text
    assert "--yolo" not in text, "flags before the subcommand are not part of the command to re-run"


def test_restart_remediation_names_the_supervisor_and_the_cli(non_root, capsys):
    gw._print_system_scope_remediation("restart")

    text = capsys.readouterr().out
    assert f"sudo systemctl restart {gw.get_service_name()}" in text
    assert f"{RENDERED_BINARY} gateway restart --system" in text
    _assert_no_bare_sudo_hermes(text)


def test_status_hint_constants_resolve_the_binary_at_print_time(non_root):
    rendered = gw._with_runnable_sudo(
        ("  sudo hermes gateway install --system  # Install as boot-time system service",)
    )

    assert rendered == (
        f"  sudo {RENDERED_BINARY} gateway install --system  # Install as boot-time system service",
    )


def test_binary_resolution_prefers_the_path_entry(monkeypatch):
    monkeypatch.setattr(
        guidance.shutil, "which", lambda name, *a, **kw: RENDERED_BINARY if name == "hermes" else None
    )

    assert guidance._hermes_binary_for_sudo() == RENDERED_BINARY


def test_binary_resolution_falls_back_to_the_script_that_was_invoked(monkeypatch, tmp_path):
    shim = tmp_path / "hermes"
    shim.write_text("#!/bin/sh\nexec /opt/hermes/venv/bin/hermes \"$@\"\n", encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setattr(guidance.shutil, "which", lambda name, *a, **kw: None)
    monkeypatch.setattr(guidance.sys, "argv", [str(shim), "gateway", "status"])

    assert guidance._hermes_binary_for_sudo() == str(shim)
