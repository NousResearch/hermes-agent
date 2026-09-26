"""`hermes pm install --extra NAME` is the one command every missing-extra hint names."""
import sys
from types import SimpleNamespace

import pytest

from pm import install_hint


def _wants_an_extra(monkeypatch, extras_mod, *, supported=True):
    """Put ensure_import on the path where the extra is missing but installable."""
    monkeypatch.setattr(extras_mod, "available", lambda extra: False)
    monkeypatch.setattr(extras_mod, "extra_supported", lambda extra, **kwargs: supported)
    # A prompt_toolkit app left in sys.modules by an earlier test would suppress the
    # question, which is the branch two of these tests are here to reach.
    monkeypatch.delitem(sys.modules, "prompt_toolkit.application.current", raising=False)


def test_hint_names_a_command_the_cli_accepts(monkeypatch):
    from pm import cli, runtime
    from pm import install as install_mod

    synced = []
    monkeypatch.setattr(runtime, "is_runtime", lambda: True)
    monkeypatch.setattr(install_mod, "sync_venv", lambda extras, **kwargs: synced.append((list(extras), kwargs)))
    monkeypatch.setattr(cli, "_install_names",
                        lambda names, target=None, **kwargs: 0 if not names else pytest.fail(f"tools installed: {names}"))
    monkeypatch.setattr(install_mod, "activate", lambda **kwargs: [])

    argv = install_hint("anthropic").split()[2:]
    assert cli.main(argv) == 0
    assert synced == [(["anthropic"], {"explicit": True})]


def test_extra_syncs_only_the_named_extras(monkeypatch):
    from pm import cli
    from pm import install as install_mod

    synced = []
    monkeypatch.setattr(install_mod, "sync_venv", lambda extras, **kwargs: synced.append(list(extras)))
    monkeypatch.setattr(install_mod, "activate", lambda **kwargs: [])
    monkeypatch.setattr(cli, "_install_names", lambda names, target=None, **kwargs: 0)
    assert cli.cmd_install(SimpleNamespace(names=[], extra=["otlp", "mcp", "otlp"], target=None, tools_only=False)) == 0
    assert synced == [["otlp", "mcp"]]


def test_cold_runtime_refusal_names_the_extra(monkeypatch, tmp_path):
    import pm.client as client
    from pm import receipt
    from pm.package import InstallError

    monkeypatch.setattr("pm.install.lazy_installs_allowed", lambda: False)
    monkeypatch.setattr(client, "runtime_environment", lambda: {})
    monkeypatch.setattr("pm.registry.package_definitions", lambda names: [])
    monkeypatch.setattr(client, "runtime_command", lambda *args, **kwargs: (_ for _ in ()).throw(
        InstallError("pm-runtime", "not installed or outdated and lazy installs are disabled")))
    for name in ("begin", "record_refusal", "record_step", "finalize"):
        monkeypatch.setattr(receipt, name, lambda *args, **kwargs: None)

    with pytest.raises(InstallError) as info:
        client._request("sync_venv", {"extras": ["bedrock"], "explicit": False, "repair": False},
                        project_root=tmp_path)
    assert install_hint("bedrock") in info.value.remedy
    assert "bedrock" in info.value.cause


def test_a_declined_extra_still_names_the_install_command(monkeypatch):
    """Answering "n" is the one failure where the install command IS the answer: the prompt
    does not come back on its own, so the error has to carry it."""
    from pm import extras
    from pm.package import InstallError

    _wants_an_extra(monkeypatch, extras)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    monkeypatch.setattr("pm.client.sync_venv",
                        lambda extras_: pytest.fail("declined, yet the venv was synced"))

    with pytest.raises(InstallError) as info:
        extras.ensure_import("bedrock")
    assert install_hint("bedrock") in info.value.remedy
    # str(InstallError) is what the provider adapters put in front of a user.
    assert install_hint("bedrock") in str(info.value)


def test_a_platform_gated_extra_does_not_promise_an_install(monkeypatch):
    """No install can succeed here, so neither the install command nor the generic
    "retry" default belongs in the remedy — both loop the user forever."""
    from pm import extras
    from pm.package import InstallError

    _wants_an_extra(monkeypatch, extras, supported=False)
    monkeypatch.setattr(extras, "_platform_gates", lambda: {"neutts": 'sys_platform != "win32"'})

    with pytest.raises(InstallError) as info:
        extras.ensure_import("neutts")
    assert install_hint("neutts") not in info.value.remedy
    assert "hermes pm doctor" not in info.value.remedy


def test_an_installed_extra_awaiting_restart_asks_for_a_restart(monkeypatch, tmp_path):
    """The extra is already installed. "Install it" and "retry" are both wrong."""
    from pm import extras
    from pm.package import InstallError

    _wants_an_extra(monkeypatch, extras)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("pm.client.sync_venv", lambda extras_: None)
    facts = tmp_path / "facts.json"
    facts.write_text("{}")
    monkeypatch.setattr("pm.paths.runtime_facts_path", lambda: facts)
    monkeypatch.setattr("pm.paths.repo_root", lambda: tmp_path)
    monkeypatch.setattr("pm.environments_adopt.adopt_selected", lambda root: None)
    monkeypatch.setattr("pm.environments_adopt.restart_needed", lambda root: "")
    monkeypatch.setattr("pm.environments.selected_venv", lambda root: tmp_path / "venv")
    monkeypatch.setattr("pm.environments.site_packages", lambda venv: tmp_path / "site-packages")

    with pytest.raises(InstallError) as info:
        extras.ensure_import("bedrock")
    assert "restart" in info.value.remedy.lower()
    assert install_hint("bedrock") not in info.value.remedy
    assert "hermes pm doctor" not in info.value.remedy
