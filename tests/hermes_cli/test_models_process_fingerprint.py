"""An external-process provider's cached /model catalog survives vendor token refreshes.

Regression for #129094: Claude Code rewrites ``~/.claude/.credentials.json`` on every OAuth
refresh, and that file was folded into every provider's catalog fingerprint, so the DirectSDK
picker kept dropping its live list. A process provider is keyed on its own declared files only.
"""

from __future__ import annotations

import pytest

import providers as _providers
from providers.base import ProviderProfile


@pytest.fixture
def process_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _providers._discover_providers()
    monkeypatch.setattr(_providers, "_REGISTRY", dict(_providers._REGISTRY))
    monkeypatch.setattr(_providers, "_ALIASES", dict(_providers._ALIASES))
    monkeypatch.setattr(_providers, "_PROVIDER_LIST_CACHE", None)
    _providers.register_provider(ProviderProfile(
        name="fp-process", auth_type="external_process", base_url="process://fp-process",
        process_command="fp-process-cli-not-installed", catalog_credential_files=("~/.fp-process/account.json",),
    ))
    return tmp_path


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_vendor_token_rewrite_keeps_the_process_catalog_fingerprint(process_provider):
    from hermes_cli.models import _credential_fingerprint

    _write(process_provider / ".claude" / ".credentials.json", '{"accessToken": "a"}')
    before = _credential_fingerprint("fp-process")
    _write(process_provider / ".claude" / ".credentials.json", '{"accessToken": "b"}')

    assert _credential_fingerprint("fp-process") == before


def test_the_profiles_own_credential_file_still_invalidates(process_provider):
    from hermes_cli.models import _credential_fingerprint

    _write(process_provider / ".fp-process" / "account.json", '{"account": "one"}')
    before = _credential_fingerprint("fp-process")
    _write(process_provider / ".fp-process" / "account.json", '{"account": "two"}')

    assert _credential_fingerprint("fp-process") != before


@pytest.mark.platforms("posix")
def test_retargeting_an_override_symlink_invalidates(process_provider, monkeypatch):
    # The override text stays the same while an upgrade repoints the symlink it names.
    from hermes_cli.models import _credential_fingerprint

    monkeypatch.setattr(_providers._REGISTRY["fp-process"], "process_command_env_vars", ("FP_PROCESS_COMMAND",))
    for version in ("1", "2"):
        _write(process_provider / "versions" / version / "fp", "#!/bin/sh\n")
        (process_provider / "versions" / version / "fp").chmod(0o755)
    link = process_provider / "bin" / "fp"
    link.parent.mkdir()
    link.symlink_to(process_provider / "versions" / "1" / "fp")
    monkeypatch.setenv("FP_PROCESS_COMMAND", str(link))
    before = _credential_fingerprint("fp-process")
    link.unlink()
    link.symlink_to(process_provider / "versions" / "2" / "fp")

    assert _credential_fingerprint("fp-process") != before
