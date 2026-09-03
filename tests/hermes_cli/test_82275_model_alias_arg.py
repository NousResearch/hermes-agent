"""Regression tests for `hermes -m <alias>` resolving config aliases (#82275).

`hermes -m <alias>` was forwarding the raw alias string to the provider
API (404 error), while `hermes -q` (oneshot) and the in-session `/model`
command both resolve `model_aliases:` / `model.aliases:` entries from
config.yaml. `_resolve_model_alias_arg` must apply the same resolution to
the `--model` flag before the interactive agent is built.
"""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import main as cli_main
from hermes_cli import model_switch as ms


def _make_args(model=None, provider=None, base_url=None):
    return SimpleNamespace(model=model, provider=provider, base_url=base_url)


def _install_alias(monkeypatch, alias):
    monkeypatch.setattr(ms, "DIRECT_ALIASES", {"glm": alias})
    monkeypatch.setattr(ms, "_ensure_direct_aliases", lambda: None)


def test_alias_resolved(monkeypatch):
    alias = ms.DirectAlias(
        model="z-ai/glm-5.2",
        provider="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
    )
    _install_alias(monkeypatch, alias)

    args = _make_args(model="glm")
    cli_main._resolve_model_alias_arg(args)

    assert args.model == "z-ai/glm-5.2"
    assert args.provider == "nvidia"
    assert args.base_url == "https://integrate.api.nvidia.com/v1"


def test_alias_case_insensitive(monkeypatch):
    alias = ms.DirectAlias(model="z-ai/glm-5.2", provider="nvidia", base_url="")
    _install_alias(monkeypatch, alias)

    args = _make_args(model="GLM")
    cli_main._resolve_model_alias_arg(args)

    assert args.model == "z-ai/glm-5.2"
    assert args.provider == "nvidia"


def test_explicit_provider_and_base_url_win(monkeypatch):
    alias = ms.DirectAlias(
        model="z-ai/glm-5.2",
        provider="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
    )
    _install_alias(monkeypatch, alias)

    args = _make_args(
        model="glm", provider="anthropic", base_url="https://custom.example/v1"
    )
    cli_main._resolve_model_alias_arg(args)

    # The alias resolves the model id, but an explicit --provider /
    # --base-url on the command line keeps winning.
    assert args.model == "z-ai/glm-5.2"
    assert args.provider == "anthropic"
    assert args.base_url == "https://custom.example/v1"


def test_non_alias_model_untouched(monkeypatch):
    alias = ms.DirectAlias(model="z-ai/glm-5.2", provider="nvidia", base_url="")
    _install_alias(monkeypatch, alias)

    args = _make_args(model="anthropic/claude-sonnet-4")
    cli_main._resolve_model_alias_arg(args)

    assert args.model == "anthropic/claude-sonnet-4"
    assert args.provider is None
    assert args.base_url is None


def test_empty_model_noop(monkeypatch):
    args = _make_args(model="")
    cli_main._resolve_model_alias_arg(args)
    assert args.model == ""
    assert args.provider is None
