"""Setup-wizard regression tests for the empty-allowlist messaging.

The gateway is default-deny: an empty allowlist denies every sender unless the
operator explicitly sets ``{PLATFORM}_ALLOW_ALL_USERS`` or
``GATEWAY_ALLOW_ALL_USERS`` (``gateway/authz_mixin.py::_is_user_authorized``,
SECURITY.md §2.6 rule 2). The Matrix / Mattermost / Discord setup wizards used
to tell operators the opposite — "leave empty for open access" — so a user who
left the allowlist blank expecting open access actually got default-deny and
their own messages were rejected as unauthorized.

These tests pin the corrected behavior: the empty path is described as
default-deny, never saves an allowlist or allow-all env var on its own, and
points the operator at the explicit opt-in. ``interactive_setup`` lazy-imports
its helpers from ``hermes_cli.config`` (get_env_value / save_env_value) and
``hermes_cli.cli_output`` (prompt / prompt_yes_no / print_*), so we patch those
source modules — the same approach as ``test_slack_plugin_setup.py``.
"""
import importlib

import pytest

import hermes_cli.config as config_mod
import hermes_cli.cli_output as cli_output_mod

# (module path, env-var prefix, a representative allowlist value)
PLATFORMS = [
    pytest.param("plugins.platforms.matrix.adapter", "MATRIX", "@bot:server.org", id="matrix"),
    pytest.param("plugins.platforms.mattermost.adapter", "MATTERMOST", "user123", id="mattermost"),
    pytest.param("plugins.platforms.discord.adapter", "DISCORD", "123456789012345678", id="discord"),
]


def _drive_setup(monkeypatch, modpath, allowlist_input):
    """Run a platform's interactive_setup with mocked I/O.

    Returns (saved_env, prompts_shown, output_lines) where output_lines is the
    combined text emitted via print_info and print_warning during the security
    step.
    """
    saved = {}
    prompts = []
    output = []

    def fake_prompt(message="", *_a, **_kw):
        msg = str(message)
        prompts.append(msg)
        low = msg.lower()
        if "allowed user" in low or "allowed nick" in low:
            return allowlist_input
        # Supply the credentials needed to reach the security/allowlist step;
        # leave every other optional field (user id, home channel, password) blank.
        if "access token" in low or "homeserver" in low or "server url" in low or "bot token" in low:
            return "dummy"
        return ""

    monkeypatch.setattr(config_mod, "get_env_value", lambda key: "")
    monkeypatch.setattr(config_mod, "save_env_value", lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(cli_output_mod, "prompt", fake_prompt)
    monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: False)
    monkeypatch.setattr(cli_output_mod, "print_header", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_output_mod, "print_success", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        cli_output_mod,
        "print_info",
        lambda *a, **_kw: output.append(" ".join(str(x) for x in a)),
    )
    monkeypatch.setattr(
        cli_output_mod,
        "print_warning",
        lambda *a, **_kw: output.append(" ".join(str(x) for x in a)),
    )

    # Matrix's wizard probes optional E2EE deps and may shell out to install
    # them; force "nothing missing" so the test never touches the network.
    lazy_mod = importlib.import_module("tools.lazy_deps")
    monkeypatch.setattr(lazy_mod, "feature_missing", lambda *_a, **_kw: [])

    mod = importlib.import_module(modpath)
    mod.interactive_setup()
    return saved, prompts, output


@pytest.mark.parametrize("modpath,prefix,sample", PLATFORMS)
def test_empty_allowlist_is_default_deny_not_open_access(monkeypatch, modpath, prefix, sample):
    saved, prompts, output = _drive_setup(monkeypatch, modpath, "")

    allow_prompt = next((p for p in prompts if "allowed user" in p.lower()), None)
    assert allow_prompt is not None, "the allowlist prompt was never shown"
    assert "open access" not in allow_prompt.lower(), (
        f"{prefix} setup still promises 'open access' for an empty allowlist, "
        "but the gateway is default-deny"
    )

    # Leaving it blank must not silently grant access through either env var.
    assert f"{prefix}_ALLOWED_USERS" not in saved
    assert f"{prefix}_ALLOW_ALL_USERS" not in saved

    # The operator is told access is denied by default and how to opt into open access.
    joined = " ".join(output)
    assert "denied by default" in joined
    assert f"{prefix}_ALLOW_ALL_USERS=true" in joined
    assert "GATEWAY_ALLOW_ALL_USERS=true" in joined


@pytest.mark.parametrize("modpath,prefix,sample", PLATFORMS)
def test_allowlist_saved_when_provided(monkeypatch, modpath, prefix, sample):
    saved, _, _ = _drive_setup(monkeypatch, modpath, sample)
    assert saved.get(f"{prefix}_ALLOWED_USERS"), f"{prefix}_ALLOWED_USERS was not saved"
