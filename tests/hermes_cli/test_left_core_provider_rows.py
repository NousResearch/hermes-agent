"""``LEFT_CORE`` rows for a model provider that moved into a catalog plugin.

Real config/.env/auth.json/jobs.json files under a temp home. The generic seams (``providers``,
``profile_env``, ``notice``) run through a fake row; the GitHub Copilot row's predicate runs through its
own function, since which config slots count as "this home uses it" is that row's contract.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm

ACME = lcm.LeftCoreFeature(
    plugin="acme-llm", label="Acme LLM", unchanged="your ACME_KEY is unchanged",
    in_use=lambda home, **kw: "acme" in lcm.providers_selected(home, ("acme", "acme-acp"), **kw),
    providers=("acme", "acme-acp"), profile_env=("ACME_CLI_PATH",),
)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lcm, "_attempted", set())
    for name in ("HERMES_INFERENCE_PROVIDER", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "ACME_CLI_PATH"):
        monkeypatch.delenv(name, raising=False)
    import hermes_cli.memory_provider_migration as mpm
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)  # never the network


def _home(tmp_path: Path, *, env: str = "", config: str = "", auth: dict | None = None,
          jobs: list | None = None) -> Path:
    home = tmp_path / "home"
    home.mkdir(parents=True)
    if env:
        (home / ".env").write_text(env, encoding="utf-8")
    if config:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    if auth is not None:
        (home / "auth.json").write_text(json.dumps(auth), encoding="utf-8")
    if jobs is not None:
        (home / "cron").mkdir()
        (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    return home


@pytest.mark.parametrize(("config", "env", "auth", "jobs"), [
    ("model:\n  provider: copilot\n  default: gpt-5.4\n", "", None, None),
    ("model:\n  provider: GitHub-Copilot\n", "", None, None),
    ("model:\n  provider: copilot-acp\n", "", None, None),
    ("fallback_providers:\n  - provider: copilot\n    model: gpt-5.4\n", "", None, None),
    ("fallback_model:\n  provider: github-models\n  model: gpt-4.1\n", "", None, None),
    ("auxiliary:\n  vision:\n    provider: copilot\n", "", None, None),
    ("delegation:\n  provider: copilot-acp\n", "", None, None),
    ("delegation:\n  fallback_providers:\n    - provider: copilot\n      model: gpt-5.4\n", "", None, None),
    ("moa:\n  presets:\n    mine:\n      reference_models:\n        - provider: copilot\n          model: gpt-5.4\n"
     "      aggregator:\n        provider: nous\n        model: hermes-4\n", "", None, None),
    ("", "HERMES_INFERENCE_PROVIDER=copilot\n", None, None),
    ("", "", {"active_provider": "copilot"}, None),
    ("", "", None, [{"id": "j1", "provider": "copilot", "model": "gpt-5.4"}]),
    ("cron:\n  model: gpt-5.4\n  model_provider: copilot\n", "", None, None),
    ("delegation:\n  backend: model_map\n  backends:\n    model_map:\n      options:\n        - name: fast\n"
     "          provider: copilot\n          model: gpt-5.4\n", "", None, None),
    ("delegation:\n  backends:\n    model_map:\n      options:\n        - name: deep\n          provider: nous\n"
     "          fallback_providers:\n            - provider: github-copilot\n              model: gpt-5.4\n",
     "", None, None),
])
def test_copilot_in_use_is_any_provider_slot_naming_it(tmp_path, config, env, auth, jobs):
    assert lcm.copilot_in_use(_home(tmp_path, config=config, env=env, auth=auth, jobs=jobs)) is True


@pytest.mark.parametrize(("config", "env", "auth"), [
    ("model:\n  provider: nous\n", "GH_TOKEN=gho_x\nCOPILOT_GITHUB_TOKEN=gho_y\n", None),
    ("model:\n  provider: openrouter\n  default: github/copilot-like\n", "", None),
    ("", "", {"credential_pool": {"copilot": [{"source": "gh_cli", "access_token": "t"}]}}),
    ("model: copilot\n", "", None),  # a bare model string names a model, not a provider
    # the user's own endpoint under a name that is also a Copilot alias: the resolver serves it as custom
    ("model:\n  provider: github\ncustom_providers:\n  - name: github\n"
     "    base_url: https://llm.example.test/v1\n", "", None),
    ("model:\n  provider: github-models\nproviders:\n  github-models:\n"
     "    base_url: https://llm.example.test/v1\n", "", None),
])
def test_copilot_credentials_alone_are_not_use(tmp_path, config, env, auth):
    assert lcm.copilot_in_use(_home(tmp_path, config=config, env=env, auth=auth)) is False


def test_process_env_provider_counts_only_for_the_launch_home(tmp_path, monkeypatch):
    home = _home(tmp_path, config="model:\n  provider: nous\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "copilot")
    assert lcm.copilot_in_use(home) is False
    assert lcm.copilot_in_use(home, process_env=True) is True


_GH_ONLY = {"credential_pool": {"copilot": [{"source": "gh_cli", "label": "gh auth token", "access_token": "t"}]}}


@pytest.mark.parametrize(("config", "env", "auth", "notice"), [
    ("model:\n  provider: copilot\n", "", _GH_ONLY, True),
    ("model:\n  provider: copilot\n", "", None, True),
    ("model:\n  provider: copilot\n", "GITHUB_TOKEN=gho_x\n", _GH_ONLY, False),
    ("model:\n  provider: copilot\n", "", {"credential_pool": {"copilot": [{"source": "manual"}]}}, False),
    ("model:\n  provider: copilot\n", "", {"credential_pool": {"copilot": [{"source": "manual:device_code"}]}},
     False),
    ("model:\n  provider: copilot-acp\n", "", None, False),  # the Copilot CLI keeps its own login
    ("model:\n  provider: nous\n", "", _GH_ONLY, False),
])
def test_relogin_notice_only_for_a_copilot_home_without_a_token_the_plugin_reads(tmp_path, config, env, auth,
                                                                                  notice):
    text = lcm.copilot_relogin_notice(_home(tmp_path, config=config, env=env, auth=auth))
    assert ("hermes auth add copilot" in text) is notice


def test_gh_only_home_is_told_to_log_in_on_every_migration_outcome(tmp_path):
    home = _home(tmp_path, config="model:\n  provider: copilot\n", auth=_GH_ONLY)
    said = []

    def install(name):
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    assert lcm.migrate_home(home, install=install, say=said.append) == ["github-copilot"]
    assert "✓ GitHub Copilot moved out of core" in said[0] and "run `hermes auth add copilot`" in said[0]
    other = _home(tmp_path / "b", config="model:\n  provider: copilot\n", auth=_GH_ONLY)
    said.clear()
    lcm.migrate_home(other, install=lambda name: {"ok": False, "error": "network down"}, say=said.append)
    assert "could not be installed automatically" in said[0] and "run `hermes auth add copilot`" in said[0]


def test_update_reads_the_launch_homes_exported_token_for_the_relogin_notice(tmp_path, monkeypatch):
    """`hermes update` from a shell that exports GH_TOKEN: the launch home's agent will read it, so no
    "sign in again" notice; a sibling profile never sees the launch shell's env."""
    import pm.plugins_state as ps
    launch, other = (_home(tmp_path / name, config="model:\n  provider: copilot\n").rename(tmp_path / name / name)
                     for name in ("launch", "other"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("GH_TOKEN", "gho_exported")
    monkeypatch.setattr(ps, "dependency_homes", lambda: [launch, other])
    monkeypatch.setattr(lcm, "_home_consent", lambda home: True)

    def install_into(home, **kw):
        return lambda name: (home / "plugins" / name).mkdir(parents=True) or {"ok": True}
    monkeypatch.setattr(lcm, "_install_into", install_into)
    said = []
    lcm.migrate_all_homes(say=said.append)
    by_home = {h: " ".join(m for m in said if lcm._home_label(h) in m) for h in (launch, other)}
    assert "✓ GitHub Copilot moved out of core" in by_home[launch]
    assert "run `hermes auth add copilot`" not in by_home[launch]
    assert "run `hermes auth add copilot`" in by_home[other]


# ── generic seams through a fake row ─────────────────────────────────────────────────────────────


@pytest.fixture
def acme_row(monkeypatch, tmp_path):
    monkeypatch.setattr(lcm, "LEFT_CORE", (ACME,))
    home = _home(tmp_path, config="model:\n  provider: acme\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    import providers
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: None)
    return home


def test_unknown_left_core_provider_names_the_install_on_every_surface(acme_row):
    from hermes_cli.auth import AuthError, resolve_provider
    from hermes_cli.auth_commands import _unknown_provider_exit
    from hermes_cli.model_switch import _unknown_provider_message

    with pytest.raises(AuthError) as err:
        resolve_provider("acme-acp")
    for text in (str(err.value), str(_unknown_provider_exit("acme")), _unknown_provider_message("ACME")):
        assert "Acme LLM moved out of core into the 'acme-llm' plugin" in text
        assert "hermes plugins install acme-llm" in text
    (acme_row / "plugins" / "acme-llm").mkdir(parents=True)
    (acme_row / "config.yaml").write_text("model:\n  provider: acme\nplugins:\n  disabled: [acme-llm]\n",
                                          encoding="utf-8")
    assert "hermes plugins enable acme-llm" in lcm.provider_install_hint("acme")  # installed but disabled
    assert lcm.provider_install_hint("other") == ""


@pytest.mark.parametrize("listed", ["", "plugins:\n  enabled: [acme-llm]\n"])
def test_a_provider_plugin_that_failed_to_load_gets_its_error_not_enable(acme_row, listed):
    """A model-provider plugin loads without `plugins.enabled`: installed, not disabled and no profile means
    its import failed. The hint carries the error providers discovery recorded for this home."""
    plugin = acme_row / "plugins" / "acme-llm"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: acme-llm\nkind: model-provider\n", encoding="utf-8")
    (plugin / "__init__.py").write_text("from acme_sdk import client\n", encoding="utf-8")
    (acme_row / "config.yaml").write_text("model:\n  provider: acme\n" + listed, encoding="utf-8")
    hint = lcm.provider_install_hint("acme")
    assert "ModuleNotFoundError: No module named 'acme_sdk'" in hint and "hermes plugins update acme-llm" in hint
    assert "enable" not in hint


def test_runtime_resolution_installs_the_plugin_first_for_a_home_using_it(acme_row, monkeypatch):
    calls = []
    monkeypatch.setattr(lcm, "recover_at_startup", lambda **kw: calls.append(kw) or [])
    lcm.recover_for_provider("acme")
    assert len(calls) == 1
    lcm.recover_for_provider("openrouter")
    (acme_row / "config.yaml").write_text("model:\n  provider: nous\n", encoding="utf-8")
    lcm.recover_for_provider("acme")  # requested, but this home never chose it: no install
    assert len(calls) == 1
    import providers
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: object())
    (acme_row / "config.yaml").write_text("model:\n  provider: acme\n", encoding="utf-8")
    lcm.recover_for_provider("acme")  # another plugin already serves it
    assert len(calls) == 1


@pytest.mark.parametrize("plugin_dir", [False, True])  # absent, or installed but not loaded (disabled/failed)
def test_launch_env_is_refused_by_env_writers_while_no_profile_declares_it(acme_row, monkeypatch, plugin_dir):
    from hermes_cli.config import validate_env_var_name_for_write
    import providers
    monkeypatch.setattr(lcm, "LEFT_CORE", (replace(ACME, profile_env=("ACME_CLI_PATH", "ACME_BASE_URL"),
                                                   launch_env=("ACME_CLI_PATH",)),))
    monkeypatch.setattr(providers, "list_providers", list)
    if plugin_dir:
        (acme_row / "plugins" / "acme-llm").mkdir(parents=True)
    with pytest.raises(ValueError, match="denylist"):
        validate_env_var_name_for_write("ACME_CLI_PATH")
    validate_env_var_name_for_write("ACME_BASE_URL")  # an address, not executable authority


@pytest.mark.parametrize("plugin_dir", [False, True])
def test_copilot_launch_names_stay_refused_and_its_base_url_writable(tmp_path, monkeypatch, plugin_dir):
    from hermes_cli.config import validate_env_var_name_for_write
    import providers
    home = _home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(providers, "list_providers", list)  # no profile registered: plugin not loaded
    if plugin_dir:
        (home / "plugins" / "github-copilot").mkdir(parents=True)
    for name in ("HERMES_COPILOT_ACP_COMMAND", "HERMES_COPILOT_ACP_ARGS", "COPILOT_CLI_PATH"):
        with pytest.raises(ValueError, match="denylist"):
            validate_env_var_name_for_write(name)
    validate_env_var_name_for_write("COPILOT_ACP_BASE_URL")


def test_models_dev_never_stands_in_for_a_provider_whose_plugin_left_core(acme_row, monkeypatch):
    """models.dev lists the vendor's endpoint under the same id: resolving it there would send the user's
    raw token to that endpoint through a generic client the plugin never was."""
    from types import SimpleNamespace
    import hermes_cli.providers as hp
    monkeypatch.setattr(hp, "_models_dev_info", lambda name, allow_network=True: SimpleNamespace(
        env=("ACME_TOKEN",), name="Acme", api="https://api.acme.test", doc="") if name == "acme" else None)
    assert hp.get_provider("acme") is None
    assert hp.get_provider("acme-acp") is None
    import providers
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: object())  # the plugin is loaded
    assert hp.get_provider("acme").base_url == "https://api.acme.test"


def test_a_row_without_notice_adds_nothing(tmp_path):
    feature = replace(ACME, notice=lcm._no_notice)
    assert lcm._noticed("x", tmp_path, feature, False) == "x"


# ── hosts: a route at the plugin's host is use, whatever provider name it goes by ─────────────────


@pytest.mark.parametrize("config", [
    "model:\n  provider: custom\n  base_url: https://api.githubcopilot.com\n",
    "model:\n  provider: work\ncustom_providers:\n  - name: work\n    base_url: https://api.business.githubcopilot.com\n",
    "providers:\n  gh:\n    base_url: https://models.github.ai/inference\n",
    "auxiliary:\n  vision:\n    provider: custom\n    base_url: https://api.enterprise.githubcopilot.com/\n",
    "delegation:\n  provider: custom\n  base_url: https://api.githubcopilot.com\n",
    "fallback_providers:\n  - provider: custom\n    base_url: https://api.githubcopilot.com\n    model: gpt-5.4\n",
])
def test_a_route_at_the_rows_host_is_use(tmp_path, config):
    feature = next(f for f in lcm.LEFT_CORE if f.plugin == "github-copilot")
    assert lcm.feature_in_use(feature, _home(tmp_path, config=config)) is True


@pytest.mark.parametrize("url", ["https://githubcopilot.com.evil.test/v1", "https://evil.test/api.githubcopilot.com"])
def test_a_lookalike_host_is_not_use(tmp_path, url):
    feature = next(f for f in lcm.LEFT_CORE if f.plugin == "github-copilot")
    home = _home(tmp_path, config=f"model:\n  provider: custom\n  base_url: {url}\n")
    assert lcm.feature_in_use(feature, home) is False


def test_a_request_rejected_at_the_rows_host_names_the_install(acme_row, monkeypatch, capsys):
    from agent.error_classifier import classify_api_error
    from agent.turn_recovery import nonretryable_client_error_result
    monkeypatch.setattr(lcm, "LEFT_CORE", (replace(ACME, hosts=("acme.test",)),))
    printed = []

    class _Rejected(Exception):
        status_code = 400

    class _Agent:
        log_prefix = ""

        def _summarize_api_error(self, error):
            return str(error)

        def _vprint(self, line, **kw):
            printed.append(line)

        def __getattr__(self, name):
            return lambda *a, **kw: None

    def reject(base_url):
        printed.clear()
        error = _Rejected("HTTP 400: bad request")
        nonretryable_client_error_result(
            _Agent(), error, classify_api_error(error, provider="custom", model="m"), status_code=400,
            api_kwargs=None, api_messages=[], messages=[], conversation_history=None, api_call_count=1,
            approx_tokens=10, provider="custom", base_url=base_url, model="m")
        return "\n".join(printed)
    assert "hermes plugins install acme-llm" in reject("https://api.eu.acme.test/v1")
    assert "acme-llm" not in reject("https://api.other.test/v1")
    import providers
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: object())  # the plugin is loaded
    assert "acme-llm" not in reject("https://api.eu.acme.test/v1")
