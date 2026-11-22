"""A card whose profile names no model is refused before dispatch spawns a worker; a profile the
CLI and gateway serve through the provider's catalog default still spawns (#144027).

Fixture and refusal assertions adapted from hehuang139's #144088."""

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def dispatch_profile(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch

    root = tmp_path / ".hermes"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  default: launch-model\n", encoding="utf-8")
    (profile / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "normal")
    monkeypatch.setattr(dispatch, "_restart_safe_worker_argv", lambda task, command: command)
    spawned = []
    monkeypatch.setattr("subprocess.Popen", lambda command, **kw: spawned.append(command) or SimpleNamespace(pid=424242))
    kb.create_board("work")
    return profile, spawned


def _dispatch(profile, config_yaml, **task):
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect_closing

    (profile / "config.yaml").write_text(config_yaml, encoding="utf-8")
    with connect_closing(board="work") as conn:
        task_id = kb.create_task(conn, title="needs a model", assignee="worker", board="work", **task)
        result = dispatch.dispatch_once(conn, board="work")
        return task_id, result, kb.get_task(conn, task_id), kb.list_events(conn, task_id), kb.list_runs(conn, task_id)


def test_unresolved_model_refused_before_spawn_with_task_log(dispatch_profile):
    from hermes_cli import kanban as cli

    profile, spawned = dispatch_profile
    task_id, result, task, events, runs = _dispatch(
        profile, "model:\n  provider: custom\n  base_url: https://llm.example.test/v1\n")
    assert not spawned and not result.spawned
    assert task.status == "ready" and task.worker_pid is None
    assert "No model configured for profile worker; pin one on this card with `hermes kanban set-model`" in task.last_failure_error
    assert any(e.kind == "spawn_failed" and "hermes -p worker model" in e.payload["error"] for e in events)
    assert len(runs) == 1 and runs[0].outcome == "spawn_failed"
    assert "hermes -p worker model" in cli.run_slash(f"--board work log {task_id}")


@pytest.mark.parametrize("config_yaml, task", [
    ("model:\n  provider: openrouter\n", {}),
    ("model:\n  provider: custom\n  base_url: https://llm.example.test/v1\n", {"model_override": "task-model"}),
], ids=["catalog-default", "task-override"])
def test_resolvable_model_still_spawns(dispatch_profile, config_yaml, task):
    profile, spawned = dispatch_profile
    task_id, result, row, _, _ = _dispatch(profile, config_yaml, **task)
    assert len(spawned) == 1 and result.spawned[0][0] == task_id
    assert row.status == "running" and row.worker_pid == 424242


@pytest.mark.parametrize("provider, env_var, yaml_url", [
    ("custom", "CUSTOM_BASE_URL", ""),
    ("custom", "CUSTOM_BASE_URL", "  base_url: https://stale.example.test/v1\n"),
    ("openrouter", "OPENROUTER_BASE_URL", ""),
], ids=["custom-dotenv", "custom-overrides-yaml", "openrouter-dotenv"])
def test_assignee_dotenv_local_endpoint_defers_in_dispatch_and_owner(
        dispatch_profile, provider, env_var, yaml_url):
    from agent.runtime_scope import bind_profile_scope
    from gateway.session_kanban import kanban_launch_policy
    from gateway.session_worker_construct import route_runtime
    from hermes_cli.config_effective import load_user_config_effective

    profile, spawned = dispatch_profile
    endpoint = "http://127.0.0.1:9/v1"
    (profile / ".env").write_text(f"{env_var}={endpoint}\nOPENROUTER_API_KEY=fixture-key\n", encoding="utf-8")
    _, result, task, _, _ = _dispatch(profile, f"model:\n  provider: {provider}\n{yaml_url}")
    with bind_profile_scope(profile, hydrate_secrets=False):
        config = load_user_config_effective()
        policy = kanban_launch_policy(task, str(profile), config)
        assert policy.model == ""
        runtime, _ = route_runtime(policy, policy.config(), policy.model, None, fallback=False)
        assert runtime["base_url"] == endpoint
    assert spawned and result.spawned


def _card(**kw):
    return SimpleNamespace(**{"id": "t_x", "assignee": "worker", "model_override": None, "provider_override": None, **kw})


_LAB_ENTRY = [{"name": "lab", "base_url": "https://lab.example.test/v1", "model": "entry-model"}]


def test_a_cards_provider_always_comes_with_its_model(dispatch_profile):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_cli.kanban_worker_preflight import task_model

    config = {"model": {"provider": "openrouter", "default": "profile-model"}, "custom_providers": _LAB_ENTRY}
    with connect_closing(board="work") as conn:
        with pytest.raises(ValueError, match="provider_override requires a model_override"):
            kb.create_task(conn, title="provider only", assignee="worker", board="work", provider_override="custom:lab")
        task_id = kb.create_task(conn, title="paired", assignee="worker", board="work",
                                 model_override="task-model", provider_override="custom:lab")
        card = kb.get_task(conn, task_id)
    assert task_model(card, config) == ("task-model", False)


@pytest.mark.parametrize("profile_model, expected", [
    ("profile-model", "profile-model"),
    ({"provider": "openrouter", "default": "profile-model"}, "profile-model"),
    ({"provider": "custom:lab"}, "entry-model"),
    ({"provider": "anthropic"}, "claude-opus-4-6"),
], ids=["string-profile", "profile-default", "held-named-entry", "catalog-default"])
def test_a_card_without_a_model_takes_the_profiles(profile_model, expected, monkeypatch):
    from hermes_cli import models
    from hermes_cli.kanban_worker_preflight import task_model

    monkeypatch.setattr(models, "get_default_model_for_provider", lambda provider: {"anthropic": "claude-opus-4-6"}.get(provider, ""))
    assert task_model(_card(), {"model": profile_model, "custom_providers": _LAB_ENTRY}) == (expected, False)


@pytest.mark.parametrize("config", [
    {"model": {"provider": "custom", "base_url": "http://127.0.0.1:9/v1"}},
    {"model": {"provider": "openrouter", "base_url": "http://127.0.0.1:8000/v1"}},
    {"model": {"provider": "lmstudio"}},
    {"model": {"provider": "llamacpp"}},
    {"model": {"provider": "llamacpp", "base_url": "https://stale.example.test/v1"}},
    {"model": {"provider": "custom", "base_url": "http://localhost.:8000/v1"}},
    {"model": {"provider": "custom", "base_url": "http://192.168.1.50:8000/v1"}},
    {"model": {"provider": "custom", "base_url": "https://box.tail1234.ts.net/v1"}},
    {"model": {"provider": "custom", "base_url": "http://llm.home.arpa:8000/v1"}},
    {"model": {"provider": "custom", "base_url": "http://gpu.internal:8000/v1"}},
    {"model": {"provider": "custom", "base_url": "http://ollama.lan:11434/v1"}},
    {"model": {"provider": "custom", "base_url": "http://ollama.localhost:11434/v1"}},
    {"model": {"provider": "custom", "base_url": "http://gpu.home:8000/v1"}},
    {"model": {"provider": "custom", "base_url": "http://llm.localdomain:8000/v1"}},
    {"model": {"provider": "custom", "base_url": "http://100.101.102.103:8000/v1"}},
    {"model": {"provider": "custom", "base_url": "https://llm.example.test/v1"},
     "fallback_providers": [{"provider": "openrouter", "model": "fallback-model"}]},
    {"model": {"provider": "custom:lan", "base_url": "https://stale.example.test/v1"},
     "custom_providers": [{"name": "lan", "base_url": "http://192.168.1.50:8000/v1"}]},
    {"model": {"provider": "auto"}},
    {},
], ids=["custom-loopback", "builtin-loopback", "builtin-local-default", "managed-llamacpp", "managed-llamacpp-stale-url", "loopback-trailing-dot", "lan-server", "tailnet-hostname", "home-arpa-hostname", "internal-hostname", "lan-hostname", "localhost-subdomain", "home-hostname", "localdomain-hostname", "tailscale-ip", "fallback-chain", "named-lan-entry", "auto", "unset"])
def test_deferred_routes_are_left_to_the_turn_without_a_network_call(config, tmp_path, monkeypatch):
    import socket

    from gateway.session_kanban import kanban_launch_policy
    from hermes_cli import auth
    from hermes_cli.kanban_worker_preflight import task_model

    opened = []

    def no_network(*args, **kwargs):
        opened.append(args)
        raise OSError("network blocked by test")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    real_resolve = auth.resolve_provider

    def resolve(requested=None, *args, **kwargs):
        if str(requested or "auto").strip().lower() == "auto":
            opened.append(("credential discovery", requested))
        return real_resolve(requested, *args, **kwargs)

    monkeypatch.setattr(auth, "resolve_provider", resolve)
    assert task_model(_card(), config) == ("", False)
    assert kanban_launch_policy(_card(reasoning_effort=None), str(tmp_path), config).model == ""
    assert opened == []


@pytest.mark.parametrize("model, env_var, url, expected", [
    ({"provider": "openai"}, "OPENAI_BASE_URL", "http://127.0.0.1:9/v1", ("", False)),
    ({"provider": "openai-api"}, "OPENAI_BASE_URL", "http://192.168.1.50:8000/v1", ("", False)),
    ({"provider": "deepseek"}, "DEEPSEEK_BASE_URL", "http://127.0.0.1:9/v1", ("", False)),
    ({"provider": "openai-api"}, "OPENAI_BASE_URL", "https://proxy.example.test/v1", ("cloud-model", False)),
    ({"provider": "deepseek", "base_url": "https://api.deepseek.com/v1"}, "DEEPSEEK_BASE_URL", "http://127.0.0.1:9/v1",
     ("", False)),
    ({"provider": "openai", "base_url": "https://api.openai.com/v1"}, "OPENAI_BASE_URL", "http://127.0.0.1:9/v1", ("", False)),
    ({"provider": "deepseek", "base_url": "http://127.0.0.1:8000/v1"}, "DEEPSEEK_BASE_URL", "", ("", False)),
    ({"provider": "deepseek", "base_url": "https://box.tail1234.ts.net/v1"}, "DEEPSEEK_BASE_URL", "", ("", False)),
    ({"provider": "deepseek", "base_url": "https://api.notts.net/v1"}, "DEEPSEEK_BASE_URL", "", ("cloud-model", False)),
    ({"provider": "deepseek", "base_url": "https://llm.milan/v1"}, "DEEPSEEK_BASE_URL", "", ("cloud-model", False)),
], ids=["openai-alias-local", "openai-api-lan", "deepseek-local", "openai-api-public", "env-over-model-base-url",
        "openai-env-over-model-base-url", "model-base-url-without-env", "tailnet-model-base-url", "tailnet-lookalike-host",
        "lan-lookalike-tld"])
def test_a_builtin_providers_own_base_url_variable_decides_locality(model, env_var, url, expected, monkeypatch):
    from hermes_cli import models
    from hermes_cli.kanban_worker_preflight import task_model

    monkeypatch.setenv(env_var, url)
    monkeypatch.setattr(models, "get_default_model_for_provider", lambda provider: "cloud-model")
    assert task_model(_card(), {"model": model}) == expected


@pytest.mark.parametrize("profile_model, expected", [
    ({"provider": "custom:cloud"}, ("", True)),
    ({"provider": "anthropic"}, ("claude-opus-4-6", False)),
], ids=["named-entry", "builtin-provider"])
def test_a_local_dotenv_url_does_not_move_a_route_that_ignores_it(profile_model, expected, monkeypatch):
    from hermes_cli import models
    from hermes_cli.kanban_worker_preflight import task_model

    monkeypatch.setenv("CUSTOM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(models, "get_default_model_for_provider", lambda provider: {"anthropic": "claude-opus-4-6"}.get(provider, ""))
    config = {"model": profile_model, "custom_providers": [{"name": "cloud", "base_url": "https://cloud.example.test/v1"}]}
    assert task_model(_card(), config) == expected


def test_owner_policy_takes_the_card_model_or_refuses_before_a_receipt(tmp_path, monkeypatch):
    from contextlib import closing

    from gateway.session_contract import SessionActor
    from gateway.session_kanban import build_kanban_policy
    from hermes_cli import kanban_db as kb, models
    from hermes_cli.kanban_db_connect import connect
    from persistence.core.errors import AuthorityRefusal

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda name: str(tmp_path))
    monkeypatch.setattr(models, "get_default_model_for_provider", lambda provider: {"anthropic": "claude-opus-4-6"}.get(provider, ""))
    kb.create_board("owned")
    with closing(connect(board="owned")) as conn:
        task_id = kb.create_task(conn, title="needs a model", assignee="default", workspace_kind="dir", workspace_path=str(tmp_path))
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    actor = SessionActor("owner", str(tmp_path), frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board="owned", task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    assert build_kanban_policy(connection, params, {"model": {"provider": "anthropic"}})[0].model == "claude-opus-4-6"
    with pytest.raises(AuthorityRefusal) as refused:
        build_kanban_policy(connection, params, {"model": {"provider": "custom", "base_url": "https://llm.example.test/v1"}})
    assert (refused.value.reason, refused.value.field) == ("invalid_params", "model")
    assert refused.value.detail.startswith("No model configured for profile default")


def test_preflight_never_imports_the_gateway_runner(tmp_path):
    import os
    import subprocess
    import sys

    root = tmp_path / ".hermes"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (profile / ".env").write_text("", encoding="utf-8")
    (profile / "config.yaml").write_text("model:\n  provider: custom\n  base_url: https://llm.example.test/v1\n", encoding="utf-8")
    probe = (
        "import os, sys\n"
        "from types import SimpleNamespace\n"
        "from hermes_cli.kanban_worker_preflight import refuse_unresolved_model\n"
        "card = SimpleNamespace(id='t_fresh', assignee='worker', model_override=None, provider_override=None)\n"
        "try:\n"
        "    refuse_unresolved_model(card, '.', None)\n"
        "except RuntimeError as exc:\n"
        "    print('refused', exc)\n"
        "print('gateway.run' in sys.modules, os.environ.get('_HERMES_GATEWAY'))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "_HERMES_GATEWAY"}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(root), HERMES_KANBAN_HOME=str(root))
    out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0].startswith("refused No model configured for profile worker")
    assert f"HERMES_HOME={profile.parent.parent}" in lines[0] and str(profile) not in lines[0]
    assert lines[-1] == "False None"


def test_unavailable_config_backend_is_an_infrastructure_spawn_failure(dispatch_profile, monkeypatch):
    from hermes_cli.config_backend import ConfigBackendUnavailable

    from hermes_cli import config_effective

    profile, spawned = dispatch_profile
    real = config_effective.load_user_config_effective

    def unavailable(config_path=None, **kwargs):
        if config_path is not None and Path(config_path).parent == profile:
            raise ConfigBackendUnavailable("config plane unreachable")
        return real(config_path, **kwargs)

    monkeypatch.setattr(config_effective, "load_user_config_effective", unavailable)
    _, result, task, _, runs = _dispatch(profile, "model:\n  provider: openrouter\n")
    assert not spawned and not result.spawned
    assert task.status == "ready" and task.worker_pid is None and task.consecutive_failures == 0
    assert "config plane unreachable" in task.last_failure_error
    assert len(runs) == 1 and runs[0].outcome == "spawn_failed" and runs[0].metadata.get("infrastructure") is True


@pytest.mark.parametrize("provider, env_var", [("deepseek", "DEEPSEEK_BASE_URL"), ("openai-api", "OPENAI_BASE_URL")])
def test_an_env_url_that_is_the_registry_default_yields_to_a_local_model_base_url(provider, env_var, tmp_path,
                                                                                  monkeypatch):
    """The pool row keeps model.base_url over an env URL equal to the provider's registry default, so the turn
    goes to the loopback server: the preflight must defer instead of freezing a cloud id for it."""
    from types import SimpleNamespace

    from hermes_cli import models
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.kanban_worker_preflight import task_model
    from hermes_cli.runtime_provider import _pool_entry_mode_and_url

    default_url, local = PROVIDER_REGISTRY[provider].inference_base_url, "http://127.0.0.1:9/v1"
    monkeypatch.setenv(env_var, default_url)
    monkeypatch.setattr(models, "get_default_model_for_provider", lambda provider: "cloud-model")
    model_cfg = {"provider": provider, "base_url": local}
    env_row = SimpleNamespace(source=f"env:{env_var}", runtime_api_key="k")
    assert _pool_entry_mode_and_url(provider, env_row, model_cfg, "", default_url.rstrip("/"))[1] == local
    assert task_model(_card(), {"model": model_cfg}) == ("", False)


@pytest.mark.parametrize("model, env, expected", [
    ({"provider": "lm-studio", "base_url": "https://stale.example.test/v1"}, {}, ("", False)),
    ({"provider": "azure-foundry", "base_url": "http://127.0.0.1:9/v1"},
     {"AZURE_FOUNDRY_BASE_URL": "https://foundry.example.test/v1"}, ("", False)),
    ({"provider": "anthropic", "base_url": "http://127.0.0.1:4000/anthropic"},
     {"ANTHROPIC_BASE_URL": "https://relay.example.test"}, ("", False)),
    ({"provider": "deepseek", "base_url": "https://proxy.example.test/v1"},
     {"DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1"}, ("cloud-model", False)),
], ids=["alias-with-stale-url", "config-first-provider", "anthropic-local-relay", "all-public"])
def test_a_self_hosted_url_the_runtime_may_pick_defers(model, env, expected, monkeypatch):
    """The runtime orders the scoped env URL, model.base_url and the registry default differently per branch (alias
    match, Foundry config-first, Anthropic's /anthropic relay), so any self-hosted candidate defers the card."""
    from hermes_cli import models
    from hermes_cli.kanban_worker_preflight import task_model

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(models, "get_default_model_for_provider", lambda provider: "cloud-model")
    assert task_model(_card(), {"model": model}) == expected


@pytest.mark.parametrize("url, expected", [
    ("http://[2606:4700::1111]:8000/v1", ("cloud-model", False)),
    ("http://8.8.8.8:8000/v1", ("cloud-model", False)),
    ("http://[::1]:8000/v1", ("", False)),
    ("http://[fd00::5]:8000/v1", ("", False)),
    ("http://[fe80::1]:8000/v1", ("", False)),
], ids=["public-ipv6", "public-ipv4", "loopback-ipv6", "ula-ipv6", "link-local-ipv6"])
def test_an_ip_literal_is_judged_by_its_scope(url, expected, monkeypatch):
    """A global IPv6 literal has no dot, but it is public: only a name gets the private-use name rules."""
    from hermes_cli import models
    from hermes_cli.kanban_worker_preflight import task_model

    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setattr(models, "get_default_model_for_provider", lambda provider: "cloud-model")
    assert task_model(_card(), {"model": {"provider": "deepseek", "base_url": url}}) == expected


def test_a_pinned_card_spawns_without_reading_its_profiles_config(dispatch_profile, monkeypatch):
    """A card with its own model never needs the profile's: a config backend outage must not defer it."""
    from hermes_cli import config_effective
    from hermes_cli.config_backend import ConfigBackendUnavailable

    profile, spawned = dispatch_profile
    real = config_effective.load_user_config_effective

    def unavailable(config_path=None, **kwargs):
        if config_path is not None and Path(config_path).parent == profile:
            raise ConfigBackendUnavailable("config plane unreachable")
        return real(config_path, **kwargs)

    monkeypatch.setattr(config_effective, "load_user_config_effective", unavailable)
    task_id, result, row, _, _ = _dispatch(profile, "model:\n  provider: openrouter\n", model_override="task-model")
    assert len(spawned) == 1 and result.spawned[0][0] == task_id and row.status == "running"


def test_an_unreadable_profile_config_is_left_to_the_owner_not_refused(dispatch_profile, monkeypatch):
    """Broken YAML with no last-good copy must not read as an empty config: with a managed overlay naming
    a public custom route, that would be refused as "No model configured" and spend the retry budget."""
    from hermes_cli import managed_scope

    profile, spawned = dispatch_profile
    route = {"provider": "custom", "base_url": "https://llm.example.test/v1"}
    monkeypatch.setattr(managed_scope, "apply_managed_overlay", lambda config: {**config, "model": dict(route)})
    task_id, result, row, _, _ = _dispatch(profile, "model: [unclosed\n")
    assert len(spawned) == 1 and result.spawned[0][0] == task_id
    assert row.consecutive_failures == 0


def test_a_refusal_names_the_unblock_that_re_runs_a_blocked_card(dispatch_profile):
    """Refusals spend the retry budget and block the card at the limit; set-model alone then does nothing."""
    profile, _ = dispatch_profile
    task_id, _, task, _, _ = _dispatch(profile, "model:\n  provider: custom\n  base_url: https://llm.example.test/v1\n")
    assert f"`hermes kanban --board work unblock {task_id}`" in task.last_failure_error
