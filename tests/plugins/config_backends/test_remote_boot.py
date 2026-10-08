"""Remote config: backend selection, the fail-closed boot fetch, the plane credential and the deployment bootstrap (``.env`` layers, secret sources, routed children)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.config_backend import (
    ConfigBackendUnavailable,
    get_config_backend,
)

from .conftest import _gets
from .stub_plane import INSTANCE, StubPlane, remote_env


def test_selected_from_env_and_file_tooling_off(plane):
    backend = get_config_backend()
    assert backend.name == "remote"
    assert backend.supports_file_tooling() is False
    assert backend.honors_managed_config() is False
    assert "GATEWAY_RELAY_IDP_CLIENT_SECRET" in backend.protected_env_names()


def test_boot_fetch_runs_inside_load_hermes_dotenv(plane):
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=plane.home)
    assert len(_gets(plane)) == 1
    load_hermes_dotenv(hermes_home=plane.home)  # idempotent: one fetch per process per profile
    assert len(_gets(plane)) == 1


def test_boot_retries_then_exits_on_outage(plane):
    from hermes_cli.env_loader import load_hermes_dotenv
    plane.fail_status = 503
    with pytest.raises(ConfigBackendUnavailable) as exc:
        load_hermes_dotenv(hermes_home=plane.home)
    assert len(_gets(plane)) == 3  # attempts at t=0, ~10 s, ~30 s (delays patched to 0)
    assert "does not start" in str(exc.value.code)
    assert isinstance(exc.value, SystemExit)


def test_boot_does_not_retry_a_refusal(plane, monkeypatch):
    from hermes_cli.env_loader import load_hermes_dotenv
    monkeypatch.setenv("HERMES_CONFIG_INSTANCE_ID", "someone-else")
    with pytest.raises(ConfigBackendUnavailable) as exc:
        load_hermes_dotenv(hermes_home=plane.home)
    assert len(_gets(plane)) == 1
    assert "config_agent_unknown" in str(exc.value.code)


def test_boot_requires_instance_id(plane, monkeypatch):
    from hermes_cli.config import load_config
    monkeypatch.delenv("HERMES_CONFIG_INSTANCE_ID")
    with pytest.raises(ConfigBackendUnavailable, match="HERMES_CONFIG_INSTANCE_ID"):
        load_config()
    assert plane.requests == []


def test_bad_idp_credentials_fail_closed(plane, monkeypatch):
    from hermes_cli.config import load_config
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_SECRET", "wrong")
    with pytest.raises(ConfigBackendUnavailable, match="IdP token request"):
        load_config()
    assert plane.requests == []  # no /self call without a token


def test_partial_idp_config_is_not_retried(plane, monkeypatch):
    from hermes_cli.config import load_config
    monkeypatch.delenv("GATEWAY_RELAY_IDP_CLIENT_SECRET")
    with pytest.raises(ConfigBackendUnavailable, match="CLIENT_SECRET"):
        load_config()
    assert plane.token_requests == 0


def test_secret_source_supplying_plane_credential_refuses_start(plane):
    from types import SimpleNamespace

    from hermes_cli.env_loader import _refuse_protected_env_from_sources
    report = SimpleNamespace(provenance={"GATEWAY_RELAY_IDP_CLIENT_SECRET": object()},
                             sources=[SimpleNamespace(skipped_existing=["HERMES_CONFIG_REMOTE_URL"])])
    with pytest.raises(ConfigBackendUnavailable) as exc:
        _refuse_protected_env_from_sources(report)
    assert "GATEWAY_RELAY_IDP_CLIENT_SECRET" in str(exc.value.code)
    assert "HERMES_CONFIG_REMOTE_URL" in str(exc.value.code)
    ok = SimpleNamespace(provenance={"OPENROUTER_API_KEY": object()}, sources=[])
    _refuse_protected_env_from_sources(ok)  # unrelated names pass


def test_boot_refuses_remote_secrets_source_that_maps_plane_credential(plane, monkeypatch):
    """End to end through load_hermes_dotenv: the REMOTE secrets: section drives the sources (D32)."""
    from types import SimpleNamespace

    from agent.secret_sources import registry
    from hermes_cli import env_loader
    plane.upper = {"secrets": {"command": {"enabled": True}}}
    seen = {}

    def fake_apply_all(cfg, home_path, environ=None):
        seen["cfg"] = cfg
        return SimpleNamespace(sources=[SimpleNamespace(skipped_existing=[])], applied_any=True,
                               provenance={"HERMES_CONFIG_REMOTE_URL": SimpleNamespace(source="command")})

    monkeypatch.setattr(registry, "apply_all", fake_apply_all)
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    with pytest.raises(ConfigBackendUnavailable, match="HERMES_CONFIG_REMOTE_URL"):
        env_loader.load_hermes_dotenv(hermes_home=plane.home)
    assert seen["cfg"] == {"command": {"enabled": True}}


_BACKEND_FLIP_SOURCE = {"command": {"enabled": True, "override_existing": True,
                                    "command": "printf 'HERMES_CONFIG_BACKEND=file\\n'"}}


def test_secret_source_cannot_switch_remote_mode_off_at_boot(plane, monkeypatch):
    """D32 through the REAL startup path and a real command source: a source that writes
    HERMES_CONFIG_BACKEND=file must not disarm remote mode (and with it the protected-name check)
    and let the next read fall back to the local config.yaml."""
    from hermes_cli import env_loader
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    (plane.home / "config.yaml").write_text(json.dumps({"display": {"personality": "local"}}))
    plane.upper = {"display": {"personality": "remote"}, "secrets": _BACKEND_FLIP_SOURCE}

    with pytest.raises(ConfigBackendUnavailable, match="HERMES_CONFIG_BACKEND"):
        env_loader.load_hermes_dotenv(hermes_home=plane.home)

    import os
    assert os.environ["HERMES_CONFIG_BACKEND"] == "remote"  # the source's write was reverted
    assert get_config_backend().name == "remote"


def test_refused_source_plane_url_is_never_visible_to_a_concurrent_request(plane, monkeypatch):
    """D32 credential boundary: a real command source supplies HERMES_CONFIG_REMOTE_URL pointing
    at another plane. A poll scheduled right after the sources ran (before the refusal) must still
    go to the configured plane: the forbidden URL is never published, so the bearer and instance id
    never reach the other server."""
    import os
    import threading

    from agent.secret_sources import registry
    from hermes_cli import env_loader
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    with StubPlane() as other:
        plane.upper = {"secrets": {"command": {
            "enabled": True, "override_existing": True,
            "command": f"printf 'HERMES_CONFIG_REMOTE_URL={other.url}\\n'"}}}
        backend = get_config_backend()
        st = backend._state(plane.home)
        real_apply_all = registry.apply_all
        seen = {}

        def apply_then_poll(*args, **kwargs):
            report = real_apply_all(*args, **kwargs)
            seen["url_after_sources"] = os.environ.get("HERMES_CONFIG_REMOTE_URL")
            poll = threading.Thread(target=backend.poll_one, args=(st,))
            poll.start()
            poll.join(10)
            assert not poll.is_alive()
            return report

        monkeypatch.setattr(registry, "apply_all", apply_then_poll)
        gets_before = len(_gets(plane))
        with pytest.raises(ConfigBackendUnavailable, match="HERMES_CONFIG_REMOTE_URL"):
            env_loader.load_hermes_dotenv(hermes_home=plane.home)

        assert "url_after_sources" in seen, "precondition: the source ran"
        assert other.requests == []  # nothing — least of all the bearer — reached the other plane
        assert seen["url_after_sources"] == plane.url
        assert len(_gets(plane)) > gets_before  # the gap poll went to the configured plane
        assert os.environ["HERMES_CONFIG_REMOTE_URL"] == plane.url


def test_file_mode_publishes_permitted_source_values(monkeypatch, tmp_path):
    """Staging must not lose ordinary secrets: permitted names are published after the check."""
    import os

    from hermes_cli import env_loader
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    monkeypatch.setenv("CC_STAGED_SECRET", "x")
    monkeypatch.delenv("CC_STAGED_SECRET")
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n    command: \"printf 'CC_STAGED_SECRET=from-source\\\\n'\"\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    env_loader._apply_external_secret_sources(home)

    assert os.environ.get("CC_STAGED_SECRET") == "from-source"


def _drop_process_deployment(plane, monkeypatch, tmp_path):
    """Move the deployment out of the process env (as on a host where it lives in .env files);
    returns the managed dir. Every name stays recorded by monkeypatch, so whatever a dotenv load
    publishes is undone at teardown."""
    from hermes_cli import config_backend
    for name in remote_env(plane):
        monkeypatch.delenv(name, raising=False)
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setattr(config_backend, "_BOOTSTRAPPED", True)  # this process is past its first read
    return managed


def test_instance_id_from_the_managed_dotenv_boots(plane, monkeypatch, tmp_path):
    """F4: the deployment in the home's .env, the instance id only in the managed .env."""
    import os

    from hermes_cli.env_loader import load_hermes_dotenv
    managed = _drop_process_deployment(plane, monkeypatch, tmp_path)
    env = remote_env(plane)
    (managed / ".env").write_text(f"HERMES_CONFIG_INSTANCE_ID={env.pop('HERMES_CONFIG_INSTANCE_ID')}\n")
    (plane.home / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    plane.upper = {"display": {"personality": "managed"}}

    load_hermes_dotenv(hermes_home=plane.home)

    gets = _gets(plane)
    assert gets and gets[0]["instance"] == INSTANCE
    assert os.environ["HERMES_CONFIG_INSTANCE_ID"] == INSTANCE


def test_child_for_another_profile_keeps_the_remote_deployment(plane, monkeypatch, tmp_path):
    """F5: the deployment came from the launch profile's .env. A Hermes child built for another
    profile must still read that profile's config remotely, never fall back to its local file."""
    import os
    import subprocess
    import sys

    from hermes_cli.env_loader import load_hermes_dotenv
    from tools.environments.local import served_profile_child_env
    _drop_process_deployment(plane, monkeypatch, tmp_path)
    env = remote_env(plane)
    (plane.home / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    load_hermes_dotenv(hermes_home=plane.home)
    beta = plane.home / "profiles" / "beta"
    beta.mkdir(parents=True)
    (beta / "config.yaml").write_text("display:\n  personality: local-child\n")
    # beta's own plane credential (credentials are never carried into another profile's child)
    (beta / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items() if k.startswith("GATEWAY_RELAY_IDP_")))
    plane.profile("beta")["values"] = {"display": {"personality": "remote-child"}}

    child_env = served_profile_child_env(target_home=beta, inherit_credentials=True)
    child_env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    code = ("import json\n"
            "from hermes_cli.env_loader import load_hermes_dotenv\n"
            "load_hermes_dotenv()\n"
            "from hermes_cli.config_backend import get_config_backend\n"
            "from hermes_cli.config import load_config\n"
            "print('RESULT=' + json.dumps([get_config_backend().name, load_config()['display']['personality']]))\n")
    proc = subprocess.run([sys.executable, "-c", code], env=child_env, capture_output=True, text=True,
                          timeout=120, stdin=subprocess.DEVNULL, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.startswith("RESULT=")][-1]
    assert json.loads(line[len("RESULT="):]) == ["remote", "remote-child"]
    assert "beta" in {r["profile"] for r in _gets(plane)}
    assert not any(k.startswith("GATEWAY_RELAY_IDP_") and v != env[k] for k, v in child_env.items())
    assert os.environ["HERMES_CONFIG_BACKEND"] == "remote"


def _routed_profile(profile_plane, source_env_line):
    named = profile_plane.home / "profiles" / "routed"
    named.mkdir(parents=True)
    (named / ".env").write_text("# synthetic\n")
    profile_plane.profile("routed").update(values={"secrets": {"command": {
        "enabled": True, "override_existing": True,
        "command": f"printf '{source_env_line}\\n'"}}}, version=1)
    return named


def test_routed_hydration_refuses_a_source_that_supplies_a_protected_name(profile_plane, monkeypatch):
    """Round 4 #2: a routed profile's remote secrets: source supplies HERMES_PORTAL_BASE_URL (where
    auth.json's refresh token is sent). Refused before any snapshot, ownership or scope publication;
    every retry refuses again; the profile hydrates once the mapping is gone."""
    import os

    from hermes_cli import env_loader
    from hermes_cli.web_server_profiles import _config_profile_scope
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    monkeypatch.setattr(env_loader, "_SOURCE_SUPPLIED_NAMES", set())
    monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    named = _routed_profile(profile_plane, "HERMES_PORTAL_BASE_URL=http://127.0.0.1:1")

    for _attempt in range(2):  # nothing is cached by a refusal: a retry refuses again
        with pytest.raises(ConfigBackendUnavailable, match="HERMES_PORTAL_BASE_URL"):
            with _config_profile_scope("routed"):
                pass
        assert env_loader.get_secret_source_values(named) == {}
        assert str(named.resolve()) not in env_loader._APPLIED_HOMES
        assert "HERMES_PORTAL_BASE_URL" not in env_loader._SOURCE_SUPPLIED_NAMES
        assert "HERMES_PORTAL_BASE_URL" not in os.environ

    profile_plane.profile("routed").update(values={"secrets": {"command": {
        "enabled": True, "override_existing": True, "command": "printf 'SYNTHETIC_KEY=routed-ok\\n'"}}}, version=2)
    backend = get_config_backend()
    assert backend.poll_one(backend._state(named))
    assert env_loader.hydrate_profile_secret_sources(named) == {"SYNTHETIC_KEY": "routed-ok"}


def test_file_mode_routed_hydration_drops_only_the_selector(monkeypatch, tmp_path, capsys):
    """Control: under the file backend a routed source may not switch the backend either, but its
    other values (a Portal URL included: file mode has no plane credential) still hydrate."""
    from hermes_cli import env_loader
    for name in ("HERMES_CONFIG_BACKEND", "HERMES_CONFIG_REMOTE_URL", "HERMES_CONFIG_INSTANCE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    home = tmp_path / "routed"
    home.mkdir()
    (home / ".env").write_text("# synthetic\n")
    (home / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n    override_existing: true\n"
        "    command: \"printf 'HERMES_CONFIG_BACKEND=remote\\\\nHERMES_PORTAL_BASE_URL=http://portal.test\\\\n'\"\n")

    values = env_loader.hydrate_profile_secret_sources(home)

    assert values == {"HERMES_PORTAL_BASE_URL": "http://portal.test"}
    assert "ignored HERMES_CONFIG_BACKEND" in capsys.readouterr().err


def test_concurrent_first_reader_waits_for_the_bootstrap(plane, monkeypatch, tmp_path):
    """Round 4 #4: while one thread publishes the deployment from .env, a concurrent first reader
    must wait for it, not select the file backend and read the local config.yaml meanwhile."""
    import threading

    from hermes_cli import config_backend, env_loader
    from hermes_cli.config_backend import read_config_doc
    _drop_process_deployment(plane, monkeypatch, tmp_path)
    (plane.home / ".env").write_text("".join(f"{k}={v}\n" for k, v in remote_env(plane).items()))
    (plane.home / "config.yaml").write_text("display:\n  personality: forbidden-local\n")
    plane.profile("default").update(values={"display": {"personality": "remote"}}, version=1)
    monkeypatch.setattr(config_backend, "_BOOTSTRAPPED", False)
    entered, resume = threading.Event(), threading.Event()
    real_apply = env_loader.apply_config_bootstrap_env

    def paused(*args, **kwargs):
        entered.set()
        assert resume.wait(10)
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(env_loader, "apply_config_bootstrap_env", paused)
    results = {}

    def run(name, fn):
        threading.Thread(target=lambda: results.__setitem__(name, fn()), name=name, daemon=True).start()

    run("first", lambda: get_config_backend().name)
    assert entered.wait(10)
    run("concurrent", lambda: read_config_doc(plane.home / "config.yaml")["display"]["personality"])
    import time
    time.sleep(0.3)
    assert "concurrent" not in results  # waiting for the bootstrap, not reading the local file
    resume.set()
    deadline = time.monotonic() + 10
    while len(results) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert results == {"first": "remote", "concurrent": "remote"}


def test_same_thread_reentry_during_the_bootstrap_does_not_deadlock(monkeypatch):
    """Control: an import-time config read on the bootstrapping thread itself returns at once."""
    from hermes_cli import config_backend, env_loader
    monkeypatch.setattr(config_backend, "_BOOTSTRAPPED", False)
    seen = []

    def reentrant(*args, **kwargs):
        seen.append(config_backend.get_config_backend().name)  # would block on a plain Lock

    monkeypatch.setattr(env_loader, "apply_config_bootstrap_env", reentrant)
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    assert config_backend.get_config_backend().name == "file"
    assert seen == ["file"] and config_backend._BOOTSTRAPPED


def _fresh_interpreter_boot(tmp_path, env, project_env):
    import os
    import subprocess
    import sys
    code = ("import json\n"
            "from hermes_cli.env_loader import load_hermes_dotenv\n"
            f"load_hermes_dotenv(project_env={str(project_env)!r}, load_external_secrets=False)\n"
            "from hermes_cli.config_backend import get_config_backend\n"
            "from hermes_cli.config import load_config\n"
            "print('RESULT=' + json.dumps([get_config_backend().name, load_config()['display']['personality']]))\n")
    child = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR", "SYSTEMROOT")}
    child.update(env, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    return subprocess.run([sys.executable, "-c", code], env=child, capture_output=True, text=True,
                          timeout=120, stdin=subprocess.DEVNULL, cwd=str(tmp_path))


def test_plane_credential_from_the_project_dotenv_boots(plane, tmp_path):
    """Round 4 #5: user .env selects remote and names the plane and instance; the project .env (a
    supported load_hermes_dotenv layer) holds the IdP credential. A fresh process boots and fetches:
    the early bootstrap sees the project layer before the sanitizer's config import reads config."""
    env = remote_env(plane)
    (plane.home / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items() if k.startswith("HERMES_CONFIG_")))
    project = tmp_path / "project.env"
    project.write_text("".join(f"{k}={v}\n" for k, v in env.items() if k.startswith("GATEWAY_RELAY_IDP_")))
    (plane.home / "config.yaml").write_text("display:\n  personality: forbidden-local\n")
    plane.profile("default").update(values={"display": {"personality": "remote"}}, version=1)

    proc = _fresh_interpreter_boot(tmp_path, {"HERMES_HOME": str(plane.home), "HOME": str(tmp_path)}, project)

    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT=")][-1]
    assert json.loads(line[len("RESULT="):]) == ["remote", "remote"]
    assert _gets(plane)


def test_bootstrap_keeps_the_dotenv_precedence(tmp_path, monkeypatch):
    """The bootstrap composes the layers as load_hermes_dotenv does: user .env overrides the
    process, the project .env only fills gaps when a user .env exists, managed .env wins last."""
    from hermes_cli.env_loader import _bootstrap_env
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("CC_A", "process")
    monkeypatch.setenv("CC_D", "process")
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    (home / ".env").write_text("CC_A=user\nCC_B=user\n")
    (home / ".op.env").write_text("CC_B=op\nCC_E=op\n")
    project = tmp_path / "project.env"
    project.write_text("CC_B=project\nCC_C=project\nCC_D=project\n")
    (managed / ".env").write_text("CC_C=managed\n")

    env = _bootstrap_env(home, project)

    assert (env["CC_A"], env["CC_B"], env["CC_C"], env["CC_D"], env["CC_E"]) == (
        "user", "user", "managed", "process", "op")
    (home / ".env").unlink()  # no user .env: the project layer overrides the process
    assert _bootstrap_env(home, project)["CC_D"] == "project"
