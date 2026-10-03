"""Launch-environment credentials remain owned by the default multiplex profile.

Container, systemd, and ``op run`` deployments commonly inject credentials into the
gateway process without writing them to ``<home>/.env``.  Once multiplexing is active,
the default profile must see the environment captured at gateway startup while every
secondary profile remains isolated from it.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from agent import secret_scope
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner, _profile_runtime_scope
from tools.tool_backend_helpers import resolve_provider_secret
from tui_gateway import launch_profile_policy


def test_default_profile_keeps_frozen_launch_env_without_leaking_to_secondary(
    tmp_path, monkeypatch,
):
    launch_home = tmp_path / "launch"
    launch_home.mkdir()
    (launch_home / ".env").write_text("OPENAI_API_KEY=launch-file-key\n", encoding="utf-8")
    secondary_home = tmp_path / "profiles" / "secondary"
    secondary_home.mkdir(parents=True)
    (secondary_home / ".env").write_text("GROQ_API_KEY=secondary-file-key\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("GROQ_API_KEY", "launch-container-key")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)

    GatewayRunner(GatewayConfig(multiplex_profiles=True))
    assert secret_scope.is_multiplex_active()

    # Later process-env mutations are not another profile's authority.  An unscoped
    # resolver must fail closed instead of accidentally reading this live value.
    monkeypatch.setenv("GROQ_API_KEY", "late-process-value")
    assert resolve_provider_secret("GROQ_API_KEY", "groq") == ""

    # A -> B -> A proves both launch preservation and secondary isolation through the
    # same production scope used by cold and busy gateway turns.
    with _profile_runtime_scope(launch_home):
        assert resolve_provider_secret("GROQ_API_KEY", "groq") == "launch-container-key"
    with _profile_runtime_scope(secondary_home):
        assert resolve_provider_secret("GROQ_API_KEY", "groq") == "secondary-file-key"
    with _profile_runtime_scope(launch_home):
        assert resolve_provider_secret("GROQ_API_KEY", "groq") == "launch-container-key"

    # Concurrent turns keep their context-local scopes: neither side borrows the
    # other's Groq credential while both scopes are installed at the same time.
    barrier = threading.Barrier(2)

    def resolve_inside(home):
        with _profile_runtime_scope(home):
            barrier.wait(timeout=5)
            return resolve_provider_secret("GROQ_API_KEY", "groq")

    with ThreadPoolExecutor(max_workers=2) as executor:
        launch_result = executor.submit(resolve_inside, launch_home)
        secondary_result = executor.submit(resolve_inside, secondary_home)
        assert launch_result.result(timeout=10) == "launch-container-key"
        assert secondary_result.result(timeout=10) == "secondary-file-key"
