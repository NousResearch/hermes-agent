"""T2 — one full-process smoke for single-gateway multi-agent routing.

The in-process integration tier (``tests/integration/multi_agent/``) shares the
pytest event loop, so it cannot prove the mechanism survives a real OS process /
executor-thread boundary. This adds exactly ONE test that does: it drives two
routed agents through the REAL ``APIServerAdapter`` over REAL aiohttp HTTP inside
a REAL separate container process, and asserts — via on-disk artifacts the run
wrote under each agent's SCOPED home — that per-agent ROUTING + per-agent HOME +
per-agent CREDENTIAL reached the run across that boundary.

Why the worktree is mounted
---------------------------
The cached harness image (``hermes-agent-harness:latest``) predates PR #62944, so
its baked-in ``/opt/hermes`` lacks the feature (``set_routing_context`` /
``_use_profile_and_secret_scope`` / persisted-agent session scope). Baking a fresh
image per run costs minutes. Instead we mount the worktree read-only at
``/host_repo`` and prepend it to ``sys.path`` in the probe, so the container
process runs the FEATURE code under test while reusing the image's Python + deps.
The probe writes ``run_dump.json`` under ``get_hermes_home()``; because the real
profile+secret scope redirects the home per agent, coder's dump lands under
``profiles/coder/`` and research's under ``profiles/research/`` — the observable
proof the ContextVar scope propagated across the real executor thread.

Skips automatically unless a Docker daemon is available (see conftest's
``pytest_collection_modifyitems``).
"""
from __future__ import annotations

import json
import os
import subprocess

from tests.docker.conftest import docker_exec

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_HOME = "/tmp/smoke_home"
_PROBE = "/host_repo/tests/docker/multi_agent_smoke_probe.py"

_EXPECTED = {
    "coder": {"key": "sk-coder-smoke", "soul": "I am CODER. Scope: coder."},
    "research": {"key": "sk-research-smoke", "soul": "I am RESEARCH. Scope: research."},
}


def _start_mounted_container(image: str, name: str) -> None:
    """Start a bare container with the worktree mounted read-only.

    We deliberately bypass ``start_container`` (which boots the full s6/cont-init
    tree): this smoke needs only a real process host running the mounted feature
    code, not the service supervisor.
    """
    subprocess.run(
        ["docker", "run", "-d", "--name", name,
         "-v", f"{REPO_ROOT}:/host_repo:ro",
         "--entrypoint", "sleep", image, "300"],
        check=True, capture_output=True, timeout=60,
    )


def test_multi_agent_routing_scope_crosses_real_process_boundary(
    built_image: str, container_name: str,
) -> None:
    _start_mounted_container(built_image, container_name)

    # Run the probe as the unprivileged hermes user — a real, separate OS
    # process executing the feature code over real aiohttp HTTP.
    r = docker_exec(
        container_name, "python3", _PROBE,
        user="hermes", timeout=120,
        extra_docker_args=("-e", f"HERMES_HOME={_HOME}", "-e", "HOST_REPO=/host_repo"),
    )
    assert "PROBE-OK" in r.stdout, (
        f"probe failed: rc={r.returncode}\nstdout={r.stdout!r}\nstderr={r.stderr!r}"
    )

    # Ground truth: each agent's run wrote its dump under its OWN scoped home.
    dumps: dict[str, dict] = {}
    for aid in _EXPECTED:
        cat = docker_exec(
            container_name, "cat", f"{_HOME}/profiles/{aid}/run_dump.json",
            user="hermes", timeout=10,
        )
        assert cat.returncode == 0, (
            f"no run_dump for {aid} under its scoped home: {cat.stderr!r}")
        dumps[aid] = json.loads(cat.stdout)

    for aid, exp in _EXPECTED.items():
        d = dumps[aid]
        # Routed to the right agent...
        assert d["agent_id"] == aid, f"{aid}: ran as {d['agent_id']}"
        # ...reached that agent's per-agent home...
        assert d["home"].endswith(f"/profiles/{aid}"), f"{aid}: home={d['home']}"
        assert d["soul_first_line"] == exp["soul"]
        # ...and resolved that agent's OWN credential, never the other's or root.
        assert d["resolved_key"] == exp["key"], f"{aid}: key={d['resolved_key']}"
        assert d["resolved_key"] != "sk-ROOT-env"

    # Cross-isolation: the two runs did not share home or credential.
    assert dumps["coder"]["home"] != dumps["research"]["home"]
    assert dumps["coder"]["resolved_key"] != dumps["research"]["resolved_key"]

    # And the process-global root home saw no run dump (no leakage to root).
    root = docker_exec(
        container_name, "test", "-f", f"{_HOME}/run_dump.json",
        user="hermes", timeout=10,
    )
    assert root.returncode != 0, "a run leaked its dump to the root home"
