"""Runtime integration checks for the baked Hindsight dependencies."""
from __future__ import annotations

from tests.docker.conftest import docker_exec, start_container


def test_hindsight_runtime_imports_in_baked_image(
    built_image: str, container_name: str,
) -> None:
    start_container(built_image, container_name, cmd="sleep 60")

    result = docker_exec(
        container_name,
        "python",
        "-c",
        (
            "import hindsight_client; "
            "import hindsight_api.main; "
            "import hindsight_embed.daemon_embed_manager; "
            "print('ok')"
        ),
        timeout=30,
    )

    assert result.returncode == 0, (
        "The baked image must import the full Hindsight embedded runtime in one "
        f"Python process. stdout={result.stdout[-2000:]!r} "
        f"stderr={result.stderr[-2000:]!r}"
    )
    assert result.stdout.strip().endswith("ok"), (
        f"Unexpected import smoke output: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
