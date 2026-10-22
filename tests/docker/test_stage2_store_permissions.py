"""SQLite stores must be owner-only after every container start, at the data root and in every profile.

The image's umask is 0022, so a ``state.db`` / ``lcm.db`` that the runtime creates is born
0644 — every session, message and context chunk in it is readable by any user in the container —
and a host-mounted volume can arrive 0664 the same way. stage2 tightens those stores on every
boot (same guard→chown→chmod shape as the ``.env`` block), and stays targeted: an unrelated
0644 file keeps the mode it had.

``lcm.db`` is the discriminating target. ``state.db`` is also tightened when the runtime opens it
writable (``hermes_state._secure_state_db_files``), so a mode assertion there passes with or
without stage2; ``lcm.db`` is created by the context-engine store, which has no such guard, and
only the boot hook can set its mode before anything reads it.
"""

from __future__ import annotations

from tests.docker.conftest import docker_exec_sh, restart_container, start_container

STORE_FILES = (
    "state.db",
    "state.db-wal",
    "state.db-shm",
    "lcm.db",
    "lcm.db-wal",
    "lcm.db-shm",
)
PROFILE = "default"
UNRELATED = "scratch-note.txt"


def _stat(container: str, fmt: str, path: str) -> str:
    r = docker_exec_sh(container, f"stat -c {fmt} {path}", user="root", timeout=5)
    assert r.returncode == 0, f"stat {path} failed: {r.stderr}"
    return r.stdout.strip()


def test_stores_are_owner_only_after_boot_at_root_and_in_every_profile(
    built_image: str,
    container_name: str,
) -> None:
    start_container(built_image, container_name)
    docker_exec_sh(
        container_name, f"mkdir -p /opt/data/profiles/{PROFILE}", user="root", timeout=5
    )

    targets = [f"/opt/data/{name}" for name in STORE_FILES]
    targets += [f"/opt/data/profiles/{PROFILE}/{name}" for name in STORE_FILES]
    # 0666 stands in for both failure modes: a store created under the image's
    # umask 0022 (0644) and one that arrived world-writable on a host mount.
    # A live store is never truncated — only its mode is loosened — so the
    # restart exercises the boot path instead of a damaged database.
    seed = " && ".join(
        f"{{ [ -e {t} ] || printf x > {t}; }} && chmod 666 {t}" for t in targets
    )
    r = docker_exec_sh(
        container_name,
        f"{seed} && printf 'keep me' > /opt/data/{UNRELATED} && chmod 644 /opt/data/{UNRELATED}",
        user="root",
        timeout=30,
    )
    assert r.returncode == 0, f"seeding store modes failed: {r.stderr}"
    assert _stat(container_name, "%a", "/opt/data/lcm.db") == "666", (
        "seed did not loosen lcm.db"
    )

    restart_container(container_name)

    for target in targets:
        mode = _stat(container_name, "%a", target)
        assert mode == "600", (
            f"{target} came back {mode} after boot — a store left readable by other users"
        )
        owner = _stat(container_name, "%U", target)
        assert owner == "hermes", f"{target} is owned by {owner} after boot"
    assert _stat(container_name, "%a", f"/opt/data/{UNRELATED}") == "644", (
        "the tighten pass re-moded a file that is not a store — it must stay targeted"
    )
