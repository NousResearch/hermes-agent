"""GPU (/dev/dri) passthrough must survive the s6-setuidgid privilege drop.

Two halves of one bug, both exercised against the real built image:

  1. The image ships the VAAPI runtime: libva itself plus a vendor driver.
     pm stages a pinned BtbN ffmpeg built --enable-vaapi, but it reaches libva
     through implib-gen stubs that dlopen ``libva-drm.so.2`` on first use, and
     libva then dlopens a per-vendor ``*_drv_video.so``. Neither ships in the
     base image now that ffmpeg no longer comes from apt, so ``-hwaccel vaapi``
     aborts inside the generated stub rather than degrading to software.
  2. stage2-hook.sh gives the ``hermes`` user an /etc/group entry matching the
     GID of each passed-in /dev/dri node. ``docker run --group-add`` alone is
     not enough: initgroups() rebuilds the supplementary list from /etc/group
     during the privilege drop and discards the kernel-granted GID. This is
     the same trap the Docker-socket block above it already handles.

The device nodes are created with ``mknod`` inside the container (CAP_MKNOD is
in Docker's default set) rather than passed through with ``--device``, so these
tests run on CI machines with no GPU. ``/dev`` is a fresh tmpfs on every boot,
so the hook is re-run in place instead of via a container restart.
"""
from __future__ import annotations

from tests.docker.conftest import docker_exec_sh, start_container

# Arbitrary GID with no name in the image, standing in for a host `render`
# group (107 on Debian, 987 on Arch — it is a host fact, not a fixed value).
UNNAMED_GID = 64000
# `video` is GID 44 in Debian base, so it exists in the image by name.
VIDEO_GID = 44

STAGE2_HOOK = "/opt/hermes/docker/stage2-hook.sh"


def _mknod(container: str, path: str, minor: int, gid: int) -> None:
    """Create a char device at ``path`` owned by group ``gid``."""
    r = docker_exec_sh(
        container,
        f"mkdir -p /dev/dri && mknod {path} c 226 {minor} && chgrp {gid} {path}",
        user="root", timeout=15,
    )
    assert r.returncode == 0, f"mknod {path} failed: {r.stderr}"


def _rerun_stage2(container: str) -> str:
    """Re-run the real cont-init hook as root; return its combined output."""
    r = docker_exec_sh(container, STAGE2_HOOK, user="root", timeout=120)
    assert r.returncode == 0, (
        f"stage2-hook.sh exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    )
    return r.stdout + r.stderr


def _dropped_user_gids(container: str) -> set[int]:
    """Supplementary GIDs of a process dropped to the hermes user.

    ``docker exec -u hermes`` calls initgroups() exactly like the
    ``s6-setuidgid hermes`` drop in every service's run script, so this is a
    faithful proxy for what the supervised agent actually gets.
    """
    r = docker_exec_sh(container, "id -G", user="hermes", timeout=10)
    assert r.returncode == 0, f"id -G failed: {r.stderr}"
    return {int(g) for g in r.stdout.split()}


def test_image_ships_a_vaapi_driver(
    built_image: str, container_name: str,
) -> None:
    """libva must have a driver to dispatch to, for Intel and AMD alike."""
    start_container(built_image, container_name)

    r = docker_exec_sh(
        container_name,
        "ls /usr/lib/x86_64-linux-gnu/dri/ | grep drv_video",
        timeout=15,
    )
    drivers = r.stdout.split()
    assert "iHD_drv_video.so" in drivers, (
        f"no Intel VAAPI driver; /dev/dri passthrough cannot work: {drivers}"
    )
    assert "radeonsi_drv_video.so" in drivers, (
        f"no AMD VAAPI driver; /dev/dri passthrough cannot work: {drivers}"
    )


def test_image_ships_the_libva_runtime(
    built_image: str, container_name: str,
) -> None:
    """ffmpeg's implib stubs dlopen libva-drm.so.2 — it must be present.

    The vendor driver alone is not enough: without libva the stub calls
    abort() on first vaapi use, which is louder than a software fallback.
    """
    start_container(built_image, container_name)

    r = docker_exec_sh(
        container_name,
        "ls /usr/lib/x86_64-linux-gnu/ | grep -E '^libva(-drm)?\\.so\\.2$'",
        timeout=15,
    )
    libs = set(r.stdout.split())
    assert {"libva.so.2", "libva-drm.so.2"} <= libs, (
        f"libva runtime incomplete; vaapi will abort in the implib stub: {libs}"
    )


def test_ffmpeg_advertises_vaapi_hwaccel(
    built_image: str, container_name: str,
) -> None:
    """The bundled ffmpeg must expose the hwaccel the driver backs."""
    start_container(built_image, container_name)

    r = docker_exec_sh(
        container_name, "ffmpeg -hide_banner -hwaccels", timeout=20,
    )
    assert "vaapi" in r.stdout, f"ffmpeg has no vaapi hwaccel: {r.stdout}"


def test_render_node_gid_survives_privilege_drop(
    built_image: str, container_name: str,
) -> None:
    """The dropped hermes user must hold the render node's GID.

    This is the whole bug: without the /etc/group entry the GID is silently
    dropped between PID 1 and the supervised process.
    """
    start_container(built_image, container_name)

    before = _dropped_user_gids(container_name)
    assert UNNAMED_GID not in before, (
        f"GID {UNNAMED_GID} was expected to be unused in the image"
    )

    _mknod(container_name, "/dev/dri/renderD128", 128, UNNAMED_GID)
    _rerun_stage2(container_name)

    after = _dropped_user_gids(container_name)
    assert UNNAMED_GID in after, (
        "render node GID did not survive the privilege drop; "
        f"dropped process has {sorted(after)}"
    )


def test_existing_group_name_is_reused_not_duplicated(
    built_image: str, container_name: str,
) -> None:
    """A GID the image already names must be joined, not shadowed by a copy."""
    start_container(built_image, container_name)

    _mknod(container_name, "/dev/dri/card0", 0, VIDEO_GID)
    _rerun_stage2(container_name)

    assert VIDEO_GID in _dropped_user_gids(container_name)

    r = docker_exec_sh(
        container_name, f"getent group {VIDEO_GID} | wc -l", timeout=10,
    )
    assert r.stdout.strip() == "1", (
        f"expected exactly one group at GID {VIDEO_GID}, got {r.stdout.strip()}"
    )


def test_rerun_is_idempotent(
    built_image: str, container_name: str,
) -> None:
    """Restarting a container with a GPU must not re-add or duplicate groups."""
    start_container(built_image, container_name)

    _mknod(container_name, "/dev/dri/renderD128", 128, UNNAMED_GID)
    _rerun_stage2(container_name)
    first = _dropped_user_gids(container_name)

    second_output = _rerun_stage2(container_name)
    assert _dropped_user_gids(container_name) == first
    assert "Added hermes to group" not in second_output, (
        f"second run re-added an existing group: {second_output}"
    )


def test_root_owned_node_does_not_grant_root_group(
    built_image: str, container_name: str,
) -> None:
    """A root-owned node must not hand hermes the root group.

    Joining GID 0 to reach a render node would grant every other root-group
    file in the image — a far larger grant than the GPU that was asked for.
    """
    start_container(built_image, container_name)

    _mknod(container_name, "/dev/dri/renderD128", 128, 0)
    _rerun_stage2(container_name)

    assert 0 not in _dropped_user_gids(container_name), (
        "hermes was added to the root group for a root-owned render node"
    )
