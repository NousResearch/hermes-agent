"""Shared ``docker run`` argv construction (Docker + gVisor / runsc)."""

from __future__ import annotations

import os

from sandbox.linux_hardening import seccomp_security_opt
from sandbox.types import IsolationProfile


def build_docker_run_argv(
    *,
    image: str,
    workdir: str,
    inner_cmd: list[str],
    child_env: dict[str, str],
    profile: IsolationProfile,
    docker_runtime: str | None = None,
) -> list[str]:
    """Host-side ``docker run`` argv mounting *workdir* at the same path."""
    if not workdir or not os.path.isabs(workdir):
        raise ValueError("workdir must be an absolute path for docker bind mount")
    cmd: list[str] = ["docker", "run", "--rm"]
    if docker_runtime:
        cmd += ["--runtime", docker_runtime]
    net = (profile.network_policy or "bridge").lower()
    if net == "none":
        cmd += ["--network", "none"]
    if profile.cpu_quota and profile.cpu_quota > 0:
        cmd += ["--cpus", str(profile.cpu_quota)]
    if profile.mem_limit_mb and profile.mem_limit_mb > 0:
        cmd += ["--memory", f"{int(profile.mem_limit_mb)}m"]
    for cap in profile.cap_drop:
        if cap:
            cmd += ["--cap-drop", cap]
    cmd.extend(seccomp_security_opt(profile.seccomp_profile_ref))
    for raw in profile.fs_mounts:
        if raw and isinstance(raw, str):
            cmd += ["-v", raw]
    cmd += ["-v", f"{workdir}:{workdir}", "-w", workdir]
    for key, val in child_env.items():
        cmd += ["-e", f"{key}={val}"]
    cmd.append(image)
    cmd.extend(inner_cmd)
    return cmd
