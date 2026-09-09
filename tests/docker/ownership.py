"""Track disposable containers created by this test process, by immutable ID."""

import re


_containers: dict[str, str] = {}


def register_container(name: str, container_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", container_id):
        raise ValueError("Docker did not return a full container ID")
    _containers[name] = container_id


def container_target(name: str) -> str:
    return _containers.get(name, name)


def release_container(name: str) -> str:
    return _containers.pop(name, name)


def is_owned_container_exec(cmd) -> bool:
    """Accept only the harness's argv form targeting an owned container ID.

    Shell strings, Docker options, arbitrary names and external containers
    do not establish isolation from the host gateway.
    """
    return (
        isinstance(cmd, (list, tuple))
        and len(cmd) >= 6
        and list(cmd[:3]) == ["docker", "exec", "-u"]
        and cmd[4] in _containers.values()
    )
