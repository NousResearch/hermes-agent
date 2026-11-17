"""Approval gate for the default sandbox image change (nikolaik base → hermes-sandbox:desktop).

A persisted Docker sandbox is state the user built up; a default flip must not replace it under
them. ``DockerEnvironment`` keeps a container whose image differs from an UNPINNED default and
this module is where the user decides: approve (pin the new image; the next terminal use
recreates the container — ``/root`` and ``/workspace`` are host mounts and carry over, system
packages get reinstalled, 3.11 venvs need a rebuild) or keep (pin the image the container runs,
which also ends the notice). Surfaces: the interactive CLI startup offer here, the Screen pane's
switch action (``display.switchSandboxImage``) and the plain ``hermes config set``. Headless
surfaces (gateway, cron) never decide; they keep the sandbox and log.

Modal restores its snapshot and Daytona reuses its labeled sandbox regardless of the configured
image, so existing sandboxes there are untouched without any gate; only fresh ones get the new
image.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PendingSwitch:
    current_image: str          # what the persisted container runs
    target_image: str           # the effective default it would move to
    containers: list[str] = field(default_factory=list)   # names, for the notice


def pending() -> Optional[PendingSwitch]:
    """The switch this profile still has to decide, or None: backend is docker, the image is the
    unpinned default, and a container labeled for this profile runs another image. ``docker`` is
    consulted (one ``ps``); any failure is "nothing pending"."""
    from hermes_cli.config import load_config_readonly
    from tools.environments.docker import _container_identity, _docker_query, find_docker

    terminal = (load_config_readonly() or {}).get("terminal") or {}
    if terminal.get("backend") != "docker":
        return None
    if _image_pinned(terminal):
        return None
    target = str(terminal.get("docker_image") or "")
    docker = find_docker()
    if not docker or not target:
        return None
    listing = _docker_query(
        [docker, "ps", "-a", "--filter", "label=hermes-agent=1",
         "--filter", f"label=hermes-profile={_container_identity(str(terminal.get('docker_shared_container_key') or ''))}",
         "--format", "{{.Names}}\t{{.Image}}"],
        timeout=10, fail="sandbox image switch probe failed: %s", nonzero="docker ps exited %s: %s")
    if listing is None:
        return None
    stale: dict[str, list[str]] = {}
    for line in listing.stdout.splitlines():
        name, _, image = line.partition("\t")
        if image and image != target:
            stale.setdefault(image, []).append(name)
    if not stale:
        return None
    current = max(stale, key=lambda img: len(stale[img]))
    return PendingSwitch(current_image=current, target_image=target,
                         containers=sorted(n for names in stale.values() for n in names))


def _image_pinned(terminal_cfg: dict) -> bool:
    """Same verdict the runtime uses. The file's own key first; then the bound terminal scope
    (a routed profile: recomputed per turn, never the launch profile's ``os.environ``); else the
    launch env, where an operator's ``TERMINAL_DOCKER_IMAGE`` differing from the effective value
    is their choice."""
    from hermes_cli.config import read_raw_config
    from tools.terminal_scope import get_terminal_scope
    raw = read_raw_config().get("terminal")
    if isinstance(raw, dict) and "docker_image" in raw:
        return True
    scope = get_terminal_scope()
    if scope is not None:
        return scope.get("TERMINAL_DOCKER_IMAGE_PINNED") == "1"
    env_image = os.environ.get("TERMINAL_DOCKER_IMAGE")
    return bool(env_image) and env_image != str(terminal_cfg.get("docker_image") or "")


def decide(switch: PendingSwitch, approve: bool) -> str:
    """Pin ``terminal.docker_image`` to the target (approve) or the current image (keep) and
    re-bridge the live process so a long-running gateway/TUI backend acts on it without a
    restart. Returns the image now pinned."""
    from hermes_cli.config import apply_terminal_config_to_env, load_config, save_config
    image = switch.target_image if approve else switch.current_image
    config = load_config()
    terminal = config.setdefault("terminal", {})
    terminal["docker_image"] = image
    save_config(config, preserve_keys={("terminal", "docker_image")})
    apply_terminal_config_to_env(env=None, override=True)
    if approve:
        # The cached environment object still points at the old container; drop it so the next
        # terminal call reattaches, sees the pin and recreates. The container itself keeps running
        # until then (persistent), nothing is removed here.
        from tools.terminal_tool_lifecycle import _evict_environment_for_task
        _evict_environment_for_task(None)
    return image


def explain_lines(switch: PendingSwitch) -> list[str]:
    n = len(switch.containers)
    return [
        f"Your Docker sandbox ({n} container{'s' if n != 1 else ''}) runs {switch.current_image}.",
        f"The default sandbox image is now {switch.target_image}: Python 3.13, Node 26, and a desktop",
        "so Bot Screen, computer_use and the browser run inside the sandbox instead of on this machine.",
        "Switching recreates the container on the next terminal call. Files in /root and /workspace",
        "stay (they live on this machine); packages installed with apt/pip/npm -g inside the container",
        "are reinstalled on demand; Python 3.11 virtualenvs need a rebuild.",
    ]


def offer_interactive(*, cprint, ask=input) -> Optional[bool]:
    """TTY startup offer. Either answer pins an image, so the question is asked once. Returns the
    decision, or None when nothing was pending / the user skipped (asked again next start)."""
    switch = pending()
    if switch is None:
        return None
    cprint("")
    cprint("☤ A new default sandbox image is available.")
    for line in explain_lines(switch):
        cprint(f"  {line}")
    try:
        answer = ask("  Switch now? [y = switch / n = keep the current image / Enter = ask later]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return None
    if answer in {"y", "yes"}:
        decide(switch, approve=True)
        cprint(f"  ✓ terminal.docker_image = {switch.target_image}; the sandbox is recreated on its next use.")
        return True
    if answer in {"n", "no"}:
        decide(switch, approve=False)
        cprint(f"  ✓ terminal.docker_image = {switch.current_image}; unset it in config.yaml to be asked again.")
        return False
    cprint("  Later. `hermes config set terminal.docker_image <image>` decides it any time.")
    return None
