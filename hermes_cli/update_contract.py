"""Image-managed install refusal contract.

A refusal prints the real update command for the deployment kind, records a
``refused`` receipt (so fleet tooling sees "this install cannot self-update,
use <command>" instead of a silent non-update), and exits 2 on CLI surfaces.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from hermes_cli.steward import STEWARD_APT_TERMUX, STEWARD_DESKTOP, STEWARD_DOCKER, STEWARD_NIX

logger = logging.getLogger(__name__)

COMMIT_BUILD_UPDATE_MESSAGE = (
    "This build doesn't get updates. Ask the developer who gave it to you for a new build."
)


def is_commit_build(project_root: Path) -> bool:
    from hermes_cli.steward import read_install_stamp

    return read_install_stamp(project_root).get("source") == "commit-build"


@dataclass(frozen=True)
class UpdateRefusal:
    """Why an in-place update is refused, and what to run instead."""

    code: str              # image-marker | image-marker-invalid | docker | nix | apt | desktop-app | <steward>
    message: str           # full user-facing text (multi-line ok)
    update_command: str    # the one-line remediation command


def _refusal(code: str, method: str, message: Optional[Callable[[str], str]] = None) -> UpdateRefusal:
    """Refusal for ``method``: ``message(command)`` if given, else docker's full message / the bare command."""
    from hermes_cli.config import format_docker_update_message, recommended_update_command_for_method

    command = recommended_update_command_for_method(method)
    if message is not None:
        text = message(command)
    else:
        text = format_docker_update_message() if method == "docker" else command
    return UpdateRefusal(code=code, message=text, update_command=command)


# Sealed-tree steward -> install method whose CLI command remediates it. The
# refusal code is the steward name itself. APT owns the Termux code tree:
# ``pkg upgrade`` replaces it wholesale, so it must never update in place.
_STEWARD_UPDATE_METHODS: dict[str, str] = {
    STEWARD_DOCKER: "docker",
    STEWARD_NIX: "nix",
    STEWARD_APT_TERMUX: "apt",
}


def _steward_refusal(steward: str) -> UpdateRefusal:
    """Refusal for a tree sealed by ``steward``."""
    from hermes_cli.steward import steward_update_message

    method = _STEWARD_UPDATE_METHODS.get(steward)
    if method == "docker":
        return _refusal(steward, method)
    if method is not None:
        return _refusal(steward, method, lambda _command: steward_update_message(steward))
    # desktop-app and future package managers have no CLI remediation: the
    # steward's own instructions are the remediation, and
    # recommended_update_command_for_method would falsely answer "hermes
    # update" for methods it doesn't know.
    command = (
        "Manage updates from within the desktop app"
        if steward == STEWARD_DESKTOP
        else f"update via {steward}"
    )
    return UpdateRefusal(code=steward, message=steward_update_message(steward), update_command=command)


UPDATE_HOLDS_DIRNAME = "hermes-update-holds"
_HOLD_MESSAGE_LIMIT = 2000


def update_holds_dir(project_root: Path) -> Optional[Path]:
    """Where integrations that patch this checkout hold its updates: ``<git common dir>/hermes-update-holds``.

    The holds live beside the checkout lock, outside the working tree, so they are neither
    stashed by an update nor reported as local changes. ``None`` when ``project_root`` is no checkout.
    """
    from hermes_cli.update_lock import _git_common_dir

    common = _git_common_dir(Path(project_root))
    return None if common is None else common / UPDATE_HOLDS_DIRNAME


def _hold_refusal(project_root: Path) -> Optional[UpdateRefusal]:
    """Refuse while any integration holds updates for this checkout.

    Each hold is ``<holder>.json`` with ``{"holder": str, "message": str}``. An integration that
    patches the checkout places one until it has qualified the next Hermes version, because an
    update would carry its patches onto an untested base or park them. A hold that cannot be read
    still refuses: an integrity defect is never permission to update.
    """
    directory = update_holds_dir(project_root)
    if directory is None or not (directory.exists() or directory.is_symlink()):
        return None
    if directory.is_symlink() or not directory.is_dir():
        return UpdateRefusal("update-hold-invalid", f"✗ In-place update is held, but {directory} is not a directory.", "")
    holds = []
    for path in sorted(directory.iterdir()):
        if path.suffix != ".json":
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            holder, message = value["holder"], value["message"]
            if (not isinstance(holder, str) or not holder.strip() or not isinstance(message, str)
                    or not message.strip() or len(holder) + len(message) > _HOLD_MESSAGE_LIMIT):
                raise ValueError("hold fields must be bounded text")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return UpdateRefusal(
                "update-hold-invalid",
                f"✗ In-place update is held, but the hold {path} cannot be read ({exc}).",
                "",
            )
        holds.append(f"  {holder.strip()}: {message.strip()}")
    if not holds:
        return None
    return UpdateRefusal("update-hold", "✗ In-place update is held by an integration that patches this install:\n"
                         + "\n".join(holds), "")


def evaluate_update_admission(project_root: Path) -> Optional[UpdateRefusal]:
    """Return an :class:`UpdateRefusal` when in-place update must not run.

    ``None`` means the install is eligible for in-place update (git checkout or unknown-but-
    mutable). Never raises; on any internal error it falls back to the heuristic layer only.
    An update hold (layer 0) fails closed instead: an unreadable hold still refuses.
    """
    try:
        held = _hold_refusal(project_root)
    except OSError as exc:
        held = UpdateRefusal("update-hold-invalid", f"✗ In-place update holds cannot be checked ({exc}).", "")
    if held is not None:
        return held

    if is_commit_build(project_root):
        return UpdateRefusal("commit-build", COMMIT_BUILD_UPDATE_MESSAGE, "")

    # Layer 1: baked provenance marker — authoritative when present.
    try:
        from hermes_cli.image_provenance import read_image_provenance

        provenance = read_image_provenance()
        if provenance is not None:
            if not provenance.valid:
                # Present but malformed: still image-managed — an integrity defect is never
                # permission to mutate the image in place.
                return _refusal("image-marker-invalid", "docker", lambda command: (
                    "✗ This install is image-managed, but its provenance "
                    f"marker is invalid ({provenance.error}).\n"
                    "  In-place update is disabled. Update by pulling a "
                    f"new image:\n    {command}"
                ))
            return _refusal("image-marker", provenance.manager)
    except Exception as exc:
        logger.debug("Image provenance check failed (using heuristics): %s", exc)

    # Layer 2: install stamp / steward classification. A sealed tree (no
    # ``.git``) belongs to a steward — the desktop app bundle, a Docker
    # image, the Nix store — and only the steward updates it. This is the
    # rung that covers ``desktop-app``, which the heuristics below never
    # detect (the payload has no .install_method stamp and no .git).
    try:
        from hermes_cli.steward import sealed_steward

        steward = sealed_steward(project_root)
        if steward is None:
            from hermes_constants import is_termux

            if is_termux():
                from hermes_cli.steward import SOURCE_ON_TERMUX_UPDATE_COMMAND, SOURCE_ON_TERMUX_UPDATE_MESSAGE

                return UpdateRefusal(
                    code=STEWARD_APT_TERMUX,
                    message=SOURCE_ON_TERMUX_UPDATE_MESSAGE,
                    update_command=SOURCE_ON_TERMUX_UPDATE_COMMAND,
                )
        elif steward != "unknown":
            return _steward_refusal(steward)
    except Exception as exc:
        logger.debug("Steward admission check failed: %s", exc)

    # Layer 3: pre-existing filesystem heuristics, verbatim semantics.
    try:
        from hermes_cli.config import detect_install_method, is_nix_install_method

        method = detect_install_method(project_root)
        if method == "docker":
            return _refusal("docker", method)
        if is_nix_install_method(method) or method == "apt":
            return _refusal(method if method == "apt" else "nix", method)
    except Exception as exc:
        logger.debug("Install-method admission check failed: %s", exc)
    return None


def record_refusal_receipt(refusal: UpdateRefusal) -> None:
    """Write a minimal ``refused`` receipt for a blocked update attempt.

    Gives fleet tooling a durable record that an update was ATTEMPTED and refused ("not updatable in
    place, use <command>") instead of a silent nothing. Best-effort; never raises.
    """
    try:
        from hermes_cli.update_receipt import begin_update_receipt, finalize_update_receipt, record_step

        begin_update_receipt()
        detail = f"not updatable in place ({refusal.code})"
        detail += f"; use: {refusal.update_command}" if refusal.update_command else f"; {refusal.message}"
        record_step("admission", False, detail)
        finalize_update_receipt("refused", stop_reason=refusal.code)
    except Exception as exc:
        logger.debug("Could not record refusal receipt: %s", exc)
