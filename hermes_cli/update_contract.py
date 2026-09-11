"""Image-managed install refusal contract.

A refusal prints the real update command for the deployment kind, records a ``refused`` receipt (so
fleet tooling sees "this install cannot self-update, use <command>" instead of a silent non-update),
and exits 2 on CLI surfaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpdateRefusal:
    """Why an in-place update is refused, and what to run instead."""

    code: str              # image-marker | image-marker-invalid | docker | nix | apt
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


def evaluate_update_admission(project_root: Path) -> Optional[UpdateRefusal]:
    """Return an :class:`UpdateRefusal` when in-place update must not run.

    ``None`` means the install is eligible for in-place update (git checkout or unknown-but-
    mutable). Never raises; on any internal error it falls back to the heuristic layer only.
    """
    # Resolve the install shape first: the source policy applies only to source trees.
    try:
        from hermes_cli.config import detect_install_method
        install_method = detect_install_method(project_root)
    except Exception as exc:
        logger.debug("Install-method detection failed: %s", exc)
        install_method = "unknown"

    # Layer 1: checkout-local external-owner marker — authoritative for source installs.
    try:
        from hermes_cli.source_update_policy import read_source_update_policy

        policy = read_source_update_policy(project_root) if install_method == "git" else None
        if policy is not None:
            command = policy.update_command or "the external release manager"
            reason = policy.error or "invalid policy marker"
            code = "source-policy-invalid" if not policy.valid else "source-policy"
            message = (
                "✗ This source installation is managed by an external release manager.\n"
                f"  In-place update is disabled ({reason}). Update it with:\n    {command}"
            )
            return UpdateRefusal(code=code, message=message, update_command=command)
    except Exception as exc:
        # An inability to inspect an explicitly requested policy must not turn
        # into permission to mutate a source checkout.
        logger.debug("Source update policy check failed: %s", exc)
        return _refusal("source-policy-invalid", "docker", lambda _: (
            "✗ The source installation update policy could not be verified.\n"
            "  In-place update is disabled; update it through the external release manager."
        ))

    # Layer 2: baked provenance marker — authoritative when present.
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

    # Layer 3: pre-existing filesystem heuristics, verbatim semantics.
    try:
        from hermes_cli.config import is_nix_install_method

        method = install_method
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
        record_step("admission", False, f"not updatable in place ({refusal.code}); use: {refusal.update_command}")
        finalize_update_receipt("refused", stop_reason=refusal.code)
    except Exception as exc:
        logger.debug("Could not record refusal receipt: %s", exc)
