"""Image-managed install refusal contract (#91277 Phase 3).

One shared admission gate for every surface that can start an in-place
``hermes update`` mutation (CLI apply, CLI --check, dashboard update
endpoint). The decision layers:

1. **Baked provenance marker** (``/etc/hermes/image-provenance.json``,
   written by the image build — see :mod:`hermes_cli.image_provenance`):
   authoritative ground truth that this filesystem came from an immutable
   image. Fail-closed: a present-but-malformed marker still refuses.
2. **Filesystem heuristics** (``detect_install_method()``): the pre-existing
   docker/nix/apt detection, kept as the fallback for images built before
   the marker existed and for package-managed installs that have no image
   marker at all.

A refusal prints the real update command for the deployment kind, records a
``refused`` receipt (so fleet tooling sees "this install cannot self-update,
use <command>" instead of a silent non-update), and exits 2 on CLI surfaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpdateRefusal:
    """Why an in-place update is refused, and what to run instead."""

    code: str              # image-marker | image-marker-invalid | docker | nix | apt
    message: str           # full user-facing text (multi-line ok)
    update_command: str    # the one-line remediation command


def evaluate_update_admission(project_root: Path) -> Optional[UpdateRefusal]:
    """Return an :class:`UpdateRefusal` when in-place update must not run.

    ``None`` means the install is eligible for in-place update (git checkout
    or unknown-but-mutable). Never raises; on any internal error it falls
    back to the heuristic layer only.
    """
    # Layer 1: baked provenance marker — authoritative when present.
    try:
        from hermes_cli.image_provenance import read_image_provenance

        provenance = read_image_provenance()
        if provenance is not None:
            from hermes_cli.config import (
                format_docker_update_message,
                recommended_update_command_for_method,
            )

            if not provenance.valid:
                # Present but malformed: still image-managed — an integrity
                # defect is never permission to mutate the image in place.
                command = recommended_update_command_for_method("docker")
                return UpdateRefusal(
                    code="image-marker-invalid",
                    message=(
                        "✗ This install is image-managed, but its provenance "
                        f"marker is invalid ({provenance.error}).\n"
                        "  In-place update is disabled. Update by pulling a "
                        f"new image:\n    {command}"
                    ),
                    update_command=command,
                )
            manager = provenance.manager
            if manager == "docker":
                return UpdateRefusal(
                    code="image-marker",
                    message=format_docker_update_message(),
                    update_command=recommended_update_command_for_method("docker"),
                )
            command = recommended_update_command_for_method(manager)
            return UpdateRefusal(
                code="image-marker",
                message=command,
                update_command=command,
            )
    except Exception as exc:
        logger.debug("Image provenance check failed (using heuristics): %s", exc)

    # Layer 2: pre-existing filesystem heuristics, verbatim semantics.
    try:
        from hermes_cli.config import (
            detect_install_method,
            format_docker_update_message,
            is_nix_install_method,
            recommended_update_command_for_method,
        )

        method = detect_install_method(project_root)
        if method == "docker":
            return UpdateRefusal(
                code="docker",
                message=format_docker_update_message(),
                update_command=recommended_update_command_for_method("docker"),
            )
        if is_nix_install_method(method) or method == "apt":
            command = recommended_update_command_for_method(method)
            return UpdateRefusal(
                code=method if method == "apt" else "nix",
                message=command,
                update_command=command,
            )
    except Exception as exc:
        logger.debug("Install-method admission check failed: %s", exc)
    return None


def record_refusal_receipt(refusal: UpdateRefusal) -> None:
    """Write a minimal ``refused`` receipt for a blocked update attempt.

    Gives fleet tooling a durable record that an update was ATTEMPTED and
    refused ("not updatable in place, use <command>") instead of a silent
    nothing. Best-effort; never raises.
    """
    try:
        from hermes_cli.update_receipt import (
            begin_update_receipt,
            finalize_update_receipt,
            record_step,
        )

        begin_update_receipt()
        record_step(
            "admission",
            False,
            f"not updatable in place ({refusal.code}); use: {refusal.update_command}",
        )
        finalize_update_receipt("refused", stop_reason=refusal.code)
    except Exception as exc:
        logger.debug("Could not record refusal receipt: %s", exc)
