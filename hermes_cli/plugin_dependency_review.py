"""Candidate-bound Python dependency review for non-interactive plugin installs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class DependencyConsentRequired(Exception):
    def __init__(self, dependencies: tuple[str, ...], token: str):
        super().__init__('This plugin can add Python packages to the shared Hermes environment. '
                         'Review the declared requirements before installing. '
                         'A Python package with no listed requirements may still run its build backend.')
        self.dependencies = dependencies
        self.token = token


def review_install_dependencies(staged: Path, target: Path, record: dict, *,
                                enable: bool, force: bool, accepted: str | None) -> None:
    """Review the scanned candidate, never the client probe or a moving Git ref.

    A retry may clone again. Bind the answer to source, revision, declarations,
    destination (including profile), and install intent so drift asks again.
    This authorizes only Python admission; PM and the scanner remain mandatory.
    """
    from pm.plugin_declarations import read_python_declaration
    declaration = read_python_declaration(staged)
    # Review even download-only installs: an independently enabled selection
    # can change while the candidate is being cloned/scanned.
    if not declaration.is_member:
        return
    identity = {
        'source': record['source'],
        'revision': record['revision'],
        'target': str(target.resolve()),
        'enable': enable,
        'force': force,
        'declarations': [(p.name, p.read_bytes().hex()) for p in declaration.files],
    }
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if accepted != token:
        raise DependencyConsentRequired(declaration.install_requirements, token)
