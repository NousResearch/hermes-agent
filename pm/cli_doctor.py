"""Doctor reports for dependencies owned by an external package builder."""
from __future__ import annotations

from pm.install_external import external_drift, external_packages


def report_external_packages() -> tuple[set[str], int]:
    """Use the startup verdict without inventing PM identities for Nix inputs."""
    packages = external_packages()
    problems = external_drift(packages)
    for name, executable in packages.items():
        reason = problems.get(name)
        if reason:
            print(f"✗ {name}: {reason}")
        else:
            print(f"✓ {name}: externally managed ({executable})")
    return set(packages), len(problems)
