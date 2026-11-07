"""Guards that keep persisted supervisor definitions launchable.

systemd units and launchd plists outlive the process that wrote them, so
their launch commands may only name roots that survive dependency GC and
answer for a committed environment — the install's checkout. The failure
class refused here is a definition launching from a dependency-generation
tree (#131164): a venv console script runs with ``PROJECT_ROOT`` inside
``installs/<key>/environments/<gen>/workspace``, whose own install key has
no committed environment, so a service persisted from there crash-loops
with "no dependency environment is committed" until the definition is
regenerated from the stable install root. (A temp HERMES_HOME in a
definition is the gateway_service_owner module's refusal.)
"""
from pathlib import Path


def _gw():
    from hermes_cli import gateway  # late: the facade imports this module
    return gateway


def _service_install_root(root: Path) -> Path:
    """The root a persisted supervisor command may name: the install's checkout.

    A generation's venv console script runs with PROJECT_ROOT inside
    ``installs/<key>/environments/<gen>/workspace`` — a GC-able tree keyed by
    its own path, with no committed dependency environment of its own, so a
    unit generated from there crash-loops with "no dependency environment is
    committed" (#131164). Map such a root back to the checkout that owns the
    generation; every other root passes through unchanged.
    """
    from pm.environments import owning_install_root

    return owning_install_root(root) or Path(root)


def _generation_tree_in_exec_lines(definition: str) -> str | None:
    """A dependency-generation tree named by an Exec directive, or ``None``.

    Matches ``installs/<key>/environments/<gen>/{workspace,venv}`` on Exec
    lines only: the PATH directive may legitimately carry a generation's bin
    dir for tooling, but no persisted launch command may.
    """
    import re

    for line in definition.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Exec"):
            continue
        match = re.search(r"/installs/[^/\s\"']+/environments/[^/\s\"']+/+(?:workspace|venv)(?:/|$)", stripped)
        if match:
            return match.group(0)
    return None


def _generation_launcher_in_plist(definition: str) -> str | None:
    """A launchd ProgramArguments entry launching from a generation tree, or ``None``.

    Tighter than the Exec-line scan: plists also serialize environment values,
    and a PATH may legitimately carry a generation's bin dir, so only the
    launcher binary itself — ``{workspace,venv}/(.hermes/)?bin/hermes`` — counts.
    """
    import re

    match = re.search(
        r"/installs/[^/<>\"'\s]+/environments/[^/<>\"'\s]+/+(?:workspace|venv)/+(?:\.hermes/)?bin/hermes\b",
        definition)
    return match.group(0) if match else None


def _refuse_generation_launcher_service_write(definition: str, kind: str) -> bool:
    """Refuse (with guidance) when a service definition would launch from a
    dependency-generation tree. The tree's install key matches nothing
    recorded, so the supervised service crash-loops with "no dependency
    environment is committed" until the definition is regenerated from the
    stable install root (#131164)."""
    tree = _generation_tree_in_exec_lines(definition) or _generation_launcher_in_plist(definition)
    if tree is None:
        return False
    print(f"✗ Refusing to write the gateway {kind}: its launch commands point into a dependency-generation tree ({tree.rstrip('/')}).")
    from pm.environments import owning_install_root

    project_root = _gw().PROJECT_ROOT
    stable = (owning_install_root(project_root) or project_root) / ".hermes" / "bin" / "hermes"
    print("  Generation workspaces are rebuilt and collected; supervisor commands must launch from the stable install root.")
    print(f"  Re-run the command from the stable launcher (for example: {stable} gateway install).")
    return True
