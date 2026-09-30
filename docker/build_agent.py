"""Supply the image's fixed runtime paths to the shared offline assembler."""
from __future__ import annotations

from pathlib import Path
import sysconfig

from scripts.build.agent import assemble
from scripts.build.inputs import AgentInputs, RESOURCE_ENV
from pm.store import current_target


def assemble_image(root: Path) -> None:
    environment = root / ".venv"
    site = Path(sysconfig.get_path("purelib", vars={"base": str(environment), "platbase": str(environment)}))
    manifest = assemble(AgentInputs(
        project=root / "pyproject.toml", code=root, repo=".", placement="fixed",
        target=current_target(), python=(environment / "bin/python").absolute(),
        site_packages=site, environment=environment, tools=root / "tools",
        pm_runtime=root / "pm-runtime", bin_dir="libexec",
        resources={name: root / name for name in RESOURCE_ENV},
        frontends={"tui": root / "ui-tui", "web": root / "hermes_cli/web_dist"},
    ), root)
    # Preserve the venv command paths used by s6 and the privilege-drop shim.
    for name, command in manifest["runtime"]["commands"].items():
        link = environment / "bin" / name
        link.unlink(missing_ok=True)
        link.symlink_to(f"../../{command}")


if __name__ == "__main__":
    assemble_image(Path("/opt/hermes"))
