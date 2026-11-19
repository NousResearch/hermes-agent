"""Stage the payload's GPU engines and the floor builder's bootstrap on Ubuntu 24.04."""
from pathlib import Path
import platform

import pm
from pm.build_operations import verified_tools
from scripts.ci.setup_toolchain import file_commands


def main() -> None:
    os_release = platform.freedesktop_os_release()
    if (pm.current_target(), os_release.get("ID"), os_release.get("VERSION_ID")) != ("linux-x64", "ubuntu", "24.04"):
        raise ValueError("GPU preparation requires native x64 Ubuntu 24.04")
    root = Path.cwd() / "apps/desktop/build/agent-payload/tools"
    bootstrap = Path.cwd() / ".build/x64-bootstrap"
    for store in (root, bootstrap):
        if store != store.resolve() or store.exists():
            raise ValueError(f"x64 preparation requires a fresh job-local store: {store}")
    pm.prepare_tools(["llamacpp-cuda", "llamacpp-hip"], out=root, target="linux-x64")
    # Jammy's system Python cannot apply PM's tar extraction filter to its own replacement.
    pm.prepare_tools(["python"], out=bootstrap, target="linux-x64")
    python = verified_tools(["python"], source_store=bootstrap, target="linux-x64").entries["python"].binary
    if python is None:
        raise ValueError("PM did not stage an x64 bootstrap interpreter")
    file_commands("GITHUB_ENV", {"X64_BOOTSTRAP_PYTHON": python})


if __name__ == "__main__":
    main()
