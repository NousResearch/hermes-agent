"""Missing Linux native-build toolchain diagnosis for desktop dependency installs (#102081).

node-pty (the desktop workspace's terminal backend) ships prebuilds for Windows
and macOS only, so its install script always falls back to a local node-gyp
build on Linux — which needs ``make`` and a C++ compiler on PATH. When either
is missing the install dies deep inside a wall of npm log text whose only
actionable line is ``make: g++: No such file or directory`` (exit 127), and
the generic "run npm ci manually" advice cannot help: the manual run fails the
same way. The failure must name the missing tools and the distro-specific
install command instead.
"""

from __future__ import annotations

import shutil
import sys
from typing import Optional

# The two commands a node-gyp build invokes; anything fancier (clang wrappers,
# distro cc aliases) is out of scope — the hint only needs to catch the common
# bare-minimal server/container installs the reports come from.
REQUIRED_TOOLS = ("make", "g++")

_INSTALL_HINTS = (
    ("apt", "sudo apt install -y build-essential"),          # Debian / Ubuntu
    ("dnf", "sudo dnf install -y gcc-c++ make"),             # Fedora / RHEL
    ("yum", "sudo yum install -y gcc-c++ make"),             # older RHEL
    ("pacman", "sudo pacman -S --needed base-devel"),        # Arch
    ("zypper", "sudo zypper install -y gcc-c++ make"),       # openSUSE
    ("apk", "sudo apk add build-base"),                       # Alpine
)


def missing_native_build_tools(tools: tuple[str, ...] = REQUIRED_TOOLS,
                               which=shutil.which) -> tuple[str, ...]:
    """Names of the native-build tools missing from PATH, in declaration order."""
    return tuple(tool for tool in tools if which(tool) is None)


def linux_native_build_toolchain_hint(
    platform: Optional[str] = None, which=None,
) -> Optional[str]:
    """One-shot actionable hint when a native build cannot run on this Linux host.

    ``None`` when the platform is not Linux or the toolchain is present —
    a healthy host must not grow extra output. Failure exit codes and the
    surrounding behavior are the caller's business; this only supplies text.
    """
    if (platform if platform is not None else sys.platform) != "linux":
        return None
    missing = missing_native_build_tools(which=which if which is not None else shutil.which)
    if not missing:
        return None
    listed = ", ".join(missing)
    lines = [
        f"  ⚠ Native modules such as node-pty ship no Linux prebuilds and must",
        f"    be compiled locally, but these are missing from PATH: {listed}.",
        "    Install a C++ toolchain (or your distro's equivalent), e.g.:",
    ]
    lines.extend(f"      {hint}" for _, hint in _INSTALL_HINTS)
    lines.append("    then run this command again.")
    return "\n".join(lines)
