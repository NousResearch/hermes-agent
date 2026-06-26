#!/usr/bin/env python3
"""Unit tests for snap/flatpak confinement path logic in vaultkit_lib.

Pins the empirically-verified snap rule: a confined keepassxc-cli can reach
NON-HIDDEN files under $HOME (and ~/snap, /media, /mnt) but NOT hidden
dot-directories (~/.local, ~/.config, ~/.cache) or paths outside $HOME
(/tmp, /run/user). Run: python3 tests/test_confinement.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import vaultkit_lib as vk  # noqa: E402

HOME = Path.home()
fails = []


def check(desc, got, want):
    if got != want:
        fails.append(f"{desc}: got {got!r}, want {want!r}")


r = vk._path_reachable_by_confined_cli
# Reachable under confinement:
check("non-hidden under home", r(HOME / "vaultkit/vault.kdbx"), True)
check("home root file", r(HOME / "vault.kdbx"), True)
check("~/snap tree", r(HOME / "snap/keepassxc/common/v.kdbx"), True)
check("removable /media", r(Path("/media/usb/v.kdbx")), True)
check("removable /mnt", r(Path("/mnt/disk/v.kdbx")), True)
# NOT reachable under confinement:
check("hidden ~/.local", r(HOME / ".local/share/vaultkit/v.kdbx"), False)
check("hidden ~/.config", r(HOME / ".config/vaultkit/v.kdbx"), False)
check("hidden ~/.cache", r(HOME / ".cache/v.kdbx"), False)
check("outside home /tmp", r(Path("/tmp/v.kdbx")), False)
check("outside home /run/user", r(Path("/run/user/1000/v.kdbx")), False)
check("outside home /etc", r(Path("/etc/v.kdbx")), False)

if fails:
    print("FAIL:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PASS — confinement path rules (11 cases): hidden dot-dirs and "
      "out-of-home paths correctly rejected; non-hidden home + ~/snap + "
      "removable media allowed")
