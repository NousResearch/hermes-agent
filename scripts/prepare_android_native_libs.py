#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ANDROID_ABIS = ("arm64-v8a", "x86_64")

LAUNCHER_RENAMES = {
    "bash": "libhermes_android_bash.so",
    "llama-server": "libhermes_android_llama_server.so",
}

BIONIC_SPAWN_LAUNCHER_RENAMES = {
    "llama-server": "libhermes_android_llama_server_bionic_spawn.so",
}

NEEDED_RENAMES = {
    "libbusybox.so.1.37.0": "libbusybox.so",
    "libbz2.so.1.0": "libbz2.so",
    "libcrypto.so.3": "libcrypto.so",
    "libexpat.so.1": "libexpat.so",
    "libhistory.so.8": "libhistory.so",
    "liblzma.so.5": "liblzma.so",
    "libncursesw.so.6": "libncursesw.so",
    "libreadline.so.8": "libreadline.so",
    "libssl.so.3": "libssl.so",
    "libz.so.1": "libz.so",
}

ANDROID_SPAWN_NEEDED = b"libandroid-spawn.so\0"
BIONIC_LIBC_NEEDED = b"libc.so\0"


def packaged_library_name(name: str) -> str:
    if name == "libc++_shared.so":
        return ""
    if ".so." in name:
        return name.split(".so.", 1)[0] + ".so"
    if name.endswith(".so"):
        return name
    return ""


def patch_needed_names(path: Path) -> None:
    payload = path.read_bytes()
    patched = payload
    for old, new in NEEDED_RENAMES.items():
        old_bytes = old.encode("ascii") + b"\0"
        new_bytes = new.encode("ascii") + b"\0"
        if len(new_bytes) > len(old_bytes):
            raise ValueError(f"Replacement {new} is longer than {old}")
        patched = patched.replace(old_bytes, new_bytes + (b"\0" * (len(old_bytes) - len(new_bytes))))
    if patched != payload:
        path.write_bytes(patched)


def patch_android_spawn_needed_to_libc(path: Path) -> None:
    payload = path.read_bytes()
    if ANDROID_SPAWN_NEEDED not in payload:
        return
    replacement = BIONIC_LIBC_NEEDED + (b"\0" * (len(ANDROID_SPAWN_NEEDED) - len(BIONIC_LIBC_NEEDED)))
    path.write_bytes(payload.replace(ANDROID_SPAWN_NEEDED, replacement))


def copy_abi(linux_assets_dir: Path, output_dir: Path, abi: str) -> None:
    prefix = linux_assets_dir / "hermes-linux" / abi / "prefix"
    bin_dir = prefix / "bin"
    lib_dir = prefix / "lib"
    abi_output = output_dir / abi
    abi_output.mkdir(parents=True, exist_ok=True)

    for source_name, dest_name in LAUNCHER_RENAMES.items():
        source = bin_dir / source_name
        if source.is_file():
            shutil.copy2(source, abi_output / dest_name)

    for source_name, dest_name in BIONIC_SPAWN_LAUNCHER_RENAMES.items():
        source = bin_dir / source_name
        if source.is_file():
            destination = abi_output / dest_name
            shutil.copy2(source, destination)
            patch_android_spawn_needed_to_libc(destination)

    for source in sorted(lib_dir.glob("lib*.so*"), key=lambda item: item.name):
        if not source.is_file():
            continue
        dest_name = packaged_library_name(source.name)
        if not dest_name:
            continue
        shutil.copy2(source, abi_output / dest_name)

    for binary in abi_output.glob("lib*.so"):
        patch_needed_names(binary)


def prepare_native_libs(linux_assets_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for abi in ANDROID_ABIS:
        copy_abi(linux_assets_dir, output_dir, abi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Android-packaged native launcher libraries")
    parser.add_argument("--linux-assets-dir", required=True, help="Generated Hermes Linux assets directory")
    parser.add_argument("--output-dir", required=True, help="Generated jniLibs output directory")
    args = parser.parse_args()
    prepare_native_libs(
        linux_assets_dir=Path(args.linux_assets_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
