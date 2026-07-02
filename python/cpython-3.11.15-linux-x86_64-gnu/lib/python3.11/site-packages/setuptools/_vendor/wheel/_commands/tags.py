from __future__ import annotations

import email.policy
import itertools
import os
from collections.abc import Iterable
from email.parser import BytesParser

from ..wheelfile import WheelFile


def _compute_tags(original_tags: Iterable[str], new_tags: str | None) -> set[str]:
    """Add or replace tags. Supports dot-separated tags"""
    if new_tags is None:
        return set(original_tags)

    if new_tags.startswith("+"):
        return {*original_tags, *new_tags[1:].split(".")}

    if new_tags.startswith("-"):
        return set(original_tags) - set(new_tags[1:].split("."))

    return set(new_tags.split("."))


def tags(
    wheel: str,
    python_tags: str | None = None,
    abi_tags: str | None = None,
    platform_tags: str | None = None,
    build_tag: str | None = None,
    remove: bool = False,
) -> str:
    """Change the tags on a wheel file.

    The tags are left unchanged if they are not specified. To specify "none",
    use ["none"]. To append to the previous tags, a tag should start with a
    "+".  If a tag starts with "-", it will be removed from existing tags.
    Processing is done left to right.

    :param wheel: The paths to the wheels
    :param python_tags: The Python tags to set
    :param abi_tags: The ABI tags to set
    :param platform_tags: The platform tags to set
    :param build_tag: The build tag to set
    :param remove: Remove the original wheel
    """
    with WheelFile(wheel, "r") as f:
        assert f.filename, f"{f.filename} must be available"

        wheel_info = f.read(f.dist_info_path + "/WHEEL")
        info = BytesParser(policy=email.policy.compat32).parsebytes(wheel_info)

        original_wheel_name = os.path.basename(f.filename)
        namever = f.parsed_filename.group("namever")
        build = f.parsed_filename.group("build")
        original_python_tags = f.parsed_filename.group("pyver").split(".")
        original_abi_tags = f.parsed_filename.group("abi").split(".")
        original_plat_tags = f.parsed_filename.group("plat").split(".")

    tags: list[str] = info.get_all("Tag", [])
    existing_build_tag = info.get("Build")

    impls = {tag.split("-")[0] for tag in tags}
    abivers = {tag.split("-")[1] for tag in tags}
    platforms = {tag.split("-")[2] for tag in tags}

    if impls != set(original_python_tags):
        msg = f"Wheel internal tags {impls!r} != filename tags {original_python_tags!r}"
        raise AssertionError(msg)

    if abivers != set(original_abi_tags):
        msg = f"Wheel internal tags {abivers!r} != filename tags {original_abi_tags!r}"
        raise AssertionError(msg)

    if platforms != set(original_plat_tags):
        msg = (
            f"Wheel internal tags {platforms!r} != filename tags {original_plat_tags!r}"
        )
        raise AssertionError(msg)

    if existing_build_tag != build:
        msg = (
            f"Incorrect filename '{build}' "
            f"& *.dist-info/WHEEL '{existing_build_tag}' build numbers"
        )
        raise AssertionError(msg)

    # Start changing as needed
    if build_tag is not None:
        build = build_tag

    final_python_tags = sorted(_compute_tags(original_python_tags, python_tags))
    final_abi_tags = sorted(_compute_tags(original_abi_tags, abi_tags))
    final_plat_tags = sorted(_compute_tags(original_plat_tags, platform_tags))

    final_tags = [
        namever,
        ".".join(final_python_tags),
        ".".join(final_abi_tags),
        ".".join(final_plat_tags),
    ]
    if build:
        final_tags.insert(1, build)

    final_wheel_name = "-".join(final_tags) + ".whl"

    if original_wheel_name != final_wheel_name:
        del info["Tag"], info["Build"]
        for a, b, c in itertools.product(
            final_python_tags, final_abi_tags, final_plat_tags
        ):
            info["Tag"] = f"{a}-{b}-{c}"
        if build:
            info["Build"] = build

        original_wheel_path = os.path.join(
            os.path.dirname(f.filename), original_wheel_name
        )
        final_wheel_path = os.path.join(os.path.dirname(f.filename), final_wheel_name)

        with (
            WheelFile(original_wheel_path, "r") as fin,
            WheelFile(final_wheel_path, "w") as fout,
        ):
            fout.comment = fin.comment  # preserve the comment
            for item in fin.infolist():
                if item.is_dir():
                    continue
                if item.filename == f.dist_info_path + "/RECORD":
                    continue
                if item.filename == f.dist_info_path + "/WHEEL":
                    fout.writestr(item, info.as_bytes())
                else:
                    fout.writestr(item, fin.read(item))

        if remove:
            os.remove(original_wheel_path)

    return final_wheel_name
