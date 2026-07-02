from __future__ import annotations

import os.path
import re
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from email.message import Message
from email.parser import Parser
from email.policy import EmailPolicy
from glob import iglob
from pathlib import Path
from textwrap import dedent
from zipfile import ZipFile

from packaging.tags import parse_tag

from .. import __version__
from .._metadata import generate_requirements
from ..wheelfile import WheelFile

egg_filename_re = re.compile(
    r"""
    (?P<name>.+?)-(?P<ver>.+?)
    (-(?P<pyver>py\d\.\d+)
     (-(?P<arch>.+?))?
    )?.egg$""",
    re.VERBOSE,
)
egg_info_re = re.compile(
    r"""
    ^(?P<name>.+?)-(?P<ver>.+?)
    (-(?P<pyver>py\d\.\d+)
    )?.egg-info/""",
    re.VERBOSE,
)
wininst_re = re.compile(
    r"\.(?P<platform>win32|win-amd64)(?:-(?P<pyver>py\d\.\d))?\.exe$"
)
pyd_re = re.compile(r"\.(?P<abi>[a-z0-9]+)-(?P<platform>win32|win_amd64)\.pyd$")
serialization_policy = EmailPolicy(
    utf8=True,
    mangle_from_=False,
    max_line_length=0,
)
GENERATOR = f"wheel {__version__}"


def convert_requires(requires: str, metadata: Message) -> None:
    extra: str | None = None
    requirements: dict[str | None, list[str]] = defaultdict(list)
    for line in requires.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            extra = line[1:-1]
            continue

        requirements[extra].append(line)

    for key, value in generate_requirements(requirements):
        metadata.add_header(key, value)


def convert_pkg_info(pkginfo: str, metadata: Message) -> None:
    parsed_message = Parser().parsestr(pkginfo)
    for key, value in parsed_message.items():
        key_lower = key.lower()
        if value == "UNKNOWN":
            continue

        if key_lower == "description":
            description_lines = value.splitlines()
            if description_lines:
                value = "\n".join(
                    (
                        description_lines[0].lstrip(),
                        dedent("\n".join(description_lines[1:])),
                        "\n",
                    )
                )
            else:
                value = "\n"

            metadata.set_payload(value)
        elif key_lower == "home-page":
            metadata.add_header("Project-URL", f"Homepage, {value}")
        elif key_lower == "download-url":
            metadata.add_header("Project-URL", f"Download, {value}")
        else:
            metadata.add_header(key, value)

    metadata.replace_header("Metadata-Version", "2.4")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower().replace("-", "_")


class ConvertSource(metaclass=ABCMeta):
    name: str
    version: str
    pyver: str = "py2.py3"
    abi: str = "none"
    platform: str = "any"
    metadata: Message

    @property
    def dist_info_dir(self) -> str:
        return f"{self.name}-{self.version}.dist-info"

    @abstractmethod
    def generate_contents(self) -> Iterator[tuple[str, bytes]]:
        pass


class EggFileSource(ConvertSource):
    def __init__(self, path: Path):
        if not (match := egg_filename_re.match(path.name)):
            raise ValueError(f"Invalid egg file name: {path.name}")

        # Binary wheels are assumed to be for CPython
        self.path = path
        self.name = normalize(match.group("name"))
        self.version = match.group("ver")
        if pyver := match.group("pyver"):
            self.pyver = pyver.replace(".", "")
            if arch := match.group("arch"):
                self.abi = self.pyver.replace("py", "cp")
                self.platform = normalize(arch)

        self.metadata = Message()

    def generate_contents(self) -> Iterator[tuple[str, bytes]]:
        with ZipFile(self.path, "r") as zip_file:
            for filename in sorted(zip_file.namelist()):
                # Skip pure directory entries
                if filename.endswith("/"):
                    continue

                # Handle files in the egg-info directory specially, selectively moving
                # them to the dist-info directory while converting as needed
                if filename.startswith("EGG-INFO/"):
                    if filename == "EGG-INFO/requires.txt":
                        requires = zip_file.read(filename).decode("utf-8")
                        convert_requires(requires, self.metadata)
                    elif filename == "EGG-INFO/PKG-INFO":
                        pkginfo = zip_file.read(filename).decode("utf-8")
                        convert_pkg_info(pkginfo, self.metadata)
                    elif filename == "EGG-INFO/entry_points.txt":
                        yield (
                            f"{self.dist_info_dir}/entry_points.txt",
                            zip_file.read(filename),
                        )

                    continue

                # For any other file, just pass it through
                yield filename, zip_file.read(filename)


class EggDirectorySource(EggFileSource):
    def generate_contents(self) -> Iterator[tuple[str, bytes]]:
        for dirpath, _, filenames in os.walk(self.path):
            for filename in sorted(filenames):
                path = Path(dirpath, filename)
                if path.parent.name == "EGG-INFO":
                    if path.name == "requires.txt":
                        requires = path.read_text("utf-8")
                        convert_requires(requires, self.metadata)
                    elif path.name == "PKG-INFO":
                        pkginfo = path.read_text("utf-8")
                        convert_pkg_info(pkginfo, self.metadata)
                        if name := self.metadata.get("Name"):
                            self.name = normalize(name)

                        if version := self.metadata.get("Version"):
                            self.version = version
                    elif path.name == "entry_points.txt":
                        yield (
                            f"{self.dist_info_dir}/entry_points.txt",
                            path.read_bytes(),
                        )

                    continue

                # For any other file, just pass it through
                yield str(path.relative_to(self.path)), path.read_bytes()


class WininstFileSource(ConvertSource):
    """
    Handles distributions created with ``bdist_wininst``.

    The egginfo filename has the format::

        name-ver(-pyver)(-arch).egg-info

    The installer filename has the format::

        name-ver.arch(-pyver).exe

    Some things to note:

    1. The installer filename is not definitive. An installer can be renamed
       and work perfectly well as an installer. So more reliable data should
       be used whenever possible.
    2. The egg-info data should be preferred for the name and version, because
       these come straight from the distutils metadata, and are mandatory.
    3. The pyver from the egg-info data should be ignored, as it is
       constructed from the version of Python used to build the installer,
       which is irrelevant - the installer filename is correct here (even to
       the point that when it's not there, any version is implied).
    4. The architecture must be taken from the installer filename, as it is
       not included in the egg-info data.
    5. Architecture-neutral installers still have an architecture because the
       installer format itself (being executable) is architecture-specific. We
       should therefore ignore the architecture if the content is pure-python.
    """

    def __init__(self, path: Path):
        self.path = path
        self.metadata = Message()

        # Determine the initial architecture and Python version from the file name
        # (if possible)
        if match := wininst_re.search(path.name):
            self.platform = normalize(match.group("platform"))
            if pyver := match.group("pyver"):
                self.pyver = pyver.replace(".", "")

        # Look for an .egg-info directory and any .pyd files for more precise info
        egg_info_found = pyd_found = False
        with ZipFile(self.path) as zip_file:
            for filename in zip_file.namelist():
                prefix, filename = filename.split("/", 1)
                if not egg_info_found and (match := egg_info_re.match(filename)):
                    egg_info_found = True
                    self.name = normalize(match.group("name"))
                    self.version = match.group("ver")
                    if pyver := match.group("pyver"):
                        self.pyver = pyver.replace(".", "")
                elif not pyd_found and (match := pyd_re.search(filename)):
                    pyd_found = True
                    self.abi = match.group("abi")
                    self.platform = match.group("platform")

                if egg_info_found and pyd_found:
                    break

    def generate_contents(self) -> Iterator[tuple[str, bytes]]:
        dist_info_dir = f"{self.name}-{self.version}.dist-info"
        data_dir = f"{self.name}-{self.version}.data"
        with ZipFile(self.path, "r") as zip_file:
            for filename in sorted(zip_file.namelist()):
                # Skip pure directory entries
                if filename.endswith("/"):
                    continue

                # Handle files in the egg-info directory specially, selectively moving
                # them to the dist-info directory while converting as needed
                prefix, target_filename = filename.split("/", 1)
                if egg_info_re.search(target_filename):
                    basename = target_filename.rsplit("/", 1)[-1]
                    if basename == "requires.txt":
                        requires = zip_file.read(filename).decode("utf-8")
                        convert_requires(requires, self.metadata)
                    elif basename == "PKG-INFO":
                        pkginfo = zip_file.read(filename).decode("utf-8")
                        convert_pkg_info(pkginfo, self.metadata)
                    elif basename == "entry_points.txt":
                        yield (
                            f"{dist_info_dir}/entry_points.txt",
                            zip_file.read(filename),
                        )

                    continue
                elif prefix == "SCRIPTS":
                    target_filename = f"{data_dir}/scripts/{target_filename}"

                # For any other file, just pass it through
                yield target_filename, zip_file.read(filename)


def convert(files: list[str], dest_dir: str, verbose: bool) -> None:
    for pat in files:
        for archive in iglob(pat):
            path = Path(archive)
            if path.suffix == ".egg":
                if path.is_dir():
                    source: ConvertSource = EggDirectorySource(path)
                else:
                    source = EggFileSource(path)
            else:
                source = WininstFileSource(path)

            if verbose:
                print(f"{archive}...", flush=True, end="")

            dest_path = Path(dest_dir) / (
                f"{source.name}-{source.version}-{source.pyver}-{source.abi}"
                f"-{source.platform}.whl"
            )
            with WheelFile(dest_path, "w") as wheelfile:
                for name_or_zinfo, contents in source.generate_contents():
                    wheelfile.writestr(name_or_zinfo, contents)

                # Write the METADATA file
                wheelfile.writestr(
                    f"{source.dist_info_dir}/METADATA",
                    source.metadata.as_string(policy=serialization_policy).encode(
                        "utf-8"
                    ),
                )

                # Write the WHEEL file
                wheel_message = Message()
                wheel_message.add_header("Wheel-Version", "1.0")
                wheel_message.add_header("Generator", GENERATOR)
                wheel_message.add_header(
                    "Root-Is-Purelib", str(source.platform == "any").lower()
                )
                tags = parse_tag(f"{source.pyver}-{source.abi}-{source.platform}")
                for tag in sorted(tags, key=lambda tag: tag.interpreter):
                    wheel_message.add_header("Tag", str(tag))

                wheelfile.writestr(
                    f"{source.dist_info_dir}/WHEEL",
                    wheel_message.as_string(policy=serialization_policy).encode(
                        "utf-8"
                    ),
                )

            if verbose:
                print("OK")
