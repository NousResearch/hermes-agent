from __future__ import annotations

__all__ = ["WHEEL_INFO_RE", "WheelFile", "WheelError"]

import base64
import csv
import hashlib
import logging
import os.path
import re
import stat
import time
from io import StringIO, TextIOWrapper
from typing import IO, TYPE_CHECKING, Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

if TYPE_CHECKING:
    from _typeshed import SizedBuffer, StrPath


# Non-greedy matching of an optional build number may be too clever (more
# invalid wheel filenames will match). Separate regex for .dist-info?
WHEEL_INFO_RE = re.compile(
    r"""^(?P<namever>(?P<name>[^\s-]+?)-(?P<ver>[^\s-]+?))(-(?P<build>\d[^\s-]*))?
     -(?P<pyver>[^\s-]+?)-(?P<abi>[^\s-]+?)-(?P<plat>\S+)\.whl$""",
    re.VERBOSE,
)
MINIMUM_TIMESTAMP = 315532800  # 1980-01-01 00:00:00 UTC

log = logging.getLogger("wheel")


class WheelError(Exception):
    pass


def urlsafe_b64encode(data: bytes) -> bytes:
    """urlsafe_b64encode without padding"""
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def urlsafe_b64decode(data: bytes) -> bytes:
    """urlsafe_b64decode without padding"""
    pad = b"=" * (4 - (len(data) & 3))
    return base64.urlsafe_b64decode(data + pad)


def get_zipinfo_datetime(
    timestamp: float | None = None,
) -> tuple[int, int, int, int, int]:
    # Some applications need reproducible .whl files, but they can't do this without
    # forcing the timestamp of the individual ZipInfo objects. See issue #143.
    timestamp = int(os.environ.get("SOURCE_DATE_EPOCH", timestamp or time.time()))
    timestamp = max(timestamp, MINIMUM_TIMESTAMP)
    return time.gmtime(timestamp)[0:6]


class WheelFile(ZipFile):
    """A ZipFile derivative class that also reads SHA-256 hashes from
    .dist-info/RECORD and checks any read files against those.
    """

    _default_algorithm = hashlib.sha256

    def __init__(
        self,
        file: StrPath,
        mode: Literal["r", "w", "x", "a"] = "r",
        compression: int = ZIP_DEFLATED,
    ):
        basename = os.path.basename(file)
        self.parsed_filename = WHEEL_INFO_RE.match(basename)
        if not basename.endswith(".whl") or self.parsed_filename is None:
            raise WheelError(f"Bad wheel filename {basename!r}")

        ZipFile.__init__(self, file, mode, compression=compression, allowZip64=True)

        self.dist_info_path = "{}.dist-info".format(
            self.parsed_filename.group("namever")
        )
        self.record_path = self.dist_info_path + "/RECORD"
        self._file_hashes: dict[str, tuple[None, None] | tuple[int, bytes]] = {}
        self._file_sizes = {}
        if mode == "r":
            # Ignore RECORD and any embedded wheel signatures
            self._file_hashes[self.record_path] = None, None
            self._file_hashes[self.record_path + ".jws"] = None, None
            self._file_hashes[self.record_path + ".p7s"] = None, None

            # Fill in the expected hashes by reading them from RECORD
            try:
                record = self.open(self.record_path)
            except KeyError:
                raise WheelError(f"Missing {self.record_path} file") from None

            with record:
                for line in csv.reader(
                    TextIOWrapper(record, newline="", encoding="utf-8")
                ):
                    path, hash_sum, size = line
                    if not hash_sum:
                        continue

                    algorithm, hash_sum = hash_sum.split("=")
                    try:
                        hashlib.new(algorithm)
                    except ValueError:
                        raise WheelError(
                            f"Unsupported hash algorithm: {algorithm}"
                        ) from None

                    if algorithm.lower() in {"md5", "sha1"}:
                        raise WheelError(
                            f"Weak hash algorithm ({algorithm}) is not permitted by "
                            f"PEP 427"
                        )

                    self._file_hashes[path] = (
                        algorithm,
                        urlsafe_b64decode(hash_sum.encode("ascii")),
                    )

    def open(
        self,
        name_or_info: str | ZipInfo,
        mode: Literal["r", "w"] = "r",
        pwd: bytes | None = None,
    ) -> IO[bytes]:
        def _update_crc(newdata: bytes) -> None:
            eof = ef._eof
            update_crc_orig(newdata)
            running_hash.update(newdata)
            if eof and running_hash.digest() != expected_hash:
                raise WheelError(f"Hash mismatch for file '{ef_name}'")

        ef_name = (
            name_or_info.filename if isinstance(name_or_info, ZipInfo) else name_or_info
        )
        if (
            mode == "r"
            and not ef_name.endswith("/")
            and ef_name not in self._file_hashes
        ):
            raise WheelError(f"No hash found for file '{ef_name}'")

        ef = ZipFile.open(self, name_or_info, mode, pwd)
        if mode == "r" and not ef_name.endswith("/"):
            algorithm, expected_hash = self._file_hashes[ef_name]
            if expected_hash is not None:
                # Monkey patch the _update_crc method to also check for the hash from
                # RECORD
                running_hash = hashlib.new(algorithm)
                update_crc_orig, ef._update_crc = ef._update_crc, _update_crc

        return ef

    def write_files(self, base_dir: str) -> None:
        log.info("creating %r and adding %r to it", self.filename, base_dir)
        deferred: list[tuple[str, str]] = []
        for root, dirnames, filenames in os.walk(base_dir):
            # Sort the directory names so that `os.walk` will walk them in a
            # defined order on the next iteration.
            dirnames.sort()
            for name in sorted(filenames):
                path = os.path.normpath(os.path.join(root, name))
                if os.path.isfile(path):
                    arcname = os.path.relpath(path, base_dir).replace(os.path.sep, "/")
                    if arcname == self.record_path:
                        pass
                    elif root.endswith(".dist-info"):
                        deferred.append((path, arcname))
                    else:
                        self.write(path, arcname)

        deferred.sort()
        for path, arcname in deferred:
            self.write(path, arcname)

    def write(
        self,
        filename: str,
        arcname: str | None = None,
        compress_type: int | None = None,
    ) -> None:
        with open(filename, "rb") as f:
            st = os.fstat(f.fileno())
            data = f.read()

        zinfo = ZipInfo(
            arcname or filename, date_time=get_zipinfo_datetime(st.st_mtime)
        )
        zinfo.external_attr = (stat.S_IMODE(st.st_mode) | stat.S_IFMT(st.st_mode)) << 16
        zinfo.compress_type = compress_type or self.compression
        self.writestr(zinfo, data, compress_type)

    def writestr(
        self,
        zinfo_or_arcname: str | ZipInfo,
        data: SizedBuffer | str,
        compress_type: int | None = None,
    ) -> None:
        if isinstance(zinfo_or_arcname, str):
            zinfo_or_arcname = ZipInfo(
                zinfo_or_arcname, date_time=get_zipinfo_datetime()
            )
            zinfo_or_arcname.compress_type = self.compression
            zinfo_or_arcname.external_attr = (0o664 | stat.S_IFREG) << 16

        if isinstance(data, str):
            data = data.encode("utf-8")

        ZipFile.writestr(self, zinfo_or_arcname, data, compress_type)
        fname = (
            zinfo_or_arcname.filename
            if isinstance(zinfo_or_arcname, ZipInfo)
            else zinfo_or_arcname
        )
        log.info("adding %r", fname)
        if fname != self.record_path:
            hash_ = self._default_algorithm(data)
            self._file_hashes[fname] = (
                hash_.name,
                urlsafe_b64encode(hash_.digest()).decode("ascii"),
            )
            self._file_sizes[fname] = len(data)

    def close(self) -> None:
        # Write RECORD
        if self.fp is not None and self.mode == "w" and self._file_hashes:
            data = StringIO()
            writer = csv.writer(data, delimiter=",", quotechar='"', lineterminator="\n")
            writer.writerows(
                (
                    (fname, algorithm + "=" + hash_, self._file_sizes[fname])
                    for fname, (algorithm, hash_) in self._file_hashes.items()
                )
            )
            writer.writerow((format(self.record_path), "", ""))
            self.writestr(self.record_path, data.getvalue())

        ZipFile.close(self)
