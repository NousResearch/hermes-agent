"""Determinism seams for golden-transcript refactor checks."""

from __future__ import annotations

import os
import re
import secrets
import sqlite3
import tempfile
import time
import uuid
from contextlib import ContextDecorator
from pathlib import Path
from typing import Callable, Iterable

FROZEN_EPOCH = 1_800_000_000.0
FROZEN_MONOTONIC = 500_000.0
FROZEN_SQL_DATETIME = "2027-01-15 08:00:00"
_SCAN = re.compile(r"monotonic|CURRENT_TIMESTAMP|datetime\('now'\)|AUTOINCREMENT")


class DeterminismError(RuntimeError):
    """Raised when a target has nondeterminism without a named seam."""


class Determinism(ContextDecorator):
    """Patch process-global nondeterminism for the duration of a capture/replay."""

    def __init__(self, hermes_home: str | os.PathLike[str] | None = None) -> None:
        self._tmp = None
        self.hermes_home = Path(hermes_home) if hermes_home else None
        self._restore: list[Callable[[], None]] = []
        self._uuid_count = 0
        self._token_count = 0
        self._sqlite_connect = sqlite3.connect

    def __enter__(self) -> "Determinism":
        if self.hermes_home is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="hermes-refactor-equiv-")
            self.hermes_home = Path(self._tmp.name) / "home"
        self.hermes_home.mkdir(parents=True, exist_ok=True)
        old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.hermes_home)
        self._restore.append(lambda: _restore_env("HERMES_HOME", old_home))

        self._patch(time, "time", lambda: FROZEN_EPOCH)
        self._patch(time, "monotonic", lambda: FROZEN_MONOTONIC)
        self._patch(uuid, "uuid4", self._uuid4)
        self._patch(secrets, "token_hex", self._token_hex)
        self._patch(sqlite3, "connect", self._connect)
        return self

    def __exit__(self, *exc: object) -> None:
        while self._restore:
            self._restore.pop()()
        if self._tmp is not None:
            self._tmp.cleanup()

    def _patch(self, module: object, name: str, value: object) -> None:
        old = getattr(module, name)
        setattr(module, name, value)
        self._restore.append(lambda module=module, name=name, old=old: setattr(module, name, old))

    def _uuid4(self) -> uuid.UUID:
        self._uuid_count += 1
        return uuid.UUID(int=self._uuid_count)

    def _token_hex(self, nbytes: int | None = None) -> str:
        self._token_count += 1
        width = max(2, (nbytes or 32) * 2)
        return f"{self._token_count:0{width}x}"[-width:]

    def _connect(self, *args: object, **kwargs: object) -> sqlite3.Connection:
        conn = self._sqlite_connect(*args, **kwargs)
        conn.create_function("datetime", -1, _sqlite_datetime)
        conn.create_function("strftime", -1, _sqlite_strftime)
        return conn


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _sqlite_datetime(*args: object) -> str:
    if not args or str(args[0]).lower() == "now":
        return FROZEN_SQL_DATETIME
    return " ".join(map(str, args))


def _sqlite_strftime(fmt: object, *args: object) -> str:
    text = str(fmt)
    if not args or str(args[0]).lower() == "now":
        return (
            text.replace("%Y", "2027")
            .replace("%m", "01")
            .replace("%d", "15")
            .replace("%H", "08")
            .replace("%M", "00")
            .replace("%S", "00")
        )
    return text


def preflight_scan(paths: Iterable[str | os.PathLike[str]], named_seams: Iterable[str]) -> None:
    seams = set(named_seams)
    missing: list[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            hit = _SCAN.search(line)
            if hit and hit.group(0) not in seams:
                missing.append(f"{path}:{lineno}:{hit.group(0)}")
    if missing:
        raise DeterminismError("nondeterminism without named seam: " + "; ".join(missing))
