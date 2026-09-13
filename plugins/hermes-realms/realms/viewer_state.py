"""Shared takeover leases and revocation epochs; no viewer secrets on disk."""

import contextlib
import os
from pathlib import Path
import sqlite3
import stat
import time

from .lifecycle import alive, identity


class ControlAuthority:
    TTL = 3.0

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()  # windows-footgun: ok — runtime package rejects non-Linux hosts
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise ValueError("Viewer state must be a private owned directory")
        self.path = self.directory / "authority.sqlite"
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        with self._connection() as db:
            db.executescript(
                "CREATE TABLE IF NOT EXISTS leases(realm TEXT PRIMARY KEY, generation TEXT NOT NULL, lease TEXT NOT NULL, pid INTEGER NOT NULL, started INTEGER NOT NULL, expires REAL NOT NULL); CREATE TABLE IF NOT EXISTS epochs(realm TEXT PRIMARY KEY, value INTEGER NOT NULL);"
            )

    @contextlib.contextmanager
    def _connection(self):
        info = self.path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()  # windows-footgun: ok — runtime package rejects non-Linux hosts
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("Viewer authority ownership changed")
        db = sqlite3.connect(self.path, timeout=3)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _live(row):
        return (
            row is not None
            and row[2] > time.time()
            and alive({"pid": row[0], "start_time": row[1]})
        )

    def acquire(self, realm, generation, lease):
        own = identity(os.getpid())
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT pid,started,expires FROM leases WHERE realm=?", (realm,)
            ).fetchone()
            if self._live(row):
                return False
            db.execute(
                "INSERT OR REPLACE INTO leases VALUES(?,?,?,?,?,?)",
                (
                    realm,
                    generation,
                    lease,
                    own["pid"],
                    own["start_time"],
                    time.time() + self.TTL,
                ),
            )
            return True

    def controlled(self, realm):
        with self._connection() as db:
            row = db.execute(
                "SELECT pid,started,expires FROM leases WHERE realm=?", (realm,)
            ).fetchone()
            return self._live(row)

    def refresh(self, realm, lease):
        with self._connection() as db:
            changed = db.execute(
                "UPDATE leases SET expires=? WHERE realm=? AND lease=? AND expires>?",
                (time.time() + self.TTL, realm, lease, time.time()),
            ).rowcount
            return changed == 1

    @contextlib.contextmanager
    def forwarding(self, realm, generation, lease=None):
        """Serialize the synchronous socket write with takeover and revocation.

        Never await inside this guard: the transaction is held only through the
        final ticket check and transport write, not network backpressure.
        """
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            epoch = db.execute(
                "SELECT value FROM epochs WHERE realm=?", (realm,)
            ).fetchone()
            valid = generation.rpartition(":")[2] == str(epoch[0] if epoch else 0)
            if lease is not None:
                row = db.execute(
                    "SELECT pid,started,expires,generation,lease FROM leases WHERE realm=?",
                    (realm,),
                ).fetchone()
                valid = (
                    valid
                    and self._live(row[:3] if row else None)
                    and row[3:] == (generation, lease)
                    and identity(os.getpid()) == {"pid": row[0], "start_time": row[1]}
                )
                if valid:
                    db.execute(
                        "UPDATE leases SET expires=? WHERE realm=? AND lease=?",
                        (time.time() + self.TTL, realm, lease),
                    )
            yield bool(valid)

    def release(self, realm, lease):
        with self._connection() as db:
            db.execute("DELETE FROM leases WHERE realm=? AND lease=?", (realm, lease))

    def epoch(self, realm):
        with self._connection() as db:
            row = db.execute(
                "SELECT value FROM epochs WHERE realm=?", (realm,)
            ).fetchone()
            return row[0] if row else 0

    def revoke(self, realm):
        with self._connection() as db:
            db.execute(
                "INSERT INTO epochs VALUES(?,1) ON CONFLICT(realm) DO UPDATE SET value=value+1",
                (realm,),
            )
