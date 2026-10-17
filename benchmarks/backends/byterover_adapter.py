"""Benchmark adapter for the ByteRover memory plugin.

ByteRover exposes a 'brv' CLI binary for curating and querying memory.
This adapter shells out to that binary to exercise the real plugin behaviour
during benchmarks.

Binary resolution order:
  1. shutil.which('brv')                          -- already on PATH
  2. ~/.brv-cli/bin/brv                           -- local install
  3. /usr/local/bin/brv                           -- system install
  4. ~/.npm-global/bin/brv                        -- npm global install
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import BenchmarkableStore

BACKEND_NAME = "byterover"
BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,
)

_BRV_FALLBACK_PATHS: tuple[Path, ...] = (
    Path.home() / ".brv-cli" / "bin" / "brv",
    Path("/usr/local/bin/brv"),
    Path.home() / ".npm-global" / "bin" / "brv",
)


def _resolve_brv_binary() -> Optional[str]:
    """Return the absolute path to the brv binary, or None if not found."""
    on_path = shutil.which("brv")
    if on_path:
        return on_path
    for candidate in _BRV_FALLBACK_PATHS:
        if candidate.is_file() and shutil.os.access(candidate, shutil.os.X_OK):
            return str(candidate)
    return None


class ByteRoverBenchmarkAdapter(BenchmarkableStore):
    """Adapter exposing the ByteRover 'brv' CLI through BenchmarkableStore."""

    def __init__(self, **kwargs):
        binary = _resolve_brv_binary()
        if binary is None:
            raise RuntimeError(
                "ByteRover 'brv' binary not found. "
                "Install ByteRover or ensure 'brv' is on your PATH. "
                "Checked: PATH, ~/.brv-cli/bin/brv, /usr/local/bin/brv, "
                "~/.npm-global/bin/brv."
            )
        self._binary: str = binary
        self._tempdir_obj = tempfile.TemporaryDirectory(prefix="byterover-bench-")
        self._workdir = Path(self._tempdir_obj.name)

    # ------------------------------------------------------------------
    # BenchmarkableStore interface
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        category: str = "factual",
        scope: str = "global",
        importance: float = 0.5,
    ) -> None:
        """Curate a piece of content into ByteRover memory."""
        del category, scope, importance  # brv curate does not accept these flags
        subprocess.run(
            [self._binary, "curate", "--", content],
            cwd=self._workdir,
            timeout=120,
            check=True,
        )

    def recall(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> list[str]:
        """Query ByteRover memory and return a list of result strings."""
        del scope  # brv query does not expose a scope flag
        result = subprocess.run(
            [self._binary, "query", "--", query],
            cwd=self._workdir,
            timeout=10,
            capture_output=True,
            text=True,
        )
        raw = result.stdout.strip()
        if not raw:
            return []
        lines = [line for line in raw.splitlines() if line.strip()]
        return lines[:top_k]

    def simulate_time(self, days: float) -> None:
        """No-op: ByteRover has no time-simulation hook."""
        del days
        return None

    def simulate_access(self, content_substring: str) -> None:
        """No-op: ByteRover has no rehearsal/access API."""
        del content_substring
        return None

    def consolidate(self) -> None:
        """No-op: ByteRover has no explicit consolidation cycle."""
        return None

    def get_stats(self) -> dict[str, Any]:
        return {
            "backend": BACKEND_NAME,
            "binary": self._binary,
        }

    def reset(self) -> None:
        """Clear the working directory, removing all curated data."""
        # Remove all children of the workdir without deleting the dir itself
        # so the TemporaryDirectory cleanup handle remains valid.
        errors: list[str] = []
        for child in list(self._workdir.iterdir()):
            try:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{child}: {exc}")
        if errors:
            raise RuntimeError(
                f"ByteRoverBenchmarkAdapter.reset() could not remove some "
                f"workdir entries:\n" + "\n".join(errors)
            )

    def __del__(self) -> None:
        try:
            self._tempdir_obj.cleanup()
        except Exception:  # noqa: BLE001
            pass


BACKEND_CLASS = ByteRoverBenchmarkAdapter
