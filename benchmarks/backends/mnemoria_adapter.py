"""Benchmark adapter for the Mnemoria memory system.

Mnemoria is a local-first, markdown-native memory layer shipped as an npm
package that speaks MCP.  It stores notes on disk as markdown files and uses
a three-signal retrieval engine (semantic + keyword + graph / PersonalisedPageRank).

This adapter shells out to the ``mnemoria`` CLI binary so that the real plugin
behaviour is exercised during benchmarks rather than a mock.

Binary resolution order
-----------------------
1. shutil.which('mnemoria')                     -- already on PATH
2. ~/.npm-global/bin/mnemoria                   -- npm global install (default prefix)
3. ~/.local/share/npm/bin/mnemoria              -- alternative npm global prefix
4. /usr/local/bin/mnemoria                      -- system-wide install
5. npx mnemoria (fallback, slow first run)      -- run without permanent install

CLI commands used
-----------------
``mnemoria init <vault>``        Scaffold the vault directory structure.
``mnemoria add <content>``       Capture a note / fact to the vault inbox.
``mnemoria query ranked <q>``    Three-signal ranked retrieval (returns JSON).
``mnemoria health``              Validate vault + confirm binary works.

Mnemoria discovers its vault by looking for a ``.mnemoria`` marker in CWD or
parents.  Each benchmark run creates its own isolated temp directory and passes
it as ``cwd`` to subprocess calls, so runs never contaminate each other.

If the binary is absent the adapter raises ``RuntimeError`` at construction
time with clear installation instructions, so failures surface early rather
than on the first API call.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import BenchmarkableStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports consumed by the benchmark runner
# ---------------------------------------------------------------------------

BACKEND_NAME = "mnemoria"

BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,
    time_simulation=False,    # Mnemoria vitality decay is real-clock-based
    access_rehearsal=False,   # No dedicated rehearsal CLI command
    consolidation=False,      # Consolidation happens automatically in the vault
    scopes=False,             # Vault spaces (self/notes/ops) differ from benchmark scopes
    typed_facts=False,
    supersession=False,
    reward_learning=False,
    exploration=False,
    turn_sync=False,
    precompress_hook=False,
    session_end_hook=False,
    delegation_hook=False,
)

# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------

_MNEMORIA_FALLBACK_PATHS: tuple[Path, ...] = (
    Path.home() / ".npm-global" / "bin" / "mnemoria",
    Path.home() / ".local" / "share" / "npm" / "bin" / "mnemoria",
    Path("/usr/local/bin/mnemoria"),
    Path("/usr/bin/mnemoria"),
)

# When set to True, fall back to ``npx mnemoria`` when no local binary exists.
# This is disabled by default because ``npx`` triggers a download on first
# use and makes cold-start benchmarks unreliable.
_ALLOW_NPX_FALLBACK: bool = False


def _resolve_mnemoria_binary() -> Optional[str]:
    """Return the absolute path to the mnemoria binary, or None if not found."""
    on_path = shutil.which("mnemoria")
    if on_path:
        return on_path
    for candidate in _MNEMORIA_FALLBACK_PATHS:
        if candidate.is_file() and shutil.os.access(candidate, shutil.os.X_OK):
            return str(candidate)
    if _ALLOW_NPX_FALLBACK:
        npx = shutil.which("npx")
        if npx:
            return None  # handled separately by _run()
    return None


def _is_mnemoria_available() -> bool:
    """Return True if the mnemoria binary can be located on this system."""
    return _resolve_mnemoria_binary() is not None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class MnemoriaBenchmarkAdapter(BenchmarkableStore):
    """Adapter exposing the Mnemoria CLI through BenchmarkableStore.

    Each adapter instance owns an isolated temporary vault directory.  The
    vault is initialised via ``mnemoria init`` at construction time and torn
    down (``reset()`` or ``__del__``) afterwards.

    Parameters
    ----------
    vault_path:
        Optional path to an existing vault.  When omitted a fresh temporary
        directory is created so multiple benchmark runs remain independent.
    timeout_store:
        Subprocess timeout (seconds) for ``mnemoria add`` calls.  Defaults to
        60 s because the first add in a new vault triggers embedding model
        download.
    timeout_recall:
        Subprocess timeout (seconds) for ``mnemoria query`` calls.
    """

    def __init__(
        self,
        vault_path: Optional[str] = None,
        timeout_store: int = 60,
        timeout_recall: int = 30,
        **kwargs,
    ) -> None:
        binary = _resolve_mnemoria_binary()
        if binary is None:
            raise RuntimeError(
                "Mnemoria binary not found.  Install it with:\n"
                "  npm install -g mnemoria\n"
                "Then verify with: mnemoria health\n"
                "Searched: PATH, ~/.npm-global/bin, ~/.local/share/npm/bin, "
                "/usr/local/bin, /usr/bin."
            )
        self._binary: str = binary
        self._timeout_store: int = timeout_store
        self._timeout_recall: int = timeout_recall

        # Vault lifecycle
        if vault_path:
            self._vault = Path(vault_path)
            self._vault.mkdir(parents=True, exist_ok=True)
            self._owned_tempdir = None
        else:
            self._owned_tempdir = tempfile.TemporaryDirectory(
                prefix="mnemoria-bench-"
            )
            self._vault = Path(self._owned_tempdir.name)

        self._store_count: int = 0
        self._init_vault()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(
        self,
        args: List[str],
        *,
        timeout: int = 30,
        check: bool = True,
        capture: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run mnemoria with CWD set to vault directory.

        Mnemoria discovers its vault by walking up from CWD looking for
        a ``.mnemoria`` marker directory, so we run every subprocess with
        ``cwd=self._vault`` rather than passing a ``--vault`` flag.
        """
        cmd = [self._binary] + args
        return subprocess.run(
            cmd,
            cwd=str(self._vault),
            timeout=timeout,
            check=check,
            capture_output=capture,
            text=True,
        )

    def _init_vault(self) -> None:
        """Initialise the vault by running ``mnemoria init``."""
        try:
            # ``mnemoria init .`` is idempotent — safe to call on existing vaults.
            # We pass "." because _run already sets cwd to self._vault.
            self._run(["init", ".", "--json"], timeout=60, check=True, capture=True)
            logger.debug("Mnemoria vault initialised at %s", self._vault)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"mnemoria init failed (exit {exc.returncode}).  "
                "Make sure the mnemoria binary is functional and try "
                "'mnemoria health'."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "mnemoria init timed out.  The binary may be hanging on first "
                "use while downloading the embedding model.  "
                "Try running 'mnemoria health' manually first."
            ) from exc

    def _parse_ranked_output(self, stdout: str, top_k: int) -> List[str]:
        """Parse ``mnemoria query ranked`` stdout into a list of content strings.

        The command returns ``{"success": true, "data": {"results": [...]}}``
        where each result has ``title`` (note slug) and ``score``.  We read
        the full note body from disk when possible; otherwise fall back to the
        title (which is the fact content as a slug).

        Falls back to line-splitting when the output is not valid JSON.
        """
        text = stdout.strip()
        if not text:
            return []

        # Primary path: JSON envelope {"success": ..., "data": {"results": [...]}}
        if text.startswith("{"):
            try:
                envelope = json.loads(text)
                data = envelope.get("data", {})
                items = data.get("results") or data.get("items") or []
                results: List[str] = []
                for item in items[:top_k]:
                    if not isinstance(item, dict):
                        continue
                    title = item.get("title", "")
                    # Skip scaffold notes
                    if title in ("index", "identity", "goals", "methodology",
                                 "daily", "reminders", "related note", "relevant map"):
                        continue
                    # Try to read full note body from vault
                    content = self._read_note_body(title)
                    if content:
                        results.append(content)
                    elif title:
                        # Fall back to de-slugifying the title
                        results.append(title.replace("-", " "))
                return results
            except json.JSONDecodeError:
                pass

        # Fallback: JSON array
        if text.startswith("["):
            try:
                items = json.loads(text)
                return [
                    str(item.get("content") or item.get("title") or item)
                    for item in items[:top_k]
                    if item
                ]
            except json.JSONDecodeError:
                pass

        # Plain-text fallback
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[:top_k]

    def _read_note_body(self, slug: str) -> str:
        """Read the body content of a note by its slug.

        Searches notes/ then inbox/ for ``<slug>.md``, strips YAML
        frontmatter and wiki-link boilerplate, returns the core content.
        """
        for subdir in ("notes", "inbox", "self", "ops"):
            note_path = self._vault / subdir / f"{slug}.md"
            if note_path.exists():
                try:
                    text = note_path.read_text()
                    # Strip YAML frontmatter
                    if text.startswith("---"):
                        end = text.find("---", 3)
                        if end != -1:
                            text = text[end + 3:].strip()
                    # Strip heading (# Title)
                    lines = text.split("\n")
                    body_lines = []
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("# "):
                            continue  # skip title
                        if stripped == "---":
                            break  # stop at separator (wiki-links section)
                        if stripped.startswith("Relevant Notes:") or stripped.startswith("Areas:"):
                            break
                        body_lines.append(line)
                    body = "\n".join(body_lines).strip()
                    return body if body else ""
                except OSError:
                    pass
        return ""

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
        """Capture *content* into the Mnemoria vault via ``mnemoria add``.

        Parameters
        ----------
        content:
            The text to remember.
        category:
            Semantic category tag.  Passed as-is for interface compatibility;
            Mnemoria routes notes internally based on content analysis.
        scope:
            Scope tag (``"global"``, ``"session"``, …).  Ignored because
            Mnemoria organises storage across its own vault spaces (self /
            notes / ops) rather than arbitrary scope strings.
        importance:
            Salience weight in [0, 1].  Ignored; Mnemoria computes vitality
            from access patterns and structural links.
        """
        del category, scope, importance  # unused by this backend
        try:
            # ``mnemoria add <title>`` creates a note with a template body.
            # The title must be multi-word prose.  We use the content as the
            # title (Mnemoria slugifies it) and then overwrite the template
            # body with the actual content so ranked queries can match it.
            title = content[:120].strip() if len(content) > 120 else content.strip()
            # Ensure multi-word (Mnemoria requires prose-as-title)
            if len(title.split()) < 2:
                title = f"fact {title}"

            result = self._run(
                ["add", "--type", "insight", "--", title],
                timeout=self._timeout_store,
                check=True,
                capture=True,
            )
            # Parse the created file path and overwrite template body with real content
            slug = ""
            try:
                data = json.loads(result.stdout)
                note_path = data.get("data", {}).get("path", "")
                if note_path and Path(note_path).exists():
                    note_file = Path(note_path)
                    text = note_file.read_text()
                    # Replace template placeholder with actual content
                    text = text.replace(
                        "{Content - your reasoning, evidence, context. "
                        "Transform the material, don't just summarize.}",
                        content,
                    )
                    note_file.write_text(text)
                    slug = note_file.stem  # e.g. "the-capital-of-france-is-paris"
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Could not overwrite note body: %s", exc)

            # Promote from inbox/ to notes/ so ranked queries can find it.
            # Mnemoria only searches promoted notes, not inbox.
            if slug:
                try:
                    self._run(
                        ["promote", slug, "--no-auto"],
                        timeout=self._timeout_store,
                        check=False,
                        capture=True,
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    logger.debug("promote failed for %s, note stays in inbox", slug)

            self._store_count += 1
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "mnemoria add failed (exit %d): %s", exc.returncode, exc.stderr
            )
        except subprocess.TimeoutExpired:
            logger.warning("mnemoria add timed out for content: %.80s…", content)

    def recall(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> List[str]:
        """Retrieve memories matching *query* using three-signal ranked search.

        Uses ``mnemoria query ranked`` which fuses semantic embeddings, BM25
        keyword scoring, and Personalised PageRank into a single ranked list.

        Parameters
        ----------
        query:
            Natural-language search string.
        top_k:
            Maximum number of results to return.
        scope:
            Optional scope filter.  Ignored because Mnemoria does not expose
            benchmark-level scoping via its CLI.
        """
        del scope  # unused by this backend
        try:
            result = self._run(
                ["query", "ranked", query, "--limit", str(top_k)],
                timeout=self._timeout_recall,
                check=False,
                capture=True,
            )
            return self._parse_ranked_output(result.stdout, top_k)
        except subprocess.TimeoutExpired:
            logger.warning("mnemoria query timed out for query: %.80s…", query)
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("mnemoria query failed: %s", exc)
            return []

    def simulate_time(self, days: float) -> None:
        """No-op — Mnemoria vitality decay is tied to the real-world clock.

        Parameters
        ----------
        days:
            Number of days to advance (ignored).
        """
        del days
        return None

    def simulate_access(self, content_substring: str) -> None:
        """No-op — Mnemoria has no dedicated rehearsal CLI command.

        Vitality boosts happen automatically when notes are retrieved via
        ``mnemoria query ranked`` (activation spreading), not via an explicit
        rehearsal API.

        Parameters
        ----------
        content_substring:
            Substring identifying the memory to rehearse (ignored).
        """
        del content_substring
        return None

    def consolidate(self) -> None:
        """No-op — Mnemoria consolidates notes automatically in the background.

        The ACT-R vitality model and Hebbian link weights are maintained
        incrementally on every access rather than through an explicit
        consolidation cycle.
        """
        return None

    def get_stats(self) -> dict[str, Any]:
        """Return basic adapter statistics.

        Returns
        -------
        dict
            Contains ``backend`` name, ``vault`` path, and ``store_count``
            (number of successful ``store()`` calls since last ``reset()``).
            A ``health`` key is included when ``mnemoria health`` succeeds.
        """
        stats: dict[str, Any] = {
            "backend": BACKEND_NAME,
            "vault": str(self._vault),
            "store_count": self._store_count,
        }
        try:
            result = self._run(
                ["health"],
                timeout=15,
                check=False,
                capture=True,
            )
            stats["health"] = result.stdout.strip() or "ok"
        except Exception:  # noqa: BLE001
            stats["health"] = "unavailable"
        return stats

    def reset(self) -> None:
        """Clear all stored memories by re-initialising the vault.

        All markdown notes and the embedding database inside the vault
        directory are removed and a fresh vault is scaffolded in their place.
        The ``store_count`` counter is reset to zero.
        """
        # Wipe vault contents
        errors: list[str] = []
        for child in list(self._vault.iterdir()):
            try:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{child}: {exc}")
        if errors:
            logger.warning(
                "mnemoria reset: could not remove some vault entries:\n%s",
                "\n".join(errors),
            )
        self._store_count = 0
        self._init_vault()

    def __del__(self) -> None:
        tempdir = getattr(self, "_owned_tempdir", None)
        if tempdir is not None:
            try:
                tempdir.cleanup()
            except Exception:  # noqa: BLE001
                pass


# Module-level alias consumed by the benchmark runner's dynamic loader.
BACKEND_CLASS = MnemoriaBenchmarkAdapter
