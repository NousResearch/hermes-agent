"""Secret-source contract: the ABC every secret backend implements.

A *secret source* resolves credentials from an external secret manager
(Bitwarden Secrets Manager, 1Password, an OS keystore, a user script, ...)
into environment-variable-shaped values at process startup, AFTER
``~/.hermes/.env`` has loaded and BEFORE the rest of Hermes reads
``os.environ``.

Scope of the contract (deliberate, please do not widen):

* **Read-only.**  Sources resolve refs → values.  There is no write-back
  ("save this key to your vault"), no arbitrary secret objects, and no
  mid-session secret API.  If a future need for rotation/refresh appears
  it will arrive as a versioned optional hook — do not bolt it on.
* **Startup-time, synchronous.**  ``fetch()`` is called once per process
  (per HERMES_HOME) by the orchestrator in
  :mod:`agent.secret_sources.registry`, which enforces a wall-clock
  timeout around it.  Sources must not spawn background refreshers.
* **Never raises, never prompts.**  ``fetch()`` returns a
  :class:`FetchResult` — errors go in ``result.error`` with a
  machine-readable :class:`ErrorKind`.  Interactive auth belongs in the
  source's CLI ``setup`` flow, never on the startup path (non-TTY
  gateway/cron startup must never block on stdin).
* **Sources fetch; the orchestrator applies.**  A source returns the
  name→value mapping it *would* contribute.  Precedence (mapped-beats-bulk,
  first-wins, ``override_existing``, protected vars), conflict warnings,
  provenance tracking, and the actual ``os.environ`` writes are owned by
  the orchestrator so no backend can get them wrong.

Versioning: ``SECRET_SOURCE_API_VERSION`` gates plugin compatibility.
New *optional* hooks with default implementations do not bump it;
required-signature changes do, and the registry skips (with a warning)
sources built against a different major version instead of crashing
startup.
"""

from __future__ import annotations

import os
import re
import subprocess
from contextvars import ContextVar, Token
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
)

# Bump ONLY for breaking changes to the required contract surface
# (abstract-method signatures, FetchResult required fields).  Additive
# optional hooks must ship with defaults and must NOT bump this.
SECRET_SOURCE_API_VERSION = 1

_SOURCE_ENVIRONMENT: ContextVar[Optional[MutableMapping[str, str]]]
_SOURCE_ENVIRONMENT = ContextVar("hermes_secret_source_environment", default=None)


def set_source_environment(environ: MutableMapping[str, str]) -> Token:
    """Install a per-fetch environment view without changing ``os.environ``."""
    return _SOURCE_ENVIRONMENT.set(environ)


def reset_source_environment(token: Token) -> None:
    _SOURCE_ENVIRONMENT.reset(token)


def get_source_environment() -> MutableMapping[str, str]:
    """Return the active per-fetch environment, or the process environment."""
    environ = _SOURCE_ENVIRONMENT.get()
    return environ if environ is not None else os.environ

# Timeout the orchestrator enforces around fetch() when the source's
# config section doesn't override it.  Generous because a first run may
# include a one-time CLI binary auto-install (e.g. bws download+verify).
DEFAULT_FETCH_TIMEOUT_SECONDS = 120.0

# Default timeout for run_secret_cli() subprocess invocations.
DEFAULT_CLI_TIMEOUT_SECONDS = 30.0


class ErrorKind(str, Enum):
    """Machine-readable failure taxonomy for :class:`FetchResult.error`.

    A fixed vocabulary keeps startup warnings and ``hermes secrets status``
    uniform across backends, and lets the orchestrator implement
    kind-dependent policy (e.g. a future stale-cache fallback on
    ``NETWORK``/``TIMEOUT`` but not on ``AUTH_FAILED``) exactly once.
    """

    NOT_CONFIGURED = "not_configured"    # enabled but missing token/project/map
    BINARY_MISSING = "binary_missing"    # helper CLI not found / not installed
    AUTH_FAILED = "auth_failed"          # bad credentials
    AUTH_EXPIRED = "auth_expired"        # credentials were valid, aren't now
    REF_INVALID = "ref_invalid"          # a secret reference failed validation
    NETWORK = "network"                  # transport-level failure
    EMPTY_VALUE = "empty_value"          # backend returned nothing for a ref
    TIMEOUT = "timeout"                  # fetch exceeded its wall-clock budget
    INTERNAL = "internal"                # anything else (bug, unexpected shape)


@dataclass
class FetchResult:
    """Outcome of one source's fetch.

    ``secrets`` holds what the source *would* contribute; whether each
    var is actually applied is the orchestrator's decision.  ``applied``
    and ``skipped`` exist for backward compatibility with the original
    Bitwarden fetch-and-apply entry point and are left empty by
    conforming ``fetch()`` implementations.
    """

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    error_kind: Optional[ErrorKind] = None
    # Path of the helper binary used, when the source is CLI-driven.
    # Surfaced by status commands; None for SDK/API-driven sources.
    binary_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class SecretSource(ABC):
    """One external secret backend.

    Subclasses set the class attributes and implement :meth:`fetch`.
    Everything else has a sensible default.

    Attributes:
        name: Config-section key under ``secrets:`` in config.yaml.
            Lowercase ``[a-z0-9_]+``.  Also the provenance label stored
            for every var this source supplies.
        label: Human-readable name used in startup messages and
            ``hermes secrets status`` (e.g. ``"Bitwarden Secrets Manager"``).
        shape: ``"mapped"`` when the user explicitly binds env-var names
            to refs (1Password ``env:`` map, command source) or
            ``"bulk"`` when the backend injects whole projects/folders
            of secrets implicitly (Bitwarden BSM).  The orchestrator
            gives mapped sources precedence over bulk sources: an
            explicit binding is stronger intent than a project dump.
        scheme: Optional URI scheme this source owns for secret
            references (``"op"`` for ``op://...``).  Must be unique
            across registered sources — refs may eventually appear
            outside the ``secrets:`` block (e.g. credential-pool
            ``api_key`` fields), so scheme collisions are rejected at
            registration time to keep that future possible.
        api_version: Contract version this source was built against.
    """

    api_version: int = SECRET_SOURCE_API_VERSION
    name: str = ""
    label: str = ""
    shape: str = "mapped"  # "mapped" | "bulk"
    scheme: Optional[str] = None

    # -- required ----------------------------------------------------------

    @abstractmethod
    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        """Resolve this source's secrets. MUST NOT raise or prompt.

        ``cfg`` is the source's raw config section (``secrets.<name>``)
        from config.yaml — treat every field defensively, the section
        may be malformed.  ``home_path`` is the resolved HERMES_HOME.
        """

    # -- optional hooks (defaults are correct for most sources) ------------

    def is_enabled(self, cfg: dict) -> bool:
        """Whether the user turned this source on."""
        return bool(isinstance(cfg, dict) and cfg.get("enabled"))

    def override_existing(self, cfg: dict) -> bool:
        """May this source overwrite vars that .env / the shell already set?

        This NEVER extends to vars claimed by another secret source in the
        same startup pass — cross-source overrides are a config error the
        orchestrator warns about, not a knob.
        """
        return bool(isinstance(cfg, dict) and cfg.get("override_existing", False))

    def protected_env_vars(self, cfg: dict) -> FrozenSet[str]:
        """Env vars the orchestrator must never let ANY source overwrite.

        Typically the source's own bootstrap-auth var (e.g.
        ``BWS_ACCESS_TOKEN``) so a vault that contains its own access
        token can't clobber the credential used to reach it.
        """
        return frozenset()

    def fetch_timeout_seconds(self, cfg: dict) -> float:
        """Wall-clock budget the orchestrator enforces around fetch()."""
        try:
            val = float((cfg or {}).get("timeout_seconds", DEFAULT_FETCH_TIMEOUT_SECONDS))
        except (TypeError, ValueError):
            return DEFAULT_FETCH_TIMEOUT_SECONDS
        return val if val > 0 else DEFAULT_FETCH_TIMEOUT_SECONDS

    def config_schema(self) -> dict:
        """Optional description of this source's config keys.

        Shape: ``{key: {"description": str, "default": Any}}``.  Used by
        setup surfaces to render config without hardcoding per-source
        knowledge.  Purely informational.
        """
        return {}

    def remediation(self, kind: Optional["ErrorKind"], cfg: dict) -> str:
        """One-line, actionable next step for a failed fetch.

        Called by the startup status printer (and ``hermes secrets ...
        status``) right after a fetch error is surfaced, so the user sees
        *what to run* next to fix it — not just what broke.  Sources
        should override this to point at their own CLI verbs (e.g.
        ``hermes secrets bitwarden token`` for AUTH_FAILED).  Return an
        empty string to suppress the hint.

        Must never raise and must not perform I/O — it's a pure
        kind→string mapping on the startup path.
        """
        generic = {
            ErrorKind.NOT_CONFIGURED: (
                f"Run `hermes secrets {self.name} setup` to finish configuration."
            ),
            ErrorKind.BINARY_MISSING: (
                f"Run `hermes secrets {self.name} setup` to install the helper CLI."
            ),
            ErrorKind.AUTH_FAILED: (
                f"Credentials rejected — run `hermes secrets {self.name} setup` "
                "to re-authenticate."
            ),
            ErrorKind.AUTH_EXPIRED: (
                f"Credentials expired — run `hermes secrets {self.name} setup` "
                "to re-authenticate."
            ),
            ErrorKind.NETWORK: (
                "Network problem reaching the secrets backend — check "
                "connectivity and retry."
            ),
            ErrorKind.TIMEOUT: (
                f"Backend was slow — raise secrets.{self.name}.timeout_seconds "
                "if this recurs."
            ),
        }
        return generic.get(kind, "") if kind is not None else ""


# ---------------------------------------------------------------------------
# Shared helpers — use these instead of hand-rolling per backend
# ---------------------------------------------------------------------------


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# ECMA-48 control sequences — helper-CLI stderr often carries controls that
# must not reach Hermes' own startup output or split an exact token before it
# is redacted.  This intentionally remains separate from
# ``tools.ansi_strip.strip_ansi``: the ``$`` alternatives also consume an
# unterminated OSC/DCS/SOS/PM/APC string, which is common when a CLI is killed
# mid-write and is required for safe provider-output handling.
_ANSI_RE = re.compile(
    r"(?:"
    # 7-bit ESC-prefixed forms: CSI, OSC, DCS/SOS/PM/APC, nF, Fe/Fs.
    r"\x1b(?:"
    r"\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"
    r"|\][\s\S]*?(?:\x07|\x1b\\|$)"
    r"|[PX^_][\s\S]*?(?:\x1b\\|$)"
    r"|[\x20-\x2f]+[\x30-\x7e]"
    r"|[\x30-\x7e]"
    r")"
    # 8-bit C1 forms: CSI, OSC, DCS/SOS/PM/APC.
    r"|\x9b[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"
    r"|\x9d[\s\S]*?(?:\x07|\x9c|\x1b\\|$)"
    r"|(?:\x90|\x98|\x9e|\x9f)[\s\S]*?(?:\x9c|\x1b\\|$)"
    # Any remaining C1 byte is itself a non-printing control.
    r"|[\x80-\x9f]"
    r")",
    re.DOTALL,
)


def is_valid_env_name(name: str) -> bool:
    """True when ``name`` is a legal environment-variable name."""
    return bool(name) and bool(_ENV_NAME_RE.match(name))


def scrub_ansi(text: str) -> str:
    """Strip ANSI/ECMA-48 controls, including 8-bit C1 forms."""
    return _ANSI_RE.sub("", text or "")


# Provider diagnostics need a slightly different policy from the general
# ECMA-48 scrubber above.  An ambiguous ``ESC`` followed by a printable byte
# is deliberately treated as a lone ESC here: consuming the following byte
# can remove the first character of a known token before exact redaction.  The
# complete, unambiguous CSI/OSC/DCS families remain stripped, including their
# unterminated forms.
_PROVIDER_ANSI_RE = re.compile(
    r"(?:"
    r"\x1b(?:"
    r"\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"
    r"|\][\s\S]*?(?:\x07|\x1b\\|$)"
    r"|[PX^_][\s\S]*?(?:\x1b\\|$)"
    r")"
    r"|\x9b[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"
    r"|\x9d[\s\S]*?(?:\x07|\x9c|\x1b\\|$)"
    r"|(?:\x90|\x98|\x9e|\x9f)[\s\S]*?(?:\x9c|\x1b\\|$)"
    r"|[\x80-\x9f]"
    r")",
    re.DOTALL,
)

# Preserve ordinary tabs and line feeds for readable diagnostics.  Every
# other C0/C1 byte is removed; in particular CR is not retained because it
# can overwrite an earlier terminal line and it must not split a token.
_PROVIDER_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def normalize_provider_output(text: str) -> str:
    """Normalize untrusted provider output before it is surfaced.

    Full ECMA-48 strings are removed, ambiguous/lone ESC bytes are removed
    without consuming adjacent printable text, and unsafe bare controls are
    dropped.  Tabs and LF remain for useful line structure; CR is removed so
    it cannot spoof terminal output or split a known secret.
    """
    normalized = _PROVIDER_ANSI_RE.sub("", text or "")
    return _PROVIDER_CONTROL_RE.sub("", normalized)


def _redact_secret_with_escape_payload(
    text: str,
    secret: str,
    replacement: str,
) -> str:
    """Redact one secret across controls and malformed escape payloads.

    Provider output is untrusted and may contain an incomplete or malformed
    escape sequence between two characters of a credential.  A regular
    expression for that language is both difficult to make complete and easy
    to make catastrophically backtracking.  This scanner keeps at most two
    candidate states per secret character instead:

    * C0/C1 controls are transparent gaps;
    * after an ESC, printable bytes are a possible malformed payload, so the
      scanner retains both the path which skips a byte and the path which
      consumes it as the next secret character;
    * normal printable bytes must match the next secret character exactly.

    The state count is bounded by the secret length and every input byte is
    processed by that finite automaton once.  Completed matches are
    represented as source spans, so one replacement covers the credential and
    any controls/payload separating its characters.  This preserves the
    existing readable diagnostic shape after the later ANSI/control
    normalization while never leaving a matched token character in the output.
    """
    if not text or not secret:
        return text or ""

    # Credential values should not contain controls, but retaining exact
    # replacement semantics for such a value is safer than interpreting one
    # of its bytes as a scanner separator.
    if any(
        ord(char) < 0x20 or 0x7f <= ord(char) <= 0x9f
        for char in secret
    ):
        return text.replace(secret, replacement)

    secret_length = len(secret)
    # A repeated first-character stream can legitimately create several
    # finite candidates while an ESC payload is open.  Keep the total work
    # linear in the untrusted diagnostic; if the candidate automaton would
    # exceed its budget, fail closed by replacing the entire diagnostic.  No
    # provider-controlled text can then survive with a credential fragment.
    transition_budget = max(4096, len(text) * 8)
    transitions = 0
    # A candidate is (start offset, in_escape_gap); the matched source
    # characters need not be retained because a completed path's source span
    # is exactly ``start:index + 1``.  Avoiding per-transition position tuples
    # keeps long provider tokens from turning the finite scanner into a large
    # allocation loop.
    # Keep only the newest path for each (matched length, gap) state.  Any
    # future input has the same transition possibilities for those states;
    # retaining the newest path prevents stale candidates from spanning the
    # entire diagnostic while keeping the automaton finite.
    active: dict[tuple[int, bool], int] = {}
    spans: list[tuple[int, int]] = []

    def add_candidate(
        target: dict[tuple[int, bool], int],
        matched: int,
        in_escape_gap: bool,
        start: int,
        end: int,
    ) -> None:
        if matched >= secret_length:
            spans.append((start, end))
            return
        target[(matched, in_escape_gap)] = start

    for index, char in enumerate(text):
        codepoint = ord(char)
        is_control = codepoint < 0x20 or 0x7f <= codepoint <= 0x9f
        starts_escape_gap = char == "\x1b" or codepoint in {
            0x90, 0x98, 0x9B, 0x9D, 0x9E, 0x9F
        }
        next_active: dict[tuple[int, bool], int] = {}

        for (matched, in_escape_gap), start in active.items():
            transitions += 1
            if transitions > transition_budget:
                return replacement
            if is_control:
                # ESC and the 8-bit C1 string/CSI introducers start a
                # permissive malformed-payload gap. Other C0/C1 bytes are
                # transparent without changing the current mode.
                add_candidate(
                    next_active,
                    matched,
                    in_escape_gap or starts_escape_gap,
                    start,
                    index + 1,
                )
                continue

            expected = secret[matched]
            if in_escape_gap:
                # The payload may contain arbitrary printable bytes.  Keep
                # the skip path and, when this byte is the expected secret
                # character, a consuming path which exits the gap.
                add_candidate(
                    next_active,
                    matched,
                    True,
                    start,
                    index + 1,
                )
                if char == expected:
                    add_candidate(
                        next_active,
                        matched + 1,
                        False,
                        start,
                        index + 1,
                    )
                continue

            if char == expected:
                add_candidate(
                    next_active,
                    matched + 1,
                    False,
                    start,
                    index + 1,
                )

        # A secret may begin at any ordinary character, including one inside
        # a malformed escape payload.  It is deliberately added after the
        # active transitions so a one-character secret is handled uniformly.
        if char == secret[0]:
            add_candidate(next_active, 1, False, index, index + 1)
        active = next_active

    if not spans:
        return text

    # Merge overlapping/adjacent matches.  Adjacent occurrences should retain
    # the historical single marker shape, and overlap can arise when a secret
    # has a repeated prefix.
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def redact_provider_output(
    text: str,
    secrets: Iterable[str],
    *,
    replacement: str = "<redacted>",
) -> str:
    """Normalize provider output and redact every explicitly known secret.

    ``secrets`` must contain only authentication values intentionally passed to
    the provider child.  Matching tolerates all C0/C1 controls between secret
    characters, including tabs, newlines, CR, and a lone ESC, while preserving
    those characters in unrelated diagnostic text where safe.
    """
    normalized = text or ""
    values = sorted(
        {value for value in secrets if isinstance(value, str) and value},
        key=len,
        reverse=True,
    )
    if not values:
        return normalize_provider_output(normalized)

    # Keep the marker out of the untrusted text while ANSI normalization runs.
    # A literal ``<redacted>`` contains CSI-final characters; if it follows an
    # 8-bit CSI introducer in provider output, the normalizer could consume the
    # beginning of the marker as an escape payload.  Select a supplementary
    # Unicode sentinel absent from the complete diagnostic and every supplied
    # secret, so later scanner passes cannot match it. If a hostile diagnostic
    # occupies the whole candidate range, fail closed rather than using a
    # colliding fallback that would corrupt unrelated output.
    occupied = set(normalized)
    for value in values:
        occupied.update(value)
    sentinel = next(
        (
            chr(codepoint)
            for codepoint in range(0x100000, 0x10FFFE)
            if chr(codepoint) not in occupied
        ),
        None,
    )
    if sentinel is None:
        return replacement
    for secret in values:
        # Redact before ANSI parsing as well: a malformed ESC sequence can
        # contain arbitrary printable payload that a strict parser would
        # otherwise partially consume as its final character.  The scanner
        # is finite-state and linear in the diagnostic length; do not replace
        # it with a nested regex.
        normalized = _redact_secret_with_escape_payload(
            normalized, secret, sentinel
        )
    normalized = normalize_provider_output(normalized)
    return normalized.replace(sentinel, replacement)


def redact_secret_value(
    text: str, secret: str, *, replacement: str = "<redacted>"
) -> str:
    """Replace an exact secret value before provider text is surfaced.

    Provider diagnostics are untrusted output, so broad pattern-based
    redaction is not sufficient when the candidate credential is already
    known to the caller.  Exact replacement preserves the provider's other
    useful context without allowing it to echo that credential.
    """
    if not text or not secret:
        return text or ""
    return text.replace(secret, replacement)


# The provider CLIs only need process-location/locale basics plus the standard
# network/TLS settings used by Hermes' own HTTP clients.  In particular, do
# not add a broad ``*_TOKEN``/``*_API_KEY`` pattern here: the allowlist is
# deliberately explicit at each provider call site so its authentication and
# session contract stays reviewable.  Proxy values can contain connection
# credentials, but are intentionally limited to the standard proxy names;
# unrelated credentials remain excluded.
_PROVIDER_ENV_BASE = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "APPDATA",
    "LOCALAPPDATA",
    # Windows preserves the mixed-case spelling in some plain per-fetch
    # mappings; keep it alongside the normalized form used by os.environ.
    "SystemRoot",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_RUNTIME_DIR",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
    "no_proxy",
    "HERMES_CA_BUNDLE",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)


def build_minimal_provider_env(
    source_env: Optional[Mapping[str, str]] = None,
    *,
    allow_env: Sequence[str] = (),
    extra_env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Build a provider-child environment from an explicit allowlist.

    ``source_env`` defaults to the active per-fetch environment, which keeps
    registry-managed source fetches isolated without changing ``os.environ``.
    Callers add only the provider's required token/session/account variables
    through ``allow_env`` or ``extra_env``.
    """
    source = source_env if source_env is not None else get_source_environment()
    env: Dict[str, str] = {}
    for key in (*_PROVIDER_ENV_BASE, *allow_env):
        val = source.get(key)
        if val is not None:
            env[key] = val
    if extra_env:
        env.update(extra_env)
    env["NO_COLOR"] = "1"
    return env


_VERSION_RE = re.compile(r"(?<![\w.])v?\d+(?:\.\d+){1,3}(?![\w.])")


def sanitize_provider_version(output: str) -> str:
    """Return only a plain numeric version from provider-controlled output."""
    first_line = scrub_ansi(output or "").strip().splitlines()
    if not first_line:
        return "version unknown"
    match = _VERSION_RE.search(first_line[0])
    return match.group(0) if match else "version unknown"


def run_secret_cli(
    argv: Sequence[str],
    *,
    allow_env: Sequence[str] = (),
    extra_env: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_CLI_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    """Run a secret-manager helper CLI with a minimal, allowlisted env.

    Security posture shared by every subprocess-driven backend:

    * argv list only — never ``shell=True``.  Callers pass user-supplied
      reference strings AFTER a ``--`` option terminator in their argv.
    * The child gets ``PATH``/``HOME``/locale basics plus only the env
      vars named in ``allow_env`` (auth/session vars) and ``extra_env``
      — never a copy of the full post-dotenv ``os.environ``, which by
      this point holds every credential Hermes knows about.
    * ``NO_COLOR=1`` is set and stderr/stdout are ANSI-scrubbed so
      helper diagnostics can't smuggle escape sequences into Hermes
      output.
    * stdin is ``/dev/null`` so a helper that decides to prompt fails
      fast instead of hanging startup.

    Raises ``RuntimeError`` on spawn failure or timeout (message safe to
    surface); returns the completed process otherwise — callers own
    returncode interpretation.
    """
    env = build_minimal_provider_env(
        allow_env=allow_env,
        extra_env=extra_env,
    )

    try:
        proc = subprocess.run(  # noqa: S603 — argv list, no shell
            list(argv),
            env=env,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{Path(str(argv[0])).name} timed out after {timeout:.0f}s"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"failed to invoke {Path(str(argv[0])).name}: {exc}"
        ) from exc

    proc.stdout = proc.stdout or ""
    proc.stderr = scrub_ansi(proc.stderr or "")
    return proc
