"""Google Gemini CLI fingerprint helpers.

Hermes' Code Assist (cloudcode-pa) and Generative Language API paths must
present a User-Agent that Google's backend recognizes; internal endpoints
have been observed to reject unknown UAs.

The real ``@google/gemini-cli`` v0.44.1 (verified by inspecting the bundled
chunks in the locally installed npm package) builds its UA like::

    User-Agent: GeminiCLI/<pkg.version>/<model> (<process.platform>; <process.arch>; <surface>)
    X-Goog-Api-Client: gl-node/<process.versions.node> gccl/<pkg.version>

This module produces equivalent strings, with the package version pulled
from the npm registry (cached on disk, like
:mod:`hermes_cli.copilot_auth` does for VS Code / Copilot Chat versions),
so we stay current without manual bumps.

Env overrides
-------------
- ``HERMES_GEMINI_CLI_VERSION``      pin pkg version (skip npm lookup)
- ``HERMES_GEMINI_NODE_VERSION``     pin reported node version
- ``HERMES_GEMINI_CLI_SURFACE``      override the ``surface`` segment
                                     (default ``hermes``; the real CLI uses
                                     values like ``vscode`` / ``unknown``)
- ``HERMES_GEMINI_CLIENT_PROFILE``   ``antigravity`` (default: "Antigravity
                                     2.0", current product), ``antigravity_ide``
                                     (legacy "Antigravity IDE" 2.0.3),
                                     ``cloud_code`` (Gemini Code Assist VS
                                     Code extension), or ``cli`` (bare
                                     ``@google/gemini-cli``).
- ``HERMES_GEMINI_IDE_VERSION``      pin ``clientMetadata.ideVersion`` (skip
                                     auto-fetch from antigravity / vscode
                                     update feeds)
- ``HERMES_GEMINI_PLUGIN_VERSION``   pin ``clientMetadata.pluginVersion``
- ``HERMES_GEMINI_IDE_TYPE``         override ``clientMetadata.ideType``;
                                     accepts string names (``ANTIGRAVITY``,
                                     ``VSCODE``, ``INTELLIJ``, …) or numeric
                                     enum values (``9`` = ANTIGRAVITY in the
                                     Code Assist proto, ``2`` = INTELLIJ /
                                     ``ANTIGRAVITY`` in the Antigravity-
                                     internal proto, ``1`` = VSCODE). Numeric
                                     strings are emitted as JSON ints; names
                                     are emitted as strings.
- ``HERMES_GEMINI_PLUGIN_TYPE``      override ``clientMetadata.pluginType``;
                                     A/B test ``CLOUD_CODE`` / ``GEMINI`` /
                                     ``AIPLUGIN_INTELLIJ`` /
                                     ``AIPLUGIN_STUDIO`` / ``ANTIGRAVITY``.
                                     Note: the Code Assist PluginType enum
                                     does not define ``ANTIGRAVITY``; sending
                                     it tests server tolerance.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Hardcoded fallbacks. Bumped opportunistically; runtime resolution prefers
# the live npm registry so these only matter when offline / blocked.
_GEMINI_CLI_VERSION_FALLBACK = "0.44.1"
_NODE_VERSION_FALLBACK = "24.0.0"
_DEFAULT_SURFACE = "hermes"

_GEMINI_CLI_NPM_URL = "https://registry.npmjs.org/@google/gemini-cli/latest"
_VERSION_CACHE_TTL = 24 * 60 * 60  # 24h, matches copilot_auth pattern
_VERSION_CACHE_PATH = Path.home() / ".cache" / "hermes" / "gemini_cli_version.json"

# In-process memo so the hot path doesn't touch disk on every header build.
_version_memo: Optional[tuple[str, float]] = None


def _gemini_cli_version() -> str:
    """Return the latest ``@google/gemini-cli`` version on npm.

    Resolution order mirrors :func:`hermes_cli.copilot_auth._latest_copilot_cli_version`:
      1. ``HERMES_GEMINI_CLI_VERSION`` env override.
      2. In-process memo (TTL ``_VERSION_CACHE_TTL``).
      3. On-disk cache at ``_VERSION_CACHE_PATH``.
      4. ``GET https://registry.npmjs.org/@google/gemini-cli/latest``.
      5. Hard fallback ``_GEMINI_CLI_VERSION_FALLBACK``.

    Network failures are swallowed; we always return *something*.
    """
    override = os.getenv("HERMES_GEMINI_CLI_VERSION", "").strip()
    if override:
        return override

    global _version_memo
    now = time.time()
    if _version_memo and now - _version_memo[1] < _VERSION_CACHE_TTL:
        return _version_memo[0]

    cache_path = _VERSION_CACHE_PATH
    try:
        if cache_path.is_file():
            data = json.loads(cache_path.read_text())
            ver = str(data.get("version") or "").lstrip("v").strip()
            ts = float(data.get("fetched_at") or 0)
            if ver and now - ts < _VERSION_CACHE_TTL:
                _version_memo = (ver, ts)
                return ver
    except Exception as exc:
        logger.debug("gemini-cli version cache read failed: %s", exc)

    ver = _GEMINI_CLI_VERSION_FALLBACK
    try:
        import urllib.request

        req = urllib.request.Request(
            _GEMINI_CLI_NPM_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": "gemini-cli",
            },
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload = json.loads(resp.read().decode())
        latest = str(payload.get("version") or "").lstrip("v").strip()
        if latest:
            ver = latest
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"version": ver, "fetched_at": now})
                )
            except Exception as exc:
                logger.debug("gemini-cli version cache write failed: %s", exc)
    except Exception as exc:
        logger.debug(
            "failed to fetch latest @google/gemini-cli version, using fallback %s: %s",
            _GEMINI_CLI_VERSION_FALLBACK,
            exc,
        )

    _version_memo = (ver, now)
    return ver


def _node_version() -> str:
    """Return the node version we report in the UA / X-Goog-Api-Client."""
    return os.getenv("HERMES_GEMINI_NODE_VERSION", "").strip() or _NODE_VERSION_FALLBACK


def _surface() -> str:
    """Return the ``surface`` segment of the UA.

    The real gemini-cli reads this from ``GEMINI_CLI_SURFACE`` and falls
    back to detection (``vscode``, ``cursor``, …) or ``unknown``. We default
    to ``hermes`` so Google's logs can attribute traffic correctly.
    """
    return os.getenv("HERMES_GEMINI_CLI_SURFACE", "").strip() or _DEFAULT_SURFACE


def _process_platform() -> str:
    """Mimic Node's ``process.platform`` (``linux``/``darwin``/``win32``)."""
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform in ("win32", "cygwin"):
        return "win32"
    return sys.platform


def _process_arch() -> str:
    """Mimic Node's ``process.arch`` (``x64``/``arm64``/``ia32``)."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine in ("i386", "i686", "x86"):
        return "ia32"
    return machine or "x64"


def gemini_cli_user_agent(model: str = "") -> str:
    """Return a User-Agent string matching gemini-cli's wire format.

    Examples::

        GeminiCLI/0.44.1 (linux; x64)
        GeminiCLI/0.44.1/gemini-2.5-pro (linux; x64)
    """
    ver = _gemini_cli_version()
    head = f"GeminiCLI/{ver}/{model}" if model else f"GeminiCLI/{ver}"
    return f"{head} ({_process_platform()}; {_process_arch()}; {_surface()})"


def gemini_cli_x_goog_api_client() -> str:
    """Return the ``X-Goog-Api-Client`` header value gemini-cli sends."""
    return f"gl-node/{_node_version()} gccl/{_gemini_cli_version()}"


# ---------------------------------------------------------------------------
# clientMetadata (request-body) helpers
# ---------------------------------------------------------------------------
#
# The cloudcode-pa endpoints accept (require, on some routes) a
# ``clientMetadata`` field describing who the caller is. Two profiles
# observed in real Google clients:
#
# 1. ``cli``         : what ``@google/gemini-cli`` sends. Three fields, all
#                      "unspecified". This is the path Hermes used to take.
# 2. ``cloud_code``  : what the official Gemini Code Assist VS Code extension
#                      (``google.geminicodeassist``) sends. Claims to be an
#                      IDE plugin: uppercase platform like ``LINUX_AMD64``,
#                      ``ideType: "VSCODE"`` (or workstation / cloud-shell
#                      variants), ``pluginType: "CLOUD_CODE"``, plus
#                      ``ideName``, ``ideVersion``, ``pluginVersion``,
#                      ``updateChannel``.
# 3. ``antigravity`` : claims to be Google Antigravity 2.0 (the current
#                      product, ``https://antigravity.google/product/antigravity-2``).
#                      The IdeType enum in the Code Assist proto includes
#                      ``ANTIGRAVITY=9``, so the server recognizes this
#                      value. Antigravity 2.0 is an Electron shell that
#                      spawns a Codeium-style ``language_server`` binary
#                      with these flags (verified by extracting the asar
#                      and stringing the LS binary):
#
#                          --override_ide_name antigravity
#                          --subclient_type hub
#                          --override_ide_version <app.getVersion()>
#                          --override_user_agent_name antigravity
#                          --cloud_code_endpoint \
#                              https://daily-cloudcode-pa.googleapis.com
#
#                      The LS then maps ``ide_name=antigravity`` to
#                      ``ClientMetadata.ideType=ANTIGRAVITY`` and emits
#                      ``ideName: "antigravity"`` (lowercase) to the
#                      cloudcode-pa endpoint. Note that ``subclient_type``
#                      is internal to the Jetski/Codeium LS proto and is
#                      NOT a cloudcode-pa ``ClientMetadata`` field.
# 4. ``antigravity_ide`` : legacy Antigravity *IDE* 2.0.3 fingerprint
#                      (``ideName: "Antigravity IDE"``). Use as a fallback
#                      if the new ``antigravity`` (2.0) fingerprint is
#                      rejected server-side.
#
# The ``antigravity`` profile is the default; it tracks the freshest
# product. Override with ``HERMES_GEMINI_CLIENT_PROFILE=antigravity_ide``
# / ``cloud_code`` / ``cli`` to switch.

_CLIENT_PROFILE_DEFAULT = "antigravity"

# Pinned IDE / plugin versions. These only need to be plausible; Google
# doesn't gate on specific values, but they should look like a real recent
# install.
_GEMINI_CODE_ASSIST_PLUGIN_VERSION_FALLBACK = "2.86.0"
_VSCODE_VERSION_FALLBACK = "1.95.0"
_ANTIGRAVITY_VERSION_FALLBACK = "2.0.10"      # current Antigravity 2.0
_ANTIGRAVITY_IDE_VERSION_FALLBACK = "2.0.3"   # legacy Antigravity IDE

# Public version sources for Antigravity 2.0. Tried in order. The brew API
# is canonical and refreshed within hours of a release; the chocolatey
# manifest is community-maintained but cleanly versioned; the Cloud Run
# feed only covers the legacy *Antigravity IDE* (Electron fork, separate
# product line, last seen at 2.0.3). All three are queried with short
# timeouts and any failure falls through.
_ANTIGRAVITY_VERSION_SOURCES: tuple[tuple[str, str], ...] = (
    ("brew", "https://formulae.brew.sh/api/cask/antigravity.json"),
    (
        "choco",
        "https://raw.githubusercontent.com/targed/chocolatey-projects/refs/heads/main/Antigravity/antigravity.nuspec",
    ),
)
# Same pattern for the legacy *Antigravity IDE* (Electron fork). The brew
# cask ``antigravity-ide`` is canonical; the Cloud Run releases feed
# (Google's own auto-updater) is the secondary source.
_ANTIGRAVITY_IDE_VERSION_SOURCES: tuple[tuple[str, str], ...] = (
    ("brew", "https://formulae.brew.sh/api/cask/antigravity-ide.json"),
    (
        "cloud_run",
        "https://antigravity-ide-auto-updater-974169037036.us-central1.run.app/releases",
    ),
)
_ANTIGRAVITY_VERSION_CACHE_PATH = (
    Path.home() / ".cache" / "hermes" / "antigravity_version.json"
)
_ANTIGRAVITY_IDE_VERSION_CACHE_PATH = (
    Path.home() / ".cache" / "hermes" / "antigravity_ide_version.json"
)
_antigravity_version_memo: Optional[tuple[str, float]] = None
_antigravity_ide_version_memo: Optional[tuple[str, float]] = None


def _parse_brew_version(blob: bytes) -> str:
    """Pull the cask version from formulae.brew.sh JSON, stripping build suffix."""
    payload = json.loads(blob.decode())
    raw = str(payload.get("version") or "")
    # brew encodes "version,build"; we only want the version half.
    return raw.split(",", 1)[0].lstrip("v").strip()


def _parse_nuspec_version(blob: bytes) -> str:
    """Pull ``<version>...</version>`` from a chocolatey nuspec without an XML dep."""
    text = blob.decode("utf-8", errors="replace")
    import re

    m = re.search(r"<version>\s*([^<\s]+)\s*</version>", text)
    return (m.group(1).lstrip("v").strip() if m else "")


def _parse_cloud_run_releases(blob: bytes) -> str:
    """Pull the newest version from the Cloud Run ``/releases`` JSON array."""
    payload = json.loads(blob.decode())
    if isinstance(payload, list) and payload:
        return str(payload[0].get("version") or "").lstrip("v").strip()
    return ""


_ANTIGRAVITY_VERSION_PARSERS = {
    "brew": _parse_brew_version,
    "choco": _parse_nuspec_version,
    "cloud_run": _parse_cloud_run_releases,
}


def _fetch_cached_version(
    cache_path: Path,
    memo: Optional[tuple[str, float]],
    fetch: callable,
    fallback: str,
    label: str,
) -> tuple[str, tuple[str, float]]:
    """Shared env-override → memo → disk → network → fallback resolver."""
    now = time.time()
    if memo and now - memo[1] < _VERSION_CACHE_TTL:
        return memo[0], memo

    try:
        if cache_path.is_file():
            data = json.loads(cache_path.read_text())
            ver = str(data.get("version") or "").lstrip("v").strip()
            ts = float(data.get("fetched_at") or 0)
            if ver and now - ts < _VERSION_CACHE_TTL:
                return ver, (ver, ts)
    except Exception as exc:
        logger.debug("%s version cache read failed: %s", label, exc)

    ver = fallback
    try:
        latest = fetch()
        if latest:
            ver = latest
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"version": ver, "fetched_at": now})
                )
            except Exception as exc:
                logger.debug("%s version cache write failed: %s", label, exc)
    except Exception as exc:
        logger.debug("failed to fetch latest %s version, using fallback %s: %s", label, fallback, exc)

    return ver, (ver, now)


def _fetch_from_sources(
    sources: tuple[tuple[str, str], ...],
    label_prefix: str,
) -> str:
    """Walk a (label, url) list using ``_ANTIGRAVITY_VERSION_PARSERS``."""
    import urllib.request

    for label, url in sources:
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json", "User-Agent": "hermes-cli"}
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                blob = resp.read()
            ver = _ANTIGRAVITY_VERSION_PARSERS[label](blob)
            if ver:
                return ver
        except Exception as exc:
            logger.debug("%s version source %s failed: %s", label_prefix, label, exc)
    return ""


def _antigravity_version() -> str:
    """Return the latest **Antigravity 2.0** (current product) version.

    Tries brew, then chocolatey, then falls back. ``HERMES_GEMINI_IDE_VERSION``
    overrides everything.
    """
    override = os.getenv("HERMES_GEMINI_IDE_VERSION", "").strip()
    if override:
        return override

    global _antigravity_version_memo
    ver, _antigravity_version_memo = _fetch_cached_version(
        _ANTIGRAVITY_VERSION_CACHE_PATH,
        _antigravity_version_memo,
        lambda: _fetch_from_sources(_ANTIGRAVITY_VERSION_SOURCES, "antigravity"),
        _ANTIGRAVITY_VERSION_FALLBACK,
        "antigravity",
    )
    return ver


def _antigravity_ide_version() -> str:
    """Return the latest legacy **Antigravity IDE** version (Electron fork).

    Tries brew (cask ``antigravity-ide``), then the Cloud Run releases
    feed, then falls back. ``HERMES_GEMINI_IDE_VERSION`` overrides.
    """
    override = os.getenv("HERMES_GEMINI_IDE_VERSION", "").strip()
    if override:
        return override

    global _antigravity_ide_version_memo
    ver, _antigravity_ide_version_memo = _fetch_cached_version(
        _ANTIGRAVITY_IDE_VERSION_CACHE_PATH,
        _antigravity_ide_version_memo,
        lambda: _fetch_from_sources(_ANTIGRAVITY_IDE_VERSION_SOURCES, "antigravity_ide"),
        _ANTIGRAVITY_IDE_VERSION_FALLBACK,
        "antigravity_ide",
    )
    return ver


def _ide_platform() -> str:
    """Match Code Assist's ``platform()`` getter (uppercase, with arch)."""
    if sys.platform == "darwin":
        return "DARWIN_ARM64" if _process_arch() == "arm64" else "DARWIN_AMD64"
    if sys.platform in ("win32", "cygwin"):
        return "WINDOWS_AMD64"
    # linux + everything else falls through to the linux path Code Assist uses
    return "LINUX_ARM64" if _process_arch() == "arm64" else "LINUX_AMD64"


def _ide_type() -> str:
    """Match Code Assist's ``ideType()`` getter."""
    if os.getenv("GOOGLE_CLOUD_WORKSTATIONS"):
        return "VSCODE_CLOUD_WORKSTATION"
    if os.getenv("CLOUD_SHELL", "").lower() == "true":
        return "CLOUD_SHELL"
    return "VSCODE"


def _client_profile() -> str:
    return (
        os.getenv("HERMES_GEMINI_CLIENT_PROFILE", "").strip().lower()
        or _CLIENT_PROFILE_DEFAULT
    )


def gemini_client_metadata(profile: Optional[str] = None) -> dict:
    """Return the ``clientMetadata`` body field for cloudcode-pa requests.

    ``profile`` overrides the env / default selection. Use ``"cli"`` to
    impersonate the bare ``@google/gemini-cli``; ``"cloud_code"`` (default)
    to impersonate the Gemini Code Assist VS Code extension.
    """
    chosen = (profile or _client_profile()).lower()
    if chosen == "cli":
        meta: dict = {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
    else:
        plugin_version = (
            os.getenv("HERMES_GEMINI_PLUGIN_VERSION", "").strip()
            or _GEMINI_CODE_ASSIST_PLUGIN_VERSION_FALLBACK
        )
        if chosen == "antigravity":
            meta = {
                "ideType": "ANTIGRAVITY",
                "ideVersion": _antigravity_version(),
                # Lowercase "antigravity" matches the literal value the
                # Antigravity 2.0 Electron shell passes to its bundled
                # language server via ``--override_ide_name antigravity``.
                "ideName": "antigravity",
                "platform": _ide_platform(),
                "pluginType": "CLOUD_CODE",
                "pluginVersion": plugin_version,
                "updateChannel": "stable",
            }
        elif chosen == "antigravity_ide":
            meta = {
                "ideType": "ANTIGRAVITY",
                "ideVersion": _antigravity_ide_version(),
                "ideName": "Antigravity IDE",
                "platform": _ide_platform(),
                "pluginType": "CLOUD_CODE",
                "pluginVersion": plugin_version,
                "updateChannel": "stable",
            }
        else:
            # cloud_code: fingerprint matches google.geminicodeassist
            ide_version = (
                os.getenv("HERMES_GEMINI_IDE_VERSION", "").strip()
                or _VSCODE_VERSION_FALLBACK
            )
            meta = {
                "ideType": _ide_type(),
                "ideVersion": ide_version,
                "ideName": "Visual Studio Code",
                "platform": _ide_platform(),
                "pluginType": "CLOUD_CODE",
                "pluginVersion": plugin_version,
                "updateChannel": "stable",
            }

    # Final A/B-testing overrides: apply after profile resolution so they
    # work for every profile (including ``cli``).
    ide_type_override = os.getenv("HERMES_GEMINI_IDE_TYPE", "").strip()
    if ide_type_override:
        # Numeric strings emit as JSON ints (some servers expect proto enum
        # ints, others expect string names). Names emit as-is, uppercased.
        if ide_type_override.lstrip("-").isdigit():
            meta["ideType"] = int(ide_type_override)
        else:
            meta["ideType"] = ide_type_override.upper()
    plugin_type_override = os.getenv("HERMES_GEMINI_PLUGIN_TYPE", "").strip()
    if plugin_type_override:
        if plugin_type_override.lstrip("-").isdigit():
            meta["pluginType"] = int(plugin_type_override)
        else:
            meta["pluginType"] = plugin_type_override.upper()
    return meta
