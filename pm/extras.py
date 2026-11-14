"""Extras: the runtime features of the python venv.

Feature names ARE pyproject extra names. This table maps each extra to the
import that proves it (the anchor), so availability is one find_spec — no
package-manager machinery on the hot path. sync_venv([extra]) makes an extra
true; pm owns HOW (uv sync inside the venv package).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable


# extra name -> module that proves it is installed
ANCHORS: dict[str, str | tuple[str, ...]] = {
    "anthropic": "anthropic",
    "bedrock": "boto3",
    "vertex": "google.auth",
    "azure-identity": "azure.identity",
    "exa": "exa_py",
    "firecrawl": "firecrawl",
    "parallel-web": "parallel",
    "ddgs": "ddgs",
    "otlp": "opentelemetry.sdk",
    "mistral": "mistralai",
    "edge-tts": "edge_tts",
    "neutts": "neutts",
    "kittentts": ("kittentts", "soundfile"),
    "piper": "piper",
    "tts-premium": "elevenlabs",
    "voice": "faster_whisper",
    "stt-whisper": "faster_whisper",
    "audio-io": ("sounddevice", "numpy"),
    "silk": "pilk",
    "wake": "pyopen_wakeword",
    "wake-openwakeword": "pyopen_wakeword",
    "wake-sherpa": "sherpa_onnx",
    "wake-porcupine": "pvporcupine",
    "fal": "fal_client",
    "messaging": "telegram",
    "telegram": "telegram",
    "discord": "discord",
    "slack": "slack_bolt",
    "matrix": ("mautrix", "asyncpg", "aiosqlite", "markdown", "aiohttp_socks"),
    "dingtalk": "dingtalk_stream",
    "feishu": "lark_oapi",
    "wecom": "defusedxml",
    "keychain": "keyring",
    "teams": "microsoft_teams.apps",
    "modal": "modal",
    "daytona": "daytona",
    "vercel": "vercel",
    "google": ("googleapiclient", "google.auth", "google_auth_oauthlib.flow", "google_auth_httplib2"),
    "google-chat": "google.cloud.pubsub_v1",
    "google-meet": ("playwright.sync_api", "websockets"),
    "youtube": "youtube_transcript_api",
    "acp": "acp",
    "web": "fastapi",
    "doc-extract": "anydoc",
    "documents": ("docx", "openpyxl", "pptx", "reportlab", "pypdf"),
    "documents-pdf": ("pdfplumber", "pypdfium2"),
    "computer-use": "mcp",
    "trace-upload": "huggingface_hub",
    # Pillow resize recovery for vision tools (the `vision` extra is a no-op
    # back-compat alias — Pillow is core — but ensure_import("vision") must
    # still resolve an anchor so availability checks work).
    "vision": "PIL",
}


def _anchors(extra: str) -> tuple[str, ...]:
    got = ANCHORS.get(extra, extra.replace("-", "_"))
    return got if isinstance(got, tuple) else (got,)


def _importable(anchor: str) -> bool:
    """An anchor already present in sys.modules counts even without a
    findable spec — tests fake SDKs by inserting modules there, and the
    caller's import right after this check resolves the same way."""
    import sys

    if anchor in sys.modules:
        return True
    try:
        return importlib.util.find_spec(anchor) is not None
    except (ImportError, ValueError):
        return False


def available(extra: str) -> bool:
    """Fast, side-effect-free: are all of this extra's anchors importable?
    A platform-gated extra whose gate excludes this machine reads as
    unavailable — the anchors are absent by design, not by accident."""
    if not extra_supported(extra):
        return False
    return all(_importable(a) for a in _anchors(extra))


def _platform_gates() -> dict[str, str]:
    """The [tool.hermes.extras-platforms] table from pyproject.toml:
    extra -> PEP 508 marker string. Cached per process."""
    global _PLATFORM_GATES
    if _PLATFORM_GATES is not None:
        return _PLATFORM_GATES
    import tomllib

    from pm.paths import repo_root

    gates: dict[str, str] = {}
    try:
        with (repo_root() / "pyproject.toml").open("rb") as f:
            data = tomllib.load(f)
        table = data.get("tool", {}).get("hermes", {}).get("extras-platforms", {})
        if isinstance(table, dict):
            gates = {str(k): str(v) for k, v in table.items()}
    except (OSError, ValueError):
        pass
    _PLATFORM_GATES = gates
    return gates


_PLATFORM_GATES: dict[str, str] | None = None


def extra_supported(extra: str, *, environment: dict[str, str] | None = None,
                    importable: Callable[[str], bool] | None = None) -> bool:
    """Is this extra installable on THIS platform? True when the extra
    carries no gate, or its marker matches the running platform. An
    extra that IS present on this machine (anchors importable) is always
    supported — an installed override beats the table (dev machines,
    hand-synced venvs)."""
    if all((importable or _importable)(a) for a in _anchors(extra)):
        return True
    marker = _platform_gates().get(extra)
    if marker is None:
        return True
    import os
    import platform
    import sys

    if environment is None:
        environment = {
            "sys_platform": sys.platform,
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "os_name": os.name,
        }
    try:
        from packaging.markers import Marker
    except ImportError:
        # The historical update takeover runs PM in the OLD venv's
        # interpreter, which need not ship `packaging`. PM's runtime does.
        return _evaluate_in_runtime(marker, environment)
    try:
        return bool(Marker(marker).evaluate(environment=environment))
    except Exception:
        # A malformed marker must never brick availability — treat as
        # ungated and let the resolver be the authority.
        return True


def _evaluate_in_runtime(marker: str, environment: dict[str, str]) -> bool:
    import json
    import subprocess

    from pm.runtime import runtime_command, runtime_environment

    try:
        command = runtime_command(Path(__file__).with_name("_marker_eval.py"),
                                  [marker, json.dumps(environment)])
        result = subprocess.run(command, env=runtime_environment(), capture_output=True,
                                text=True, timeout=60, check=False)
    except Exception:
        return True  # same fallback as a malformed marker: the resolver decides
    if result.returncode != 0:
        return True
    return result.stdout.strip() == "1"


_INTERACTIVE_INSTALL_PROMPT_SUPPRESSIONS = 0


def interactive_install_prompts_suppressed() -> bool:
    """Whether this process is inside a non-interactive lazy-install scope."""
    return _INTERACTIVE_INSTALL_PROMPT_SUPPRESSIONS > 0


def suppress_interactive_install_prompts():
    """Process-wide scope that prevents lazy installs from reading stdin.

    Gateway/dashboard runtimes may inherit a TTY from tmux or another supervisor,
    but there is no operator waiting at stdin while adapters start. The counter is
    process-wide (not a ContextVar) so worker threads obey the same daemon policy;
    nesting remains safe and the interactive default is restored on exit.
    """
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        global _INTERACTIVE_INSTALL_PROMPT_SUPPRESSIONS
        _INTERACTIVE_INSTALL_PROMPT_SUPPRESSIONS += 1
        try:
            yield
        finally:
            _INTERACTIVE_INSTALL_PROMPT_SUPPRESSIONS -= 1

    return _scope()


def install_hint(extra: str) -> str:
    """The one command users are told to run for a missing extra."""
    return f"hermes pm install --extra {extra}"


def ensure_import(extra: str) -> None:
    """Make an extra available: no-op when the anchor imports, otherwise
    sync the venv with the extra enabled. Raises InstallError on failure
    — including when a platform gate excludes this machine."""
    if available(extra):
        return
    if not extra_supported(extra):
        from pm.package import InstallError

        marker = _platform_gates().get(extra, "")
        raise InstallError(
            "venv",
            f"extra {extra!r} is not supported on this platform "
            f"(gate: {marker!r}); the adapter degrades without it",
        )
    import sys
    from pm.package import InstallError

    # prompt_toolkit already owns stdin during a CLI turn; never read from it.
    app_running = False
    if "prompt_toolkit.application.current" in sys.modules:
        from prompt_toolkit.application.current import get_app_or_none

        app_running = bool(getattr(get_app_or_none(), "is_running", False))
    # A daemon suppression scope means stdin is not an interaction surface even
    # when a supervisor/tmux left it attached to a TTY. Fold that into the
    # existing prompt-unavailable gate so policy checks can remain orthogonal.
    if interactive_install_prompts_suppressed():
        app_running = True
    if not app_running and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            answer = input(f"\nThis needs Hermes' optional {extra!r} feature, which isn't installed yet.\n"
                           "Install it now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer and answer not in {"y", "yes"}:
            raise InstallError("venv", f"installation of extra {extra!r} declined")
    from pm.client import sync_venv

    sync_venv([extra])
    # The sync published a new generation. Swap this process onto it when nothing
    # already imported would change underneath it (adopt_selected); otherwise only
    # a restart can load it.
    from pm.environments import selected_venv, site_packages
    from pm.environments_adopt import adopt_selected, note_restart_pending, restart_needed
    from pm.paths import repo_root, runtime_facts_path
    import sys
    from pathlib import Path

    if runtime_facts_path().is_file():
        root = repo_root()
        adopt_selected(root)
        selected = site_packages(selected_venv(root)).resolve()
        if selected not in {Path(entry).resolve() for entry in sys.path}:
            reason = restart_needed(root) or "this process does not run from the install's dependency environment"
            note_restart_pending(f"{extra} installed ({reason})")
            raise InstallError("venv", f"{extra} installed; restart Hermes to activate it ({reason})")


def ensure_and_bind(extra, importer, target_globals) -> bool:
    """ensure_import + rebind module-level names after a mid-process install.
    importer returns {name: value}; bound into target_globals on success."""
    try:
        ensure_import(extra)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("extra %r unavailable: %s", extra, exc)
        return False
    try:
        bindings = importer()
    except ImportError as exc:
        import logging

        logging.getLogger(__name__).warning(
            "import after installing %r failed: %s", extra, exc
        )
        return False
    target_globals.update(bindings)
    return True


def missing(extra: str) -> tuple[str, ...]:
    return tuple(a for a in _anchors(extra) if not _importable(a))


def _installed_in(site_packages: Path, anchor: str) -> bool:
    """Is ``anchor`` installed under a site-packages we must not import from?"""
    *parents, leaf = anchor.split(".")
    parent = site_packages.joinpath(*parents)
    if (parent / leaf).is_dir():
        return True
    # A top-level module can be a plain file or a compiled extension.
    return any(entry.name.rsplit(".", 1)[-1] in {"py", "so", "pyd"}
               for entry in parent.glob(f"{leaf}.*"))


DECLINED_EXTRAS_FILENAME = "declined-extras.json"


def declined_extras_path(project_root: Path | None = None) -> Path:
    from pm.environments import install_state_dir
    from pm.paths import repo_root

    return install_state_dir(repo_root() if project_root is None else Path(project_root)) / DECLINED_EXTRAS_FILENAME


def declined_extras(project_root: Path | None = None) -> frozenset[str]:
    """Extras the user dropped with ``hermes pm install --without NAME``.

    Recorded beside the dependency facts, so a carried selection (legacy
    migration, recorded extras, configured-feature auto-installs) never
    re-adds one; only an explicit request (``--extra NAME``) opts back in.
    """
    import json

    path = declined_extras_path(project_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return frozenset()
    except (OSError, ValueError) as exc:
        # Reading a damaged record as "nothing declined" would re-add what the user removed.
        from pm.package import InstallError

        raise InstallError("venv", f"cannot read {path}: {exc}", "fix or delete the file, then retry") from exc
    names = data.get("declined") if isinstance(data, dict) else None
    return frozenset(_canonical(name) for name in names or () if isinstance(name, str))


def record_declined_extras(*, add=(), remove=(), reasons: dict[str, str] | None = None,
                           project_root: Path | None = None) -> frozenset[str]:
    """Update the declined extras; returns the new set. Writes only on change.

    *reasons* marks extras an update left out on its own (``pm.carried_extras``) with why,
    so a later refusal names the cause instead of a ``--without`` the user never ran.
    """
    import json

    from pm.filesystem import durable_write_bytes

    current = declined_extras(project_root)
    updated = (current | {_canonical(name) for name in add}) - {_canonical(name) for name in remove}
    known = declined_reasons(project_root)
    noted = {name: text for name, text in {**known, **{_canonical(name): " ".join(text.split())[:300]
                                                          for name, text in (reasons or {}).items()}}.items()
             if name in updated}
    if updated != current or noted != known:
        record: dict = {"schema": 1, "declined": sorted(updated)}
        if noted:
            record["left_out_by_update"] = dict(sorted(noted.items()))
        durable_write_bytes(declined_extras_path(project_root), (json.dumps(record, indent=2) + "\n").encode("utf-8"))
    return frozenset(updated)


def declined_reasons(project_root: Path | None = None) -> dict[str, str]:
    """Why an update left each automatically declined extra out (absent for a user's ``--without``)."""
    import json

    try:
        data = json.loads(declined_extras_path(project_root).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}  # declined_extras() is the reader that refuses a damaged record
    notes = data.get("left_out_by_update") if isinstance(data, dict) else None
    return {name: text for name, text in (notes or {}).items() if isinstance(name, str) and isinstance(text, str)}


def _canonical(name: str) -> str:
    """The PEP 685 name uv matches an extra by (``foo_bar`` is ``foo-bar``)."""
    import re

    return re.sub(r"[-_.]+", "-", name).lower()


def droppable_extras(project_root: Path) -> list[str]:
    """Declared extras ``--without`` may drop: not ``all`` and nothing ``all`` pulls in.

    The default sync selects ``[all]``, so dropping one of its members would be a no-op.
    """
    import re
    import tomllib

    with (Path(project_root) / "pyproject.toml").open("rb") as f:
        project = tomllib.load(f).get("project", {})
    table = project.get("optional-dependencies", {})
    self_ref = re.compile(rf"^{re.escape(project.get('name', ''))}\[([^\]]+)\]", re.IGNORECASE)
    kept, pending = {"all"}, ["all"]
    while pending:
        for requirement in table.get(pending.pop(), ()):
            match = self_ref.match(requirement.replace(" ", ""))
            for member in (match.group(1).split(",") if match else ()):
                if member not in kept:
                    kept.add(member)
                    pending.append(member)
    return sorted(extra for extra in table if extra not in kept)


def _pre_pm_site_packages(root: Path) -> list[Path]:
    """The site-packages of the checkout's own main-era ``venv/`` or ``.venv/`` (POSIX or Windows layout).

    A venv counts when Hermes is installed in it and PM did not build it. What Hermes was installed
    from says nothing: a pre-PM updater reinstalls the checkout into the old venv from the tree it
    just pulled, PM included. A bare ``uv venv`` holds no Hermes, and an environment PM built
    (``pm.build_env --out .venv``) is no install's old state: reading either would union its
    contents into every sync of a data root with no recorded selection.
    """
    return [tree for venv in (root / "venv", root / ".venv") if not _built_by_pm(venv)
            for tree in (*venv.glob("lib/python*/site-packages"), venv / "Lib" / "site-packages")
            if tree.is_dir() and any(tree.glob("hermes_agent-*.dist-info"))]


def _built_by_pm(venv: Path) -> bool:
    """Did PM build ``venv``? No pre-PM installer used PM's environment builder.

    PM marks every environment it creates in ``pyvenv.cfg`` (``pm.environment.PM_BUILT_KEY``), on
    any interpreter, ``pm.build_env --python`` included. One built before that marker runs a Python
    from PM's tool store: ``pyvenv.cfg``'s ``home`` is that interpreter's directory inside a store
    entry (``bin/``, the entry itself on Windows), and the store's ``facts.json`` holds a ``python``
    row. An older pin's entry still counts, so a test env survives the store moving to a newer Python.
    """
    import json

    from pm.environment import PM_BUILT_KEY  # noqa: TID251

    try:
        lines = (venv / "pyvenv.cfg").read_text(encoding="utf-8-sig").splitlines()
    except (OSError, ValueError):  # unreadable or not UTF-8: no PM-built env writes that
        return False
    cfg = {key.strip(): value.strip() for key, _, value in (line.partition("=") for line in lines)}
    if cfg.get(PM_BUILT_KEY) == "true":
        return True
    home = cfg.get("home", "")
    if not home:
        return False
    for entry in (Path(home), *Path(home).parents):
        if not entry.name.startswith("python-"):
            continue
        try:
            packages = json.loads((entry.parent / "facts.json").read_text(encoding="utf-8-sig"))["packages"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(packages, dict) and "python" in packages:
            return True
    return False


def legacy_selection(project_root: Path) -> list[str]:
    """The extras PM's first generation selects when it replaces a main-era venv.

    Main-era installers selected ``[all]`` and then lazily installed opt-in
    extras (FAL, messaging SDKs, ...) into the checkout's own venv, with no
    ledger. Selecting only ``[all]`` drops those, and the first launch after
    the migration asks to reinstall a feature that already worked. The old
    venv's site-packages is read, never imported: the migrating process may
    not run from it.
    """
    root = Path(project_root)
    trees = _pre_pm_site_packages(root)
    # The old venv is never deleted, so this scan repeats for any data root without facts.
    refused = declined_extras(root)
    carried = sorted(
        # An umbrella extra shares its anchor with one member; carrying it
        # would install every sibling the user never chose.
        extra for extra in ANCHORS if extra not in {"messaging", "voice", "wake"} and extra not in refused
        # PM refuses a gated extra outside its platform even if a hand-synced venv carried it.
        # Installed first: judging a gate may cost a PM-runtime subprocess.
        if any(all(_installed_in(tree, anchor) for anchor in _anchors(extra)) for tree in trees)
        and extra_supported(extra, importable=lambda _anchor: False)
    )
    return ["all", *carried]


def venv_selection_recorded(project_root: Path) -> bool:
    """Has PM recorded this install's dependency selection (the ``venv`` row of its facts)?

    That row, not the facts file, ends a pre-PM venv's carry: a facts file can exist without it. A
    damaged install facts file raises instead of reading as "none recorded".
    """
    from pm.environments import runtime_facts_path, store_root
    from pm.lock import Facts

    root = Path(project_root)
    # The same two rows a sync reads its recorded selection from (a payload's store records one).
    return (Facts(runtime_facts_path(root), strict=True).get("venv") is not None
            or Facts(store_root(root) / "facts.json").get("venv") is not None)


def first_generation_selection(project_root: Path) -> list[str] | None:
    """The extras a sync of the install's whole selection requests: ``None`` once PM recorded a
    selection (that ledger owns the extras), else ``[all]``, as ``hermes pm install`` selects.

    The update's completion bootstrap, the historical takeover, the launch migration and the
    ``venv_sync`` entry pass this; without it a first sync would reach only the core. The old
    venv's extras are not added here: ``sync_venv`` carries them under the dependency lock
    (``pm.carried_extras.with_pre_pm_baseline``).
    """
    return None if venv_selection_recorded(project_root) else ["all"]


def _covered_extras(root: Path, recorded) -> set[str]:
    """PEP 685 keys of the extras the core plus *recorded* already install in full.

    An extra whose every requirement is already there (``computer-use`` is ``mcp``, which ``all``
    pulls in), or that requires nothing (``vision = []``, anchored on core Pillow), reads as present:
    carrying it changes nothing but the stamp, and costs a rebuild.
    """
    import re
    import tomllib

    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    except (OSError, KeyError, ValueError):
        return set()
    def nested(spec: str):
        return re.match(r"^hermes-agent\[([^\]]+)\]", spec)

    def name_of(spec: str) -> str:
        return _canonical(re.split(r"[\s\[<>=!~;@(]", spec, maxsplit=1)[0])

    table = {_canonical(name): specs for name, specs in project.get("optional-dependencies", {}).items()}
    installed = {name_of(spec) for spec in project.get("dependencies", ())}
    seen, todo = set(), [_canonical(extra) for extra in recorded]
    while todo:
        extra = todo.pop()
        if extra in seen:
            continue
        seen.add(extra)
        for spec in table.get(extra, ()):
            if match := nested(spec):
                todo.extend(_canonical(part) for part in match.group(1).split(","))
            else:
                installed.add(name_of(spec))
    # An empty alias (``vision = []``) installs nothing, so every generation already holds it.
    return seen | {extra for extra, specs in table.items()
                   if all(nested(spec) is None and name_of(spec) in installed for spec in specs)}


def committed_generation_extras(project_root: Path) -> list[str]:
    """Extras the committed PM generation holds that its recorded selection lacks (#96180).

    A sync replaces that environment, so an SDK present there and not in the ledger (put back by
    hand after an update dropped it, or a ledger an interrupted migration wrote short) is gone once
    it publishes. Read from site-packages, never imported; umbrellas, declined and gated-off extras
    are never carried, as in ``legacy_selection``.
    """
    from pm.environments import _recorded_venv, runtime_facts_path, site_packages
    from pm.lock import Facts

    root = Path(project_root)
    try:
        fact = Facts(runtime_facts_path(root)).get("venv") or {}
        venv = _recorded_venv(root)  # the generation PM committed; a sealed payload carries its own record
    except (RuntimeError, ValueError, OSError):
        return []
    if venv is None or not isinstance(fact.get("extras"), list):
        return []
    trees = [tree for tree in (site_packages(venv), *venv.glob("lib/python*/site-packages")) if tree.is_dir()]
    held = _covered_extras(root, fact["extras"])
    refused = declined_extras(root)
    return sorted(
        extra for extra in ANCHORS if extra not in {"messaging", "voice", "wake"} and _canonical(extra) not in held
        and extra not in refused
        and any(all(_installed_in(tree, anchor) for anchor in _anchors(extra)) for tree in trees)
        and extra_supported(extra, importable=lambda _anchor: False))


def pre_pm_baseline(project_root: Path) -> list[str]:
    """What a request on a pre-PM tree unions into: the old venv's selection, or ``[]``.

    PM unions a request into the recorded selection. A tree with a main-era venv and no recorded
    selection runs that venv, so its extras are the selection every sync extends (``hermes setup``
    adding ``telegram``, ``hermes pm update``'s venv step); otherwise its first generation records
    the request alone and the re-scan that would carry the rest never runs again. ``[]`` once a
    selection is recorded, and on a tree with no old venv (nothing to keep).
    """
    root = Path(project_root)
    if venv_selection_recorded(root) or not _pre_pm_site_packages(root):
        return []
    return legacy_selection(root)
