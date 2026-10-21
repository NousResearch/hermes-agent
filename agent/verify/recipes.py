"""Static run-recipe detection for project verification.

Ported nearly 1:1 from superagent-ai/grok-cli ``src/verify/recipes.ts`` (same
detection order and command choices). Layer ownership: ``detect_project_facts``
in :mod:`agent.coding_context` owns the cheap, byte-stable prompt-time facts —
never push runtime detection into it. This module owns the deep runtime recipe
(framework, bootstrap/build/test, start command, port, readiness path) that lets
``hermes verify`` boot the app; the CLI merges project-facts verify commands the
recipe missed (``hermes_cli.verify_cmd._merge_project_facts_commands``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _as_strings(value: Any) -> list[str]:
    values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    return [v.strip() for v in values if isinstance(v, str) and v.strip()]


@dataclass
class Recipe:
    """A runnable verification recipe (grok-cli's ``VerifyRecipe``): ``name`` is the
    human label (``appLabel``), ``kind`` the detector id (``appKind``); command
    lists are shell strings executed in the project root."""

    name: str
    kind: str = "unknown"
    bootstrap: list[str] = field(default_factory=list)
    build: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    start: str | None = None
    port: int | None = None
    readiness_path: str = "/"
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "bootstrap": list(self.bootstrap),
            "build": list(self.build), "test": list(self.test), "start": self.start,
            "port": self.port, "readinessPath": self.readiness_path, "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Recipe | None:
        """Tolerant loader mirroring grok's ``normalizeVerifyRecipe``; accepts
        both this module's field names and grok's camelCase aliases."""
        if not isinstance(raw, dict):
            return None
        name = raw.get("name") or raw.get("appLabel")
        if not isinstance(name, str) or not name.strip():
            return None
        kind = raw.get("kind") or raw.get("appKind")
        if not isinstance(kind, str) or not kind.strip():
            kind = "unknown"

        start = raw.get("start") or raw.get("startCommand")
        port_raw = raw.get("port") or raw.get("startPort")
        if isinstance(port_raw, str) and port_raw.strip().isdigit():
            port_raw = int(port_raw.strip())
        readiness = raw.get("readinessPath") or raw.get("readiness_path") or "/"

        return cls(
            name=name.strip(), kind=kind.strip(),
            start=start.strip() if isinstance(start, str) and start.strip() else None,
            port=port_raw if isinstance(port_raw, int) and 0 < port_raw < 65536 else None,
            readiness_path=readiness if isinstance(readiness, str) and readiness.startswith("/") else "/",
            bootstrap=_as_strings(raw.get("bootstrap") or raw.get("installCommands")),
            build=_as_strings(raw.get("build") or raw.get("buildCommands")),
            test=_as_strings(raw.get("test") or raw.get("testCommands")),
            evidence=_as_strings(raw.get("evidence")),
        )


def _read_text(root: Path, name: str) -> str | None:
    try:
        return (root / name).read_text(encoding="utf-8-sig")
    except OSError:
        return None


def _read_package_json(root: Path) -> dict[str, Any] | None:
    try:
        parsed = json.loads(_read_text(root, "package.json") or "")
    except ValueError:  # includes JSONDecodeError; also the missing-file "" case
        return None
    return parsed if isinstance(parsed, dict) else None


# Ordered: the first lockfile present wins (grok's detectPackageManager).
_LOCKFILE_MANAGERS = (
    ("pnpm-lock.yaml", "pnpm"), ("bun.lock", "bun"), ("bun.lockb", "bun"), ("yarn.lock", "yarn"),
    ("package-lock.json", "npm"), ("uv.lock", "uv"), ("poetry.lock", "poetry"),
    ("Pipfile.lock", "pipenv"),
)


def _first_existing(root: Path, names: tuple[str, ...]) -> str | None:
    """First of ``names`` present under ``root``, else ``None``."""
    return next((n for n in names if (root / n).exists()), None)


def detect_package_manager(root: Path) -> str | None:
    """Lockfile-based package-manager detection (grok's detectPackageManager)."""
    return next((m for f, m in _LOCKFILE_MANAGERS if (root / f).exists()), None)


def _infer_port_from_command(command: str | None) -> int | None:
    """Port inference from a start command (grok's inferPortFromCommand)."""
    if not command:
        return None
    match = re.search(r"(?:--port|-p)\s+(\d{2,5})", command) or re.search(r"\bPORT=(\d{2,5})\b", command)
    return int(match.group(1)) if match else None


# Long-form spellings only, in both the space and the equals form. These are the tokens the
# frameworks we force flags for actually parse: Astro reads its argv with
# ``yargs(argv, {boolean: ["global"], alias: {g: "global"}})`` and declares no ``-H``/``-p``
# alias (astro/dist/cli/index.js), so a short form a project wrote is invisible to the app.
# Treating a short form as "already set" suppresses the only flag Astro parses, and the bind
# the flag exists to force never happens. Anchored so a flag is not matched inside a longer
# token (``--hosts``, a path containing ``--port``).
#
# The port pattern requires the digits on purpose: the append decision and the probe port
# come from this one pattern (``_pinned_port_value``), so we can never skip the append for a
# port we cannot predict and then probe a port the app never binds.
_PORT_FLAG_RE = re.compile(r"(?:^|\s)--port(?:\s+|=)(\d{2,5})(?!\d)")
_FLAG_PRESENT_RE = {
    "host": re.compile(r"(?:^|\s)--host(?:\s|=|$)"),
    "port": _PORT_FLAG_RE,
}


def _pinned_port_value(text: str | None) -> int | None:
    """Numeric value of a long-form ``--port`` already present in ``text``, else ``None``.

    The same pattern :func:`_ensure_flags` consults before appending ``--port``, so a pinned
    port both suppresses the append and becomes the recipe's probe port (``--port=4111``
    must probe 4111). A short ``-p 4111`` does not count: Astro declares no ``-p`` alias, so
    the value only feeds the flag we append ourselves.
    """
    if not text:
        return None
    match = _PORT_FLAG_RE.search(text)
    return int(match.group(1)) if match else None


def _ensure_flags(command: str, script_body: str = "", **flags: str | int | None) -> str:
    """Return ``command`` with ``--name value`` appended for every recognised flag that is
    not already present.

    A flag is skipped when its value is ``None``/empty, when its name is not in
    :data:`_FLAG_PRESENT_RE`, or when its **long form** already appears in ``command``
    **or** in ``script_body``. ``script_body`` is the package script the command invokes
    (the body of ``package.json``'s ``dev`` script, say): the runner command alone hides it
    (``bun run dev``), so a project that pins the flag itself would otherwise get it
    appended a second time. A duplicated flag is not a no-op -- ``yargs-parser`` returns an
    *array* for it and consumers accepting only ``string | boolean`` silently drop the
    setting (Astro then falls back to its ``[::1]`` bind, the very default this flag exists
    to override).

    Only long-form spellings count as present (:data:`_FLAG_PRESENT_RE`), because that is
    all Astro parses; see the note on the pattern table. Short forms never suppress the
    append.

    Recognised names: ``host``, ``port``. Useful for binding Astro ``dev`` explicitly to
    IPv4 127.0.0.1 so ``hermes verify``'s IPv4 readiness probe reaches it.
    """
    tokens = [command]
    for name, value in flags.items():
        if value is None or value == "" or name not in _FLAG_PRESENT_RE:
            continue
        present = _FLAG_PRESENT_RE[name]
        if present.search(command) or present.search(script_body):
            continue
        tokens.append(f"--{name} {value}")
    return " ".join(tokens)


def _dedupe(values: list[str | None]) -> list[str]:
    """Strip, drop empties, keep first occurrence order."""
    return list(dict.fromkeys(v.strip() for v in values if v and v.strip()))


_SCRIPT_RUNNERS = {"pnpm": "pnpm {}", "bun": "bun run {}", "yarn": "yarn {}"}
_NODE_INSTALL = {"pnpm": "pnpm install", "bun": "bun install", "yarn": "yarn install"}
# Ordered: the first dependency present decides the framework (kind, label, default port).
_NODE_FRAMEWORKS = (
    (("next",), "nextjs", "Next.js", 3000), (("@sveltejs/kit",), "sveltekit", "SvelteKit", 5173),
    (("astro",), "astro", "Astro", 4321),
    (("@remix-run/dev", "@remix-run/react"), "remix", "Remix", 3000),
    (("react-scripts",), "cra", "Create React App", 3000), (("vite",), "vite", "Vite", 5173),
)


def _script_runner(package_manager: str | None, entry: str) -> str:
    return _SCRIPT_RUNNERS.get(package_manager or "", "npm run {}").format(entry)


def _detect_node_recipe(root: Path, pkg: dict[str, Any]) -> Recipe:
    raw_scripts = pkg.get("scripts")
    scripts: dict[str, str] = raw_scripts if isinstance(raw_scripts, dict) else {}
    deps: dict[str, Any] = {}
    for section in (pkg.get("dependencies"), pkg.get("devDependencies")):
        if isinstance(section, dict):
            deps.update(section)

    package_manager = detect_package_manager(root)
    kind, label, default_port = next(
        ((k, lbl, p) for names, k, lbl, p in _NODE_FRAMEWORKS if any(n in deps for n in names)),
        ("node", "Node.js app", None),
    )

    def runners(*names: str) -> list[str]:
        return _dedupe([_script_runner(package_manager, s) for s in names if scripts.get(s)])

    start_script = next((s for s in ("dev", "start") if scripts.get(s)), None)
    start_body = scripts.get(start_script, "") if start_script else ""
    # The probe port must be the port the app will bind. A long-form ``--port`` the project
    # already pins (space or equals spelling) wins over the looser grok-style inference:
    # both ``--port 4111`` and ``--port=4111`` are understood by Astro, while ``-p 4111`` is
    # not -- a short form only informs the value we append ourselves, never the probe port
    # on its own. This is the same pattern ``_ensure_flags`` uses to decide the append, so
    # "skip the flag" and "probe its value" cannot disagree. (The resolved command is built
    # by ``_script_runner`` and never carries flags, so the script body is the only place a
    # pin can come from here.)
    resolved_port = (
        _pinned_port_value(start_body) or _infer_port_from_command(start_body) or default_port
    )
    start_cmd = runners(start_script)[0] if start_script else None
    # Astro binds [::1]:4321 by default; the verify readiness probe hits IPv4 127.0.0.1
    # and would connection-refuse. Force the dev server to listen on 127.0.0.1 and the
    # recipe's resolved port so the probe always reaches it.
    if start_cmd and kind == "astro":
        # The pinned script body matters here too: ``bun run dev`` alone cannot reveal a
        # flag the project already sets inside its own ``dev`` script.
        start_cmd = _ensure_flags(start_cmd, start_body, host="127.0.0.1", port=resolved_port)
    return Recipe(
        name=label, kind=kind, start=start_cmd,
        port=resolved_port if start_script else None,
        bootstrap=[_NODE_INSTALL.get(package_manager or "", "npm install")],
        build=runners("build", "typecheck"), test=runners("test", "check", "lint"),
        evidence=_dedupe([
            "Detected package.json",
            f"Package manager: {package_manager}" if package_manager else None,
            f"Scripts: {', '.join(scripts) or '(none)'}",
        ]),
    )


_PYTHON_INSTALL = {"uv": "uv sync", "poetry": "poetry install", "pipenv": "pipenv install"}


def _detect_python_recipe(root: Path) -> Recipe | None:
    pyproject = _read_text(root, "pyproject.toml")
    requirements = _read_text(root, "requirements.txt")
    manage_py = (root / "manage.py").exists()
    if not (pyproject or requirements or manage_py or (root / "setup.py").exists()):
        return None

    lower = f"{pyproject or ''}\n{requirements or ''}".lower()
    install = _PYTHON_INSTALL.get(detect_package_manager(root) or "") or (
        "pip install -e ." if pyproject and not requirements else "pip install -r requirements.txt"
    )
    pytest_or_empty = ["pytest"] if (root / "tests").exists() else []

    # Precedence: Django, then FastAPI/uvicorn, then Flask, then generic.
    if manage_py or "django" in lower:
        return Recipe(
            name="Django app", kind="django", bootstrap=[install], test=["python manage.py test"],
            start="python manage.py runserver 0.0.0.0:8000", port=8000,
            evidence=["Detected manage.py" if manage_py else "Detected Django dependency"]
            + (["Detected pyproject.toml"] if pyproject else []),
        )
    if "fastapi" in lower or "uvicorn" in lower:
        app_module = (_first_existing(root, ("main.py", "app.py")) or "main.py").removesuffix(".py") + ":app"
        return Recipe(
            name="FastAPI app", kind="fastapi", bootstrap=[install], test=pytest_or_empty,
            start=f"uvicorn {app_module} --host 0.0.0.0 --port 8000", port=8000,
            evidence=["Detected Python project", "Detected FastAPI/Uvicorn dependency"],
        )
    if "flask" in lower:
        app_module = _first_existing(root, ("app.py", "main.py")) or "app.py"
        return Recipe(
            name="Flask app", kind="flask", bootstrap=[install], test=pytest_or_empty,
            start=f"flask --app {app_module} run --host 0.0.0.0 --port 5000", port=5000,
            evidence=["Detected Python project", "Detected Flask dependency"],
        )
    return Recipe(
        name="Python project", kind="python", bootstrap=[install],
        test=pytest_or_empty or ["python -m unittest discover"],
        evidence=["Detected Python project"],
    )


# Single-manifest toolchains: manifest -> (label, kind, build, test, start command, entry file).
_SIMPLE_TOOLCHAINS = (
    ("go.mod", "Go project", "go", "go build ./...", "go test ./...", "go run .", Path("main.go")),
    ("Cargo.toml", "Rust project", "rust", "cargo build", "cargo test", "cargo run", Path("src") / "main.rs"),
)


def _detect_simple_recipe(root: Path) -> Recipe | None:
    for manifest, label, kind, build, test, start, entry in _SIMPLE_TOOLCHAINS:
        if (root / manifest).exists():
            return Recipe(
                name=label, kind=kind, build=[build], test=[test],
                start=start if (root / entry).exists() else None, evidence=[f"Detected {manifest}"],
            )
    return None


def _detect_java_recipe(root: Path) -> Recipe | None:
    if (root / "pom.xml").exists():
        return Recipe(name="Maven project", kind="maven", build=["mvn package"], test=["mvn test"], evidence=["Detected pom.xml"])
    if _first_existing(root, ("build.gradle", "build.gradle.kts")):
        gradle = "./gradlew" if (root / "gradlew").exists() else "gradle"
        return Recipe(
            name="Gradle project", kind="gradle", build=[f"{gradle} build"], test=[f"{gradle} test"],
            evidence=["Detected Gradle build file"],
        )
    return None


_MAKE_TARGET_RE = re.compile(r"^([A-Za-z0-9_.-]+):(?:\s|$)")
# Makefile phase -> candidate targets, first present wins.
_MAKE_PHASE_TARGETS = {
    "bootstrap": ("install", "setup", "bootstrap"),
    "build": ("build", "compile"),
    "test": ("test", "check"),
    "start": ("run", "start", "serve", "dev"),
}


def _detect_make_recipe(root: Path) -> Recipe | None:
    makefile = _read_text(root, "Makefile")
    if makefile is None:
        return None
    targets = [m.group(1) for m in map(_MAKE_TARGET_RE.match, makefile.splitlines()) if m]
    picked = {
        phase: [f"make {n}" for n in names if n in targets][:1]
        for phase, names in _MAKE_PHASE_TARGETS.items()
    }
    return Recipe(
        name="Makefile-driven project", kind="make", start=(picked["start"] or [None])[0],
        bootstrap=picked["bootstrap"], build=picked["build"], test=picked["test"],
        evidence=["Detected Makefile", f"Targets: {', '.join(targets) or '(none)'}"],
    )


_COMPOSE_FILES = ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml")


def _detect_compose_recipe(root: Path) -> Recipe | None:
    compose_file = _first_existing(root, _COMPOSE_FILES)
    return None if compose_file is None else Recipe(
        name="docker-compose project", kind="compose", build=["docker compose build"],
        start="docker compose up", evidence=[f"Detected {compose_file}"],
    )


def detect_recipe(root: Path) -> Recipe | None:
    """Detect a recipe for ``root`` (grok-cli's ``inferFallbackRecipe`` order): package.json
    wins, then Python, Go, Rust, Java, then Makefile / docker-compose; ``None`` if nothing matches."""
    root = Path(root)
    pkg = _read_package_json(root)
    if pkg is not None:
        return _detect_node_recipe(root, pkg)
    return (
        _detect_python_recipe(root)
        or _detect_simple_recipe(root)
        or _detect_java_recipe(root)
        or _detect_make_recipe(root)
        or _detect_compose_recipe(root)
    )
