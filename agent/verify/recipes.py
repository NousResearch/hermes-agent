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
import tomllib
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
    return Recipe(
        name=label, kind=kind, start=runners(start_script)[0] if start_script else None,
        port=(_infer_port_from_command(scripts[start_script]) or default_port) if start_script else None,
        bootstrap=[_NODE_INSTALL.get(package_manager or "", "npm install")],
        build=runners("build", "typecheck"), test=runners("test", "check", "lint"),
        evidence=_dedupe([
            "Detected package.json",
            f"Package manager: {package_manager}" if package_manager else None,
            f"Scripts: {', '.join(scripts) or '(none)'}",
        ]),
    )


_PYTHON_INSTALL = {"uv": "uv sync", "poetry": "poetry install", "pipenv": "pipenv install"}

# Directory names that hold vendored code, virtualenvs or build output: files found
# there belong to dependencies, not to the project, so they are never evidence.
_VENDORED_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", ".tox", ".nox", ".venv", "venv", "env",
        "node_modules", "site-packages", "build", "dist", "__pycache__",
    }
)
_TEST_FILE_PATTERNS = ("test_*.py", "*_test.py", "conftest.py")
# An ASGI application object, not merely a dependency named fastapi.
_ASGI_APP_RE = re.compile(r"^\s*(?:app|application)\s*=\s*(?:FastAPI|Starlette)\s*\(", re.MULTILINE)


def _iter_project_files(root: Path, patterns: tuple[str, ...], max_depth: int = 2) -> list[Path]:
    """Files matching ``patterns`` under ``root``, bounded in depth and count.

    Vendored and generated trees are skipped so that a dependency of a dependency
    cannot be mistaken for evidence about this project.
    """
    found: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and len(found) < 200:
        directory, depth = stack.pop()
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if depth < max_depth and entry.name not in _VENDORED_DIRS:
                    stack.append((entry, depth + 1))
                continue
            if any(entry.match(pattern) for pattern in patterns):
                found.append(entry)
    return found


def _runtime_dependency_text(pyproject: str | None, requirements: str | None) -> str:
    """Dependency evidence about the app itself.

    Only ``[project].dependencies`` and ``requirements.txt`` describe what the
    application runs on. Optional dependencies, ``[dependency-groups]`` and
    Poetry's dev groups describe extras, tooling and examples, so a FastAPI pin
    there must not turn a library into a "FastAPI app".
    """
    evidence: list[str] = [requirements or ""]
    if pyproject:
        try:
            parsed = tomllib.loads(pyproject)
        except (tomllib.TOMLDecodeError, ValueError):
            evidence.append(pyproject)  # unparseable: fall back to raw text
        else:
            project = parsed.get("project") if isinstance(parsed, dict) else None
            dependencies = project.get("dependencies") if isinstance(project, dict) else None
            evidence.extend(_as_strings(dependencies))
            tool = parsed.get("tool") if isinstance(parsed, dict) else None
            poetry = tool.get("poetry") if isinstance(tool, dict) else None
            poetry_deps = poetry.get("dependencies") if isinstance(poetry, dict) else None
            if isinstance(poetry_deps, dict):
                evidence.extend(str(name) for name in poetry_deps)
    return "\n".join(evidence).lower()


def _find_asgi_module(root: Path) -> str | None:
    """Dotted module path of a file that really defines an ASGI ``app``, else ``None``."""
    for path in _iter_project_files(root, ("*.py",), max_depth=1):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _ASGI_APP_RE.search(text):
            return ".".join(path.relative_to(root).with_suffix("").parts)
    entry = _first_existing(root, ("main.py", "app.py"))
    return entry.removesuffix(".py") if entry else None


def _has_tests(root: Path, pyproject: str | None) -> bool:
    """Real test files or an explicit pytest configuration, not merely a ``tests/`` dir."""
    if pyproject and "[tool.pytest.ini_options]" in pyproject:
        return True
    if any("pytest" in (_read_text(root, name) or "").lower() for name in ("pytest.ini", "setup.cfg", "tox.ini")):
        return True
    return bool(_iter_project_files(root, _TEST_FILE_PATTERNS, max_depth=3))


def _detect_python_recipe(root: Path) -> Recipe | None:
    pyproject = _read_text(root, "pyproject.toml")
    requirements = _read_text(root, "requirements.txt")
    manage_py = (root / "manage.py").exists()
    if not (pyproject or requirements or manage_py or (root / "setup.py").exists()):
        return None

    dependencies = _runtime_dependency_text(pyproject, requirements)
    install = _PYTHON_INSTALL.get(detect_package_manager(root) or "") or (
        "pip install -e ." if pyproject and not requirements else "pip install -r requirements.txt"
    )
    test_commands = ["pytest"] if _has_tests(root, pyproject) else []
    no_entrypoint: list[str] = []

    # Precedence: Django, then FastAPI/uvicorn, then Flask, then generic.
    if manage_py or "django" in dependencies:
        return Recipe(
            name="Django app", kind="django", bootstrap=[install], test=["python manage.py test"],
            start="python manage.py runserver 0.0.0.0:8000", port=8000,
            evidence=["Detected manage.py" if manage_py else "Detected Django dependency"]
            + (["Detected pyproject.toml"] if pyproject else []),
        )
    if "fastapi" in dependencies or "uvicorn" in dependencies:
        app_module = _find_asgi_module(root)
        if app_module:
            return Recipe(
                name="FastAPI app", kind="fastapi", bootstrap=[install], test=test_commands,
                start=f"uvicorn {app_module}:app --host 0.0.0.0 --port 8000", port=8000,
                evidence=["Detected Python project", "Detected FastAPI/Uvicorn dependency"]
                + (["Detected ASGI application object"] if "." in app_module else []),
            )
        no_entrypoint = ["Detected FastAPI/Uvicorn dependency but no module defining `app`"]
    if "flask" in dependencies:
        app_module = _first_existing(root, ("app.py", "main.py"))
        if app_module:
            return Recipe(
                name="Flask app", kind="flask", bootstrap=[install], test=test_commands,
                start=f"flask --app {app_module} run --host 0.0.0.0 --port 5000", port=5000,
                evidence=["Detected Python project", "Detected Flask dependency"],
            )
        no_entrypoint = no_entrypoint or ["Detected Flask dependency but no app.py or main.py"]
    return Recipe(
        name="Python project", kind="python", bootstrap=[install], test=test_commands,
        evidence=["Detected Python project"] + no_entrypoint,
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
