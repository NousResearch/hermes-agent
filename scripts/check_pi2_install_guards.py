#!/usr/bin/env python3
"""Guard Raspberry Pi 2 install paths against heavy or ARMv7-hostile deps.

This check is intentionally narrow: explanatory documentation may mention
packages such as uvloop or chromadb, but Pi2 install metadata, lazy deps,
setup scripts, and runtime dashboard startup must keep the default path light.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import Iterable

BANNED_DEFAULT_DEPS = (
    "uvicorn[standard]",
    "uvloop",
    "torch",
    "sentence-transformers",
    "chromadb",
)
HEAVY_RAG_DEPS = ("torch", "sentence-transformers", "chromadb")


class FailureCollector:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def add(self, message: str) -> None:
        self.failures.append(message)

    def exit_code(self) -> int:
        return 1 if self.failures else 0

    def report(self) -> None:
        if not self.failures:
            print("Pi2 install guard checks passed")
            return
        print("Pi2 install guard checks failed:")
        for failure in self.failures:
            print(f"- {failure}")


def normalize_dep(dep: str) -> str:
    dep = dep.strip().lower()
    # Remove environment markers and common version specifiers for prefix checks.
    dep = dep.split(";", 1)[0].strip()
    return re.split(r"\s*(?:==|>=|<=|~=|!=|>|<)", dep, maxsplit=1)[0].strip()


def iter_pyproject_deps(pyproject: Path) -> Iterable[tuple[str, str]]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    for dep in project.get("dependencies", []) or []:
        yield "project.dependencies", dep
    optional = project.get("optional-dependencies", {}) or {}
    for extra, deps in optional.items():
        for dep in deps or []:
            yield f"project.optional-dependencies.{extra}", dep


def check_pyproject(repo: Path, failures: FailureCollector) -> None:
    pyproject = repo / "pyproject.toml"
    if not pyproject.exists():
        failures.add("pyproject.toml is missing")
        return

    for section, dep in iter_pyproject_deps(pyproject):
        package = normalize_dep(dep)
        if package in BANNED_DEFAULT_DEPS:
            failures.add(
                f"pyproject.toml {section} contains Pi2-hostile dependency {dep!r}; "
                "use lightweight/remote alternatives by default"
            )


def iter_python_string_literals(path: Path) -> Iterable[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def check_lazy_deps(repo: Path, failures: FailureCollector) -> None:
    lazy_deps = repo / "tools" / "lazy_deps.py"
    if not lazy_deps.exists():
        failures.add("tools/lazy_deps.py is missing")
        return

    for literal in iter_python_string_literals(lazy_deps):
        package = normalize_dep(literal)
        if package in BANNED_DEFAULT_DEPS:
            failures.add(
                f"tools/lazy_deps.py contains Pi2-hostile lazy dependency {literal!r}; "
                "dashboard lazy deps must use plain uvicorn and avoid local ML/RAG stacks"
            )


def joined_shell_commands(path: Path) -> list[str]:
    commands: list[str] = []
    current = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if current:
            current += " " + stripped.rstrip("\\").strip()
        else:
            current = stripped.rstrip("\\").strip()
        if stripped.endswith("\\"):
            continue
        commands.append(current)
        current = ""
    if current:
        commands.append(current)
    return commands


def check_setup_pi2(repo: Path, failures: FailureCollector) -> None:
    setup = repo / "setup-pi2.sh"
    if not setup.exists():
        failures.add("setup-pi2.sh is missing")
        return

    for command in joined_shell_commands(setup):
        if not re.search(r"(^|\s)(python\s+-m\s+)?pip\s+install(\s|$)", command):
            continue
        if command.lstrip().startswith("echo "):
            continue
        lowered = command.lower()
        for dep in HEAVY_RAG_DEPS:
            if re.search(rf"(^|[\s'\"]){re.escape(dep)}([\s'\"]|$)", lowered):
                failures.add(
                    f"setup-pi2.sh default pip install command contains {dep}; "
                    "Pi2 defaults must not install local heavy RAG/embedding stacks"
                )
        if "uvicorn[standard]" in lowered or "uvloop" in lowered:
            failures.add(
                "setup-pi2.sh default pip install command contains uvicorn[standard]/uvloop; "
                "use plain uvicorn on Pi2"
            )


def check_setup_pi2_minimal(repo: Path, failures: FailureCollector) -> None:
    setup = repo / "setup-pi2-minimal.sh"
    if not setup.exists():
        failures.add("setup-pi2-minimal.sh is missing")
        return

    text = setup.read_text(encoding="utf-8")
    if "sqlite-vec" in text:
        required_markers = ("HERMES_PI2_TRY_SQLITE_VEC", "armv7l", "armv6l")
        missing = [marker for marker in required_markers if marker not in text]
        if missing:
            failures.add(
                "setup-pi2-minimal.sh installs sqlite-vec without the Pi2 opt-in/ARM guard; "
                f"missing {', '.join(missing)}"
            )


def check_iot_optional_deps(repo: Path, failures: FailureCollector) -> None:
    """Keep IoT extras useful but out of the Pi2/core dependency path."""
    pyproject = repo / "pyproject.toml"
    if not pyproject.exists():
        return
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    deps = project.get("dependencies", []) or []
    optional = project.get("optional-dependencies", {}) or {}
    core_packages = {normalize_dep(dep) for dep in deps}
    if "paho-mqtt" in core_packages:
        failures.add("paho-mqtt must stay out of project.dependencies; MQTT should remain optional/lazy on Pi2")
    mqtt_extra = {normalize_dep(dep) for dep in optional.get("mqtt", []) or []}
    if "paho-mqtt" not in mqtt_extra:
        failures.add("pyproject.toml should expose optional-dependencies.mqtt with paho-mqtt")

    lazy_deps = repo / "tools" / "lazy_deps.py"
    if lazy_deps.exists() and "tool.mqtt" not in lazy_deps.read_text(encoding="utf-8"):
        failures.add("tools/lazy_deps.py should lazy-install paho-mqtt through the tool.mqtt feature")


def check_pi2_context_templates(repo: Path, failures: FailureCollector) -> None:
    """Guard the low-resource context-floor optimization from regressions."""
    expected = {
        "config.pi2-core.yaml": 2048,
        "config.pi2-native.yaml": 8192,
        "config.pi2-rag.yaml": 8192,
    }
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - CI/dev envs have PyYAML
        failures.add(f"PyYAML is required to validate Pi2 config templates: {exc}")
        return

    for filename, expected_floor in expected.items():
        path = repo / "templates" / filename
        if not path.exists():
            failures.add(f"{path.relative_to(repo)} is missing")
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        agent_cfg = data.get("agent") or {}
        actual = agent_cfg.get("minimum_tool_context_length")
        if actual != expected_floor:
            failures.add(
                f"{path.relative_to(repo)} agent.minimum_tool_context_length should be "
                f"{expected_floor}, got {actual!r}"
            )


def check_web_server(repo: Path, failures: FailureCollector) -> None:
    web_server = repo / "hermes_cli" / "web_server.py"
    if not web_server.exists():
        failures.add("hermes_cli/web_server.py is missing")
        return

    text = web_server.read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", text)
    if "uvicorn.Config" in text and 'loop="asyncio"' not in compact and "loop='asyncio'" not in compact:
        failures.add(
            'hermes_cli/web_server.py must pass loop="asyncio" to uvicorn.Config so Pi2/ARMv7 never auto-selects uvloop'
        )

    for literal in iter_python_string_literals(web_server):
        package = normalize_dep(literal)
        if package in ("uvicorn[standard]", "uvloop"):
            failures.add(
                f"hermes_cli/web_server.py user-facing install/runtime string contains {literal!r}; "
                "Pi2 guidance should use plain uvicorn"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Repository root to check")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    failures = FailureCollector()
    check_pyproject(repo, failures)
    check_lazy_deps(repo, failures)
    check_setup_pi2(repo, failures)
    check_setup_pi2_minimal(repo, failures)
    check_iot_optional_deps(repo, failures)
    check_pi2_context_templates(repo, failures)
    check_web_server(repo, failures)
    failures.report()
    return failures.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
