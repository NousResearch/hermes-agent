"""Build-time paths, not a second runtime configuration or dependency resolver."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import json
import re
import tomllib


RESOURCE_ENV = {
    "skills": "HERMES_BUNDLED_SKILLS",
    "optional-skills": "HERMES_OPTIONAL_SKILLS",
    "plugins": "HERMES_BUNDLED_PLUGINS",
    "locales": "HERMES_BUNDLED_LOCALES",
    "optional-mcps": "HERMES_OPTIONAL_MCPS",
}


def dependency_site(environment: Path, python_version: str, target: str) -> Path:
    """Providers know the target pin; the builder's interpreter is irrelevant."""
    minor = python_version.split("+")[0].rsplit(".", 1)[0]
    return environment / ("Lib/site-packages" if target.startswith("win32") else
                          f"lib/python{minor}/site-packages")


def project_entries(project: Path) -> dict[str, str]:
    entries = tomllib.loads(project.read_text(encoding="utf-8-sig"))["project"]["scripts"]
    for name, entry in entries.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"invalid command name: {name}")
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", entry):
            raise ValueError(f"invalid script entry: {entry}")
    return entries


@dataclass(frozen=True)
class AgentInputs:
    project: Path
    code: Path
    repo: str
    placement: str
    target: str
    python: Path
    site_packages: Path
    environment: Path
    pm_runtime: Path
    bin_dir: str = "bin"
    tools: Path | None = None
    command_dir: Path | None = None
    resources: dict[str, Path] = field(default_factory=dict)
    frontends: dict[str, Path] = field(default_factory=dict)
    ref: str | None = None
    stamp: Path | None = None
    features: Path | None = None
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> AgentInputs:
        unknown = data.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown agent inputs: {', '.join(sorted(unknown))}")
        values = dict(data)
        for key in ("project", "code", "python", "site_packages", "environment", "tools",
                    "pm_runtime", "command_dir", "stamp", "features"):
            if values.get(key) is not None:
                values[key] = Path(values[key])
        for key in ("resources", "frontends"):
            values[key] = {name: Path(path) for name, path in values.get(key, {}).items()}
        return cls(**values)

    def validate(self, out: Path) -> None:
        if not re.fullmatch(r"(?:(?:linux|darwin|win32)-(?:x64|arm64)|linux-arm64-bionic)", self.target):
            raise ValueError(f"unsupported target: {self.target}")
        if self.placement not in ("contained", "fixed", "references"):
            raise ValueError(f"unknown placement: {self.placement}")
        if Path(self.repo).is_absolute() or ".." in Path(self.repo).parts:
            raise ValueError("repo must be an output-relative path")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.bin_dir):
            raise ValueError("bin_dir must be one output-relative directory")
        if self.resources.keys() - RESOURCE_ENV.keys():
            raise ValueError("unknown resource names")
        if self.frontends.keys() - {"tui", "web"}:
            raise ValueError("unknown frontend names")
        paths = [self.project, self.code, self.python, self.site_packages, self.environment,
                 self.pm_runtime, *self.resources.values(), *self.frontends.values()]
        paths += [p for p in (self.tools, self.command_dir, self.stamp, self.features) if p is not None]
        for path in paths:
            if not path.is_absolute():
                raise ValueError(f"input path must be absolute: {path}")
            if not path.exists():
                raise FileNotFoundError(f"required agent input missing: {path}")
        for path in (self.project, self.python):
            if not path.is_file():
                raise ValueError(f"expected a file: {path}")
        for path in (self.code, self.site_packages, self.environment, self.pm_runtime):
            if not path.is_dir():
                raise ValueError(f"expected a directory: {path}")
        marker = json.loads((self.pm_runtime / "pm-runtime.json").read_text(encoding="utf-8-sig"))
        pm_python = (self.pm_runtime / marker["python"]).resolve()
        pm_site = (self.pm_runtime / marker["sitePackages"]).resolve()
        if not pm_python.is_file() or not pm_site.is_dir():
            raise ValueError("prepared PM runtime is incomplete")
        if self.stamp is not None:
            stamp = json.loads(self.stamp.read_text(encoding="utf-8-sig"))
            if stamp.get("distribution") in ("nix", "docker") and stamp.get("pmRuntime") != str(self.pm_runtime):
                raise ValueError("install stamp must reference the supplied independent PM runtime")
        if self.placement == "contained":
            for path in (self.python, self.site_packages, self.environment, self.pm_runtime, self.tools):
                if path is not None and not path.resolve().is_relative_to(out):
                    raise ValueError(f"contained input must be prepared inside output: {path}")
        if self.placement == "references" and self.command_dir is None:
            raise ValueError("referenced placement requires command_dir")
        validate_frontends(self.frontends)


def validate_frontends(frontends: dict[str, Path]) -> None:
    for name, path in frontends.items():
        required = ("dist/entry.js", "package.json") if name == "tui" else ("index.html",)
        for entry in required:
            if not (path / entry).is_file():
                raise FileNotFoundError(path / entry)
