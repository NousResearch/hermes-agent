from __future__ import annotations

import tomllib
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _optional_dependencies(extra: str) -> list[str]:
    data = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text())
    return data["project"]["optional-dependencies"][extra]


def test_messaging_extra_installs_feishu_runtime_sdk() -> None:
    deps = _optional_dependencies("messaging")

    assert any(dep.startswith("lark-oapi") for dep in deps)


def test_feishu_extra_installs_feishu_runtime_sdk() -> None:
    deps = _optional_dependencies("feishu")

    assert any(dep.startswith("lark-oapi") for dep in deps)
