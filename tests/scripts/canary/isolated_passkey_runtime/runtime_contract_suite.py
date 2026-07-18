from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path

import pytest


ISOLATED_RUNTIME_ENV = "MUNCHO_OWNER_GATE_ISOLATED_TEST_RUNTIME"
PRODUCTION_RUNTIME_PROJECTS = frozenset({
    "cbor2",
    "cffi",
    "cryptography",
    "packaging",
    "pyasn1",
    "pyasn1-modules",
    "pycparser",
    "pyopenssl",
    "typing-extensions",
    "webauthn",
})


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _production_versions() -> dict[str, str]:
    lock_path = (
        _repository_root()
        / "ops"
        / "muncho"
        / "owner-gate"
        / "runtime-wheel-lock.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    return {
        item["project"].lower(): item["version"]
        for item in lock["wheels"]
        if item["project"].lower() in PRODUCTION_RUNTIME_PROJECTS
    }


def test_isolated_verifier_versions_match_signed_production_runtime() -> None:
    assert os.environ.get(ISOLATED_RUNTIME_ENV) == "1"
    assert sys.version_info[:2] == (3, 11)
    expected = _production_versions()
    assert set(expected) == PRODUCTION_RUNTIME_PROJECTS
    installed = {
        project: importlib.metadata.version(project)
        for project in sorted(PRODUCTION_RUNTIME_PROJECTS)
    }
    assert installed == expected


def test_isolated_verifier_runtime_excludes_hermes_dingtalk_dependency() -> None:
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.version("alibabacloud-tea-openapi")

    from scripts.canary import passkey_v2_webauthn

    verifier, invalid_response = passkey_v2_webauthn._load_selected_verifier()
    assert callable(verifier)
    assert issubclass(invalid_response, Exception)
