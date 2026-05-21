"""Tests for scripts/sync_all.py argument handling."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_sync_all_module():
    spec = importlib.util.spec_from_file_location("sync_all", REPO_ROOT / "scripts" / "sync_all.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_allow_preflight_blockers_flag_defaults_false():
    sync_all = load_sync_all_module()

    args = sync_all.parse_args(["--merge"])

    assert args.allow_preflight_blockers is False


def test_allow_preflight_blockers_flag_can_be_enabled():
    sync_all = load_sync_all_module()

    args = sync_all.parse_args(["--merge", "--allow-preflight-blockers"])

    assert args.allow_preflight_blockers is True
