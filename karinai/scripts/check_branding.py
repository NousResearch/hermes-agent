#!/usr/bin/env python3
"""Compatibility wrapper for the KarinAI prompt/branding audit."""

from __future__ import annotations

try:
    from .audit_prompts import main
except ImportError:  # pragma: no cover - script execution from this directory
    from audit_prompts import main


if __name__ == "__main__":
    raise SystemExit(main())
