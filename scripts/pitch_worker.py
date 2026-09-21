#!/usr/bin/env python3
"""KenseiAgent adapter for the pitch worker — thin wrapper over the package.

All pitch-lifecycle logic lives in `research-mashup-pipeline`
(`mashup.pitch_worker`). This wrapper resolves the package and passes
through the CLI (--once / default bounded loop) unchanged.
"""
import os
import sys

_PKG = os.environ.get("MASHUP_PKG", "/home/kensei/research-mashup-pipeline/src")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from mashup.pitch_worker import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
