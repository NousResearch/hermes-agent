#!/usr/bin/env python3
"""KenseiAgent adapter for proposal validation — thin wrapper over the package.

All structural validation logic lives in `research-mashup-pipeline`
(`mashup.proposal_validator`). This wrapper:
- resolves the package (env MASHUP_PKG, default ~/research-mashup-pipeline);
- passes through the CLI + exit codes unchanged.

Keeps live cron behaviour identical while removing duplicate logic.
"""
import os
import sys

_PKG = os.environ.get("MASHUP_PKG", "/home/kensei/research-mashup-pipeline/src")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from mashup.proposal_validator import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
