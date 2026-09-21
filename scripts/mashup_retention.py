#!/usr/bin/env python3
"""KenseiAgent adapter for retention — thin wrapper over the package.

All crash-recoverable, idempotent retention logic lives in
`research-mashup-pipeline` (`mashup.retention`). This wrapper:
- resolves the package (env MASHUP_PKG, default ~/research-mashup-pipeline);
- passes through the CLI + exit codes unchanged.

The live Kensei blog dir is configured via MASHUP_BLOG_DIR (default:
$HERMES_HOME/content_engine/blog_topics).
"""
import os
import sys

_PKG = os.environ.get("MASHUP_PKG", "/home/kensei/archive/research-mashup-pipeline/src")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from mashup.retention import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
