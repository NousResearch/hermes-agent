#!/usr/bin/env bash
# Blog pipeline audit wrapper.
#
# Resolves the audit script relative to the actual KenseiAgent checkout so the
# correct (worktree) copy runs, then forwards BLOG_AUDIT_ENGINE_ROOT /
# BLOG_AUDIT_BLOG_ROOT env overrides to the audit (production defaults are
# preserved when the overrides are absent).
#
# NOTE: this wrapper is deployed to ~/.hermes/scripts/ (separate from the repo),
# so it must NOT derive REPO_ROOT from SCRIPT_DIR/.. — that would resolve to
# ~/.hermes/ and the content_engine path would not exist. Resolve the repo
# root from the real checkout instead, matching the other blog-* wrappers.
set -euo pipefail

REPO="${KENSEI_REPO:-/home/kensei/repos/KenseiAgent}"
AUDIT="${REPO}/content_engine/tools/blog_pipeline_audit.py"

if [[ ! -f "${AUDIT}" ]]; then
  echo "blog-pipeline-audit: audit script not found at ${AUDIT}" >&2
  exit 2
fi

cd "${REPO}"
PYTHONPATH="${REPO}/content_engine" python3 "${AUDIT}"
