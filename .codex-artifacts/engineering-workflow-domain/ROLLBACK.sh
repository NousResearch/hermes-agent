#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"
files=(
  engineering/__init__.py
  engineering/domain/__init__.py
  engineering/domain/workflow.py
  tests/engineering/domain/test_workflow.py
)

for relative_path in "${files[@]}"; do
  rm -f -- "$root/$relative_path"
done
rmdir -- "$root/engineering/domain" "$root/engineering" 2>/dev/null || true
rmdir -- "$root/tests/engineering/domain" "$root/tests/engineering" 2>/dev/null || true
printf 'ROLLBACK_RESULT=removed_domain_layer:%s\n' "$root"