#!/usr/bin/env bash
# Reproducible, guarded Terraform via Docker.
#   scripts/tf.sh fmt|validate|init|show|state list|output   -> passthrough (read-only / local)
#   scripts/tf.sh plan [extra args]   -> plan -out=tfplan, tfplan.json, plan_guard, sha256 recorded
#   scripts/tf.sh apply               -> applies ONLY ./tfplan and only if its sha matches the guarded one
# Refuses: destroy, apply without saved plan, -auto-approve on anything but the saved plan.
set -euo pipefail
TF_IMAGE="${NOVA_TF_IMAGE:-hashicorp/terraform:1.16.3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tf() {
  docker run --rm -i -e AWS_PROFILE="${AWS_PROFILE:-default}" -e TF_IN_AUTOMATION=1 \
    -v "$HOME/.aws":/root/.aws:ro -v "$PWD":/workspace -w /workspace "$TF_IMAGE" "$@"
}
cmd="${1:-}"; shift || true
case "$cmd" in
  destroy) echo "tf.sh: 'destroy' is not permitted (not a troubleshooting or rollback tool)." >&2; exit 3 ;;
  plan)
    tf plan -input=false -out=tfplan "$@"
    tf show -json tfplan > tfplan.json
    sha256sum tfplan | awk '{print $1}' > tfplan.sha256
    set +e; python3 "$HERE/plan_guard.py" tfplan.json ${NOVA_ALLOW:-}; rc=$?; set -e
    echo "plan sha256: $(cat tfplan.sha256)"
    case $rc in
      0) echo "guard: SAFE — still requires safety review before apply" ;;
      2) echo "guard: REVIEW — human approval required" ;;
      3) echo "guard: BLOCKED — do not apply; investigate" ;;
    esac
    [ $rc -eq 3 ] && touch tfplan.blocked || rm -f tfplan.blocked
    exit $rc ;;
  apply)
    [ $# -eq 0 ] || { echo "tf.sh apply takes no arguments; it applies ./tfplan only." >&2; exit 3; }
    [ -f tfplan ] && [ -f tfplan.sha256 ] || { echo "no saved plan; run tf.sh plan first" >&2; exit 3; }
    [ -f tfplan.blocked ] && { echo "saved plan was BLOCKED by plan_guard" >&2; exit 3; }
    [ "$(sha256sum tfplan | awk '{print $1}')" = "$(cat tfplan.sha256)" ] || { echo "tfplan changed since guard ran" >&2; exit 3; }
    [ "${NOVA_APPROVED_PLAN_SHA:-}" = "$(cat tfplan.sha256)" ] || {
      echo "set NOVA_APPROVED_PLAN_SHA=$(cat tfplan.sha256) after human/safety review to apply" >&2; exit 3; }
    tf apply -input=false tfplan ;;
  "") echo "usage: tf.sh <fmt|validate|init|plan|apply|show|state|output> ..." >&2; exit 1 ;;
  *) tf "$cmd" "$@" ;;
esac
