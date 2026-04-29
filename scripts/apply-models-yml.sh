#!/usr/bin/env bash
set -euo pipefail
MODELS_YML="${1:-config/models.yml}"
[ ! -f "$MODELS_YML" ] && { echo "Error: $MODELS_YML missing. Copy from config/models.yml.example."; exit 1; }
PRIMARY_PROVIDER=$(yq -r '.roles.primary_reasoning.provider' "$MODELS_YML")
PRIMARY_MODEL=$(yq -r '.roles.primary_reasoning.model' "$MODELS_YML")
FAST_MODEL=$(yq -r '.roles.fast_iteration.model' "$MODELS_YML")
HERMES_CFG="${HERMES_HOME:-$HOME/.hermes}/config.yaml"
[ -f "$HERMES_CFG" ] && yq -i ".model.default = \"$PRIMARY_MODEL\" | .model.provider = \"$PRIMARY_PROVIDER\"" "$HERMES_CFG"
USER_AGENTS_DIR="${HOME}/.claude/agents"
mkdir -p "$USER_AGENTS_DIR"
for agent_file in .claude/agents/*.md; do
  base=$(basename "$agent_file")
  ROLE=$(grep -E '^role:' "$agent_file" | sed 's/role: //' || echo "")
  case "$ROLE" in
    primary_reasoning) MODEL="$PRIMARY_MODEL" ;;
    fast_iteration) MODEL="$FAST_MODEL" ;;
    adversarial_review) MODEL=$(yq -r '.roles.adversarial_review.model' "$MODELS_YML") ;;
    legal_tech_review) MODEL=$(yq -r '.roles.legal_tech_review.model' "$MODELS_YML") ;;
    *) MODEL="$PRIMARY_MODEL" ;;
  esac
  awk -v model="$MODEL" 'BEGIN{inserted=0} /^role:/ {print; print "model: " model; inserted=1; next} {print}' "$agent_file" > "$USER_AGENTS_DIR/$base"
done
echo "Applied. Per-machine model: lines written to $USER_AGENTS_DIR/."
