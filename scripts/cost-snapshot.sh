#!/usr/bin/env bash
set -euo pipefail
MODELS_YML="config/models.yml"
BASELINE="config/cost-baseline.yml"
PRICING="config/pricing.yml"
[ ! -f "$BASELINE" ] && { echo "Create $BASELINE from .example"; exit 1; }
[ ! -f "$PRICING" ] && { echo "Create $PRICING from .example"; exit 1; }
DATE=$(date +%Y-%m-%d)
{
  echo ""
  echo "## Cost snapshot - $DATE"
  for role in primary_reasoning fast_iteration adversarial_review legal_tech_review cheap_routine escalate_head; do
    P=$(yq -r ".roles.$role.provider" "$MODELS_YML")
    M=$(yq -r ".roles.$role.model" "$MODELS_YML")
    T=$(yq -r ".$role" "$BASELINE")
    PRICE=$(yq -r ".\"$P/$M\"" "$PRICING" 2>/dev/null || echo "unknown")
    if [ "$PRICE" = "unknown" ]; then
      echo "- **$role**: $P/$M - pricing unknown, update $PRICING"
    else
      COST=$(echo "scale=2; $T * $PRICE / 1000000" | bc)
      echo "- **$role**: $P/$M @ \$$PRICE/1M tok x $T tok/mo = \$${COST}/mo"
    fi
  done
} >> LEARNINGS.md
