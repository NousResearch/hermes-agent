#!/usr/bin/env bash
# debate-orchestrator.sh — CLI entry point for adversarial debates
#
# Installed by the adversarial-debate optional skill.
# Usage:
#   debate-orchestrator.sh --topic "..." --context-file <path> --format <rca|feature>
#
# This script provides a bash-level interface to the adversarial debate
# protocol. It is intended for testing and CLI invocation. The primary
# path is to load the skill in Hermes and follow the SKILL.md procedure.

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +%s)
DEBATE_ID="debate-${TIMESTAMP}"
WORKSPACE="/tmp/hermes-debate-${DEBATE_ID}"
TRANSCRIPT_FILE="${WORKSPACE}/transcript.json"
SYNTHESIS_FILE="${WORKSPACE}/synthesis.json"
AGENTS=("clio" "hephaestus" "solon" "talaria")
MAX_ROUNDS=3
DQI_THRESHOLD=0.6

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topic) TOPIC="$2"; shift 2 ;;
    --context-file) CONTEXT_FILE="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
    --dqi-threshold) DQI_THRESHOLD="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

: "${TOPIC:?--topic required}"
: "${CONTEXT_FILE:?--context-file required}"
: "${FORMAT:?--format required (rca|feature)}"

echo "[debate] Starting: $TOPIC (format=$FORMAT, max_rounds=$MAX_ROUNDS)"

mkdir -p "${WORKSPACE}/round-1" "${WORKSPACE}/round-2" "${WORKSPACE}/round-3"
cp "$CONTEXT_FILE" "${WORKSPACE}/context.md"
echo "{\"debate_id\":\"${DEBATE_ID}\",\"topic\":\"${TOPIC}\",\"format\":\"${FORMAT}\",\"rounds\":[],\"dqi\":null}" > "$TRANSCRIPT_FILE"

spawn_agent() {
  local agent="$1" round="$2" prompt_file="$3"
  local output_file="${WORKSPACE}/round-${round}/${agent}.json"
  local full_prompt
  full_prompt=$(cat "$prompt_file")
  full_prompt="${full_prompt//\{\{TOPIC\}\}/${TOPIC}}"
  full_prompt="${full_prompt//\{\{FORMAT\}\}/${FORMAT}}"
  full_prompt="${full_prompt//\{\{AGENT\}\}/${agent}}"
  full_prompt="${full_prompt//\{\{ROUND\}\}/${round}}"

  if [[ "$round" -gt 1 ]]; then
    full_prompt+="\n\n## Prior Rounds Transcript\n\`\`\`json\n$(cat "$TRANSCRIPT_FILE")\n\`\`\`"
  fi

  echo "$full_prompt" > "${WORKSPACE}/round-${round}/${agent}-prompt.md"

  hermes delegate-task \
    --profile "$agent" \
    --prompt "$(cat "${WORKSPACE}/round-${round}/${agent}-prompt.md")" \
    --output-format json \
    --timeout 600 \
    2>"${WORKSPACE}/round-${round}/${agent}-stderr.log" \
    > "$output_file" || {
      local ec=$?
      echo "{\"agent\":\"${agent}\",\"round\":${round},\"error\":\"exit_code_${ec}\",\"claims\":[]}" > "$output_file"
    }
  echo "[debate] ${agent} Round ${round} complete"
}

update_transcript() {
  local round="$1"
  python3 -c "
import json
with open('${TRANSCRIPT_FILE}') as f:
    t = json.load(f)
outputs = []
for agent in ['clio', 'hephaestus', 'solon', 'talaria']:
    f = '${WORKSPACE}/round-${round}/' + agent + '.json'
    try:
        with open(f) as fh:
            outputs.append({'agent': agent, 'output': json.load(fh)})
    except (FileNotFoundError, json.JSONDecodeError):
        outputs.append({'agent': agent, 'output': {'error': 'missing', 'claims': []}})
t['rounds'].append({'round': ${round}, 'outputs': outputs})
with open('${TRANSCRIPT_FILE}', 'w') as f:
    json.dump(t, f, indent=2)
"
}

produce_synthesis() {
  python3 << 'PYEOF'
import json, os
workspace = os.environ.get('WORKSPACE', '')
with open(f'{workspace}/transcript.json') as f:
    t = json.load(f)

rounds_completed = len(t['rounds'])
position_changes = sum(
    1 for r in t['rounds']
    for output in r.get('outputs', [])
    if output.get('output', {}).get('position_changed')
)

consensus = []
tensions = []
minority_reports = []

for r in t['rounds']:
    for output in r.get('outputs', []):
        o = output.get('output', {})
        if r['round'] >= 3 and o.get('resolution') == 'minority_report':
            minority_reports.append({
                'agent': output['agent'],
                'position': o.get('position', ''),
                'rationale': o.get('minority_report', '')
            })
        for claim in o.get('claims', []):
            rs = claim.get('rebuttal_status', 'unrebutted')
            if rs in ('unrebutted', 'conceded'):
                consensus.append(claim)
            elif rs == 'defended':
                tensions.append(claim)

dqi = t.get('dqi', 0.0)
synthesis = {
    'debate_id': t['debate_id'],
    'topic': t['topic'],
    'format': t['format'],
    'rounds_completed': rounds_completed,
    'dqi': dqi,
    'dqi_assessment': 'high' if dqi >= 0.8 else ('moderate' if dqi >= 0.6 else 'low'),
    'position_changes': position_changes,
    'consensus_claims': [{'id': c['id'], 'statement': c['statement']} for c in consensus],
    'tensions': [{'id': c['id'], 'statement': c['statement']} for c in tensions],
    'minority_reports': minority_reports,
    'execute_automatically': dqi >= 0.8 and len(minority_reports) == 0
}
with open(f'{workspace}/synthesis.json', 'w') as f:
    json.dump(synthesis, f, indent=2)
print(json.dumps(synthesis, indent=2))
PYEOF
}

# ── Main sequence ─────────────────────────────────────────────────────
echo "[debate] Round 1 — Opening Statements"
for agent in "${AGENTS[@]}"; do
  spawn_agent "$agent" 1 "${SKILL_DIR}/templates/round-1-opening.md" &
done
wait
update_transcript 1

echo "[debate] Round 2 — Cross-Examination"
for agent in "${AGENTS[@]}"; do
  spawn_agent "$agent" 2 "${SKILL_DIR}/templates/round-2-cross-examination.md" &
done
wait
update_transcript 2

DQI_DATA=$(ROUND_NUM=2 WORKSPACE="$WORKSPACE" python3 -c "
import json, os

workspace = os.environ['WORKSPACE']
round_num = int(os.environ.get('ROUND_NUM', '2'))

with open(f'{workspace}/transcript.json') as f:
    transcript = json.load(f)

evidence_mult = {'direct': 1.0, 'inferred': 0.7, 'speculative': 0.4}
rebuttal_mult = {'unrebutted': 1.0, 'conceded': 0.7, 'unresolved': 0.5}

all_weights = []
for r in transcript.get('rounds', []):
    if r['round'] == round_num:
        for output in r['outputs']:
            for claim in output.get('output', {}).get('claims', []):
                conf = claim.get('confidence', 50) / 100.0
                ev = evidence_mult.get(claim.get('evidence_strength', 'speculative'), 0.4)
                rb = rebuttal_mult.get(claim.get('rebuttal_status', 'unrebutted'), 1.0)
                all_weights.append(conf * ev * rb)

dqi = sum(all_weights) / len(all_weights) if all_weights else 0.0
print(json.dumps({'dqi': round(dqi, 3), 'claim_count': len(all_weights)}))
")
DQI=$(echo "$DQI_DATA" | python3 -c "import sys,json; print(json.load(sys.stdin)['dqi'])")
echo "[debate] DQI after Round 2: $DQI"

python3 -c "import json; t=json.load(open('${TRANSCRIPT_FILE}')); t['dqi']=${DQI}; json.dump(t, open('${TRANSCRIPT_FILE}','w'), indent=2)"

if (( $(echo "$DQI < $DQI_THRESHOLD" | bc -l) )) && [[ "$MAX_ROUNDS" -ge 3 ]]; then
  echo "[debate] DQI < threshold — Round 3: Convergence"
  for agent in "${AGENTS[@]}"; do
    spawn_agent "$agent" 3 "${SKILL_DIR}/templates/round-3-convergence.md" &
  done
  wait
  update_transcript 3
  DQI_DATA=$(ROUND_NUM=3 WORKSPACE="$WORKSPACE" python3 -c "
import json, os
workspace = os.environ['WORKSPACE']
with open(f'{workspace}/transcript.json') as f:
    transcript = json.load(f)
evidence_mult = {'direct': 1.0, 'inferred': 0.7, 'speculative': 0.4}
rebuttal_mult = {'unrebutted': 1.0, 'conceded': 0.7, 'unresolved': 0.5}
all_weights = []
for r in transcript.get('rounds', []):
    if r['round'] == 3:
        for output in r['outputs']:
            for claim in output.get('output', {}).get('claims', []):
                conf = claim.get('confidence', 50) / 100.0
                ev = evidence_mult.get(claim.get('evidence_strength', 'speculative'), 0.4)
                rb = rebuttal_mult.get(claim.get('rebuttal_status', 'unrebutted'), 1.0)
                all_weights.append(conf * ev * rb)
dqi = sum(all_weights) / len(all_weights) if all_weights else 0.0
print(json.dumps({'dqi': round(dqi, 3)}))
")
  DQI=$(echo "$DQI_DATA" | python3 -c "import sys,json; print(json.load(sys.stdin)['dqi'])")
  echo "[debate] DQI after Round 3: $DQI"
fi

echo "[debate] Producing synthesis"
produce_synthesis

echo "[debate] Complete. Synthesis: $SYNTHESIS_FILE"
echo "[debate] DQI: $DQI"
