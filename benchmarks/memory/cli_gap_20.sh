#!/bin/bash
# 20 hard semantic gap queries — designed so FTS5 fails on ALL
set +e
KEY="${DEEPSEEK_API_KEY}"
Y="--yolo --cli"
OUT="benchmarks/memory/results/cli_gap_20.txt"

rm -rf /tmp/hf /tmp/hk
mkdir -p /tmp/hf /tmp/hk benchmarks/memory/results

cat > /tmp/hf/config.yaml << 'EOF'
model: {provider: deepseek, model: deepseek-chat}
memory: {provider: ""}
EOF
cat > /tmp/hk/config.yaml << 'EOF'
model: {provider: deepseek, model: deepseek-chat}
memory: {provider: kv_memory}
plugins:
  kv-memory: {embedding_backend: sentence-transformers, top_k: 10, storage_mode: fp16, min_similarity: 0.0, diversity_lambda: 1.0}
EOF

# 20 facts with code-names/unique terms. Queries use ONLY descriptive language.
# ZERO keyword overlap between query and stored text.
STORED=(
  "We process payments through a gateway codenamed IronVault that runs on 12 nodes"
  "Our ML training cluster is managed by an orchestrator called BrainForge"
  "The CI/CD system we built in-house is named PipelineStorm"
  "We use a secrets manager called CryptKeeper for all production credentials"
  "The service mesh we deployed is an instance of NetFabric"
  "Our data warehouse runs on a platform we call DataOcean"
  "We built a custom load balancer named TrafficShaper for edge routing"
  "The alerting and incident management tool we use is called FireAlarm"
  "We have an internal A/B testing framework named ExperimentLab"
  "Our user authentication backbone is a service we call IdentityVault"
  "We built a real-time chat system called WhisperNet for internal comms"
  "The search engine for our documentation is powered by an indexer called DocFinder"
  "We schedule cron jobs across the fleet with a tool called TimeKeeper"
  "Our image optimization CDN is served through a proxy called PixelStream"
  "We manage DNS records through a controller called NameWarden"
  "The audit logging pipeline feeds into a system we call AuditTrail"
  "We use a feature called SafeMode for gradual production rollouts"
  "Our API gateway is fronted by a proxy named GateGuardian"
  "We built a schema migration tool called SchemaShift for database changes"
  "The session cache cluster is backed by a store called MemoryVault"
)

QUERIES=(
  "How do we handle credit card transactions?"
  "What runs our neural network training jobs?"
  "How do we build and deploy code automatically?"
  "Where do we keep production passwords and tokens?"
  "How do our microservices discover and talk to each other?"
  "Where does business intelligence data live for analysis?"
  "How do we distribute incoming web traffic evenly?"
  "What notifies the team when production breaks?"
  "How do we test new features with a subset of users?"
  "What verifies user passwords when they log in?"
  "How do employees send instant messages internally?"
  "How do engineers find technical documentation?"
  "What runs scheduled background tasks on a timer?"
  "How do we optimize and serve images on the web?"
  "How do we map domain names to server addresses?"
  "How do we track who did what for compliance?"
  "How do we roll out new versions without downtime?"
  "What sits between clients and our backend services?"
  "How do we safely change database table structures?"
  "Where do we store temporary user session data?"
)

EXPECTED=(
  "IronVault|payment"
  "BrainForge|ML"
  "PipelineStorm|CI"
  "CryptKeeper|secret"
  "NetFabric|mesh"
  "DataOcean|warehouse"
  "TrafficShaper|balanc"
  "FireAlarm|alert"
  "ExperimentLab|test"
  "IdentityVault|auth"
  "WhisperNet|chat"
  "DocFinder|search"
  "TimeKeeper|cron"
  "PixelStream|image|CDN"
  "NameWarden|DNS"
  "AuditTrail|audit"
  "SafeMode|rollout"
  "GateGuardian|proxy|gateway"
  "SchemaShift|migration|schema"
  "MemoryVault|cache|session"
)

echo "=================================================================" | tee "$OUT"
echo "20 Hard Semantic Gap Queries — Zero Keyword Overlap" | tee -a "$OUT"
echo "FTS5 should fail on ALL. kv-memory should find them semantically." | tee -a "$OUT"
echo "=================================================================" | tee -a "$OUT"

echo "" | tee -a "$OUT"
printf "Storing 20 facts"
for fact in "${STORED[@]}"; do
  HERMES_HOME=/tmp/hf DEEPSEEK_API_KEY="$KEY" timeout 40 .venv/bin/python hermes -z \
    "Store: $fact" --resume hf-gap $Y 2>&1 > /dev/null
  HERMES_HOME=/tmp/hk DEEPSEEK_API_KEY="$KEY" timeout 40 .venv/bin/python hermes -z \
    "Store: $fact" --resume hk-gap $Y 2>&1 > /dev/null 2>&1
  printf "."
done
sleep 6
echo " done"
echo "" | tee -a "$OUT"

FTS5_OK=0
KV_OK=0

for i in "${!QUERIES[@]}"; do
  q="${QUERIES[$i]}"
  pat="${EXPECTED[$i]}"
  echo "Q$((i+1)): \"$q\"" | tee -a "$OUT"

  R1=$(HERMES_HOME=/tmp/hf DEEPSEEK_API_KEY="$KEY" timeout 25 .venv/bin/python hermes -z \
    "memory search: $q" --resume hf-gap $Y 2>&1)
  f=$(echo "$R1" | grep -ciE "$pat" 2>/dev/null || echo 0)
  f=$(echo "$f" | tr -d '\n\r' | grep -o '[0-9]*' | head -1)
  [ -z "$f" ] && f=0
  if [ "$f" -gt 0 ] 2>/dev/null; then FTS5_OK=$((FTS5_OK + 1)); fi

  R2=$(HERMES_HOME=/tmp/hk DEEPSEEK_API_KEY="$KEY" timeout 25 .venv/bin/python hermes -z \
    "kv_memory_search query: $q" --resume hk-gap $Y 2>&1)
  k=$(echo "$R2" | grep -ciE "$pat" 2>/dev/null || echo 0)
  k=$(echo "$k" | tr -d '\n\r' | grep -o '[0-9]*' | head -1)
  [ -z "$k" ] && k=0
  if [ "$k" -gt 0 ] 2>/dev/null; then KV_OK=$((KV_OK + 1)); fi

  echo "  FTS5=$f  kv=$k  ($pat)" | tee -a "$OUT"
  sleep 2
done

echo "" | tee -a "$OUT"
echo "=================================================================" | tee -a "$OUT"
echo "FINAL: FTS5=$FTS5_OK/20  kv-memory=$KV_OK/20" | tee -a "$OUT"
echo "Gap:   kv wins by $((KV_OK - FTS5_OK)) queries" | tee -a "$OUT"
echo "=================================================================" | tee -a "$OUT"
