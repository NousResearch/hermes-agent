#!/bin/bash
# Hard semantic gap test — FTS5 cannot cross this gap
set +e

KEY="${DEEPSEEK_API_KEY}"
HERMES=".venv/bin/python hermes"
YOLO="--yolo --cli"
OUT="benchmarks/memory/results/cli_semantic_gap_results.txt"

rm -rf /tmp/hermes_gap_fts5 /tmp/hermes_gap_kv
mkdir -p /tmp/hermes_gap_fts5 /tmp/hermes_gap_kv benchmarks/memory/results

# ── Configs ────────────────────────────────────────────────
cat > /tmp/hermes_gap_fts5/config.yaml << 'EOF'
model: {provider: deepseek, model: deepseek-chat}
memory: {provider: ""}
EOF

cat > /tmp/hermes_gap_kv/config.yaml << 'EOF'
model: {provider: deepseek, model: deepseek-chat}
memory: {provider: kv_memory}
plugins:
  kv-memory: {embedding_backend: sentence-transformers, top_k: 5, storage_mode: fp16, diversity_lambda: 1.0}
EOF

echo "=============================================" | tee "$OUT"
echo "CLI Semantic Gap Test" | tee -a "$OUT"
echo "=============================================" | tee -a "$OUT"

# ── Teaching (same for both) ───────────────────────────────
# Use vocabulary that's DISTINCT from the queries
FACTS=(
  "My team uses a column-oriented data warehouse called ClickHouse for analytics. It runs on host 192.168.50.10 with default port 9000."
  "We handle async task queues with Celery backed by RabbitMQ. The broker URL is amqp://worker:s3cr3t@msg-broker.internal:5672/vhost1."
  "Our authentication microservice listens on port 8443 and uses JWT tokens signed with RS256. The public key fingerprint is SHA256:a1b2c3d4e5f6."
  "The CI pipeline triggers on push to main via GitHub Actions. Our self-hosted runner is at build-node-07 with label 'gpu-enabled'."
  "We migrated our primary database from MySQL 5.7 to PostgreSQL 15 last quarter. The new connection is postgresql://app_user:p@ssw0rd@pg-cluster.internal:5432/appdb."
)

QUERIES=(
  "What columnar analytics engine do we use?"              # → ClickHouse (zero overlap: "columnar analytics engine" vs "column-oriented data warehouse")
  "What message queue broker handles our background jobs?"  # → RabbitMQ (zero overlap: "message queue broker" "background jobs" vs "async task queues")
  "How does our auth service verify user identity?"         # → JWT/RS256 (zero overlap: "verify user identity" vs "authentication microservice")
  "What triggers our build pipeline on code changes?"       # → GitHub Actions (semantic: "build pipeline" "code changes" vs "CI pipeline triggers on push")
  "What database engine did we switch to from MySQL?"       # → PostgreSQL (semantic: "switch to" vs "migrated from")
)

echo "" | tee -a "$OUT"
echo "Teaching 5 facts to FTS5..." | tee -a "$OUT"
FTS5_SCORE=0
KV_SCORE=0

for i in "${!FACTS[@]}"; do
  fact="${FACTS[$i]}"
  short=$(echo "$fact" | cut -c1-50)
  echo "  Storing: $short..." | tee -a "$OUT"
  
  HERMES_HOME=/tmp/hermes_gap_fts5 DEEPSEEK_API_KEY="$KEY" timeout 60 $HERMES -z \
    "IMPORTANT - store this fact using the memory tool: $fact" \
    --resume gap-fts5 $YOLO 2>&1 > /dev/null
    
  HERMES_HOME=/tmp/hermes_gap_kv DEEPSEEK_API_KEY="$KEY" timeout 60 $HERMES -z \
    "IMPORTANT - store this fact using the memory tool: $fact" \
    --resume gap-kv $YOLO 2>&1 > /dev/null
    
  sleep 2
done

echo "" | tee -a "$OUT"
echo "Querying FTS5 and kv-memory..." | tee -a "$OUT"
echo "" | tee -a "$OUT"

for i in "${!QUERIES[@]}"; do
  query="${QUERIES[$i]}"
  fact="${FACTS[$i]}"
  
  # Extract expected answer words from the fact
  if [[ $i -eq 0 ]]; then expected="ClickHouse"; fi
  if [[ $i -eq 1 ]]; then expected="RabbitMQ\|Celery"; fi
  if [[ $i -eq 2 ]]; then expected="JWT\|RS256\|8443"; fi
  if [[ $i -eq 3 ]]; then expected="GitHub\|Actions\|build-node"; fi
  if [[ $i -eq 4 ]]; then expected="PostgreSQL\|pg-cluster"; fi
  
  echo "Q$((i+1)): \"$query\"" | tee -a "$OUT"
  
  # FTS5
  FTS5_R=$(HERMES_HOME=/tmp/hermes_gap_fts5 DEEPSEEK_API_KEY="$KEY" timeout 45 $HERMES -z \
    "Search memory for: $query" --resume gap-fts5 $YOLO 2>&1)
  fts5_hit=$(echo "$FTS5_R" | grep -ci "$expected" || echo "0")
  FTS5_SCORE=$((FTS5_SCORE + fts5_hit))
  echo "  FTS5:    hit=$fts5_hit" | tee -a "$OUT"
  
  # kv-memory
  KV_R=$(HERMES_HOME=/tmp/hermes_gap_kv DEEPSEEK_API_KEY="$KEY" timeout 45 $HERMES -z \
    "Search memory for: $query" --resume gap-kv $YOLO 2>&1)
  kv_hit=$(echo "$KV_R" | grep -ci "$expected" || echo "0")
  KV_SCORE=$((KV_SCORE + kv_hit))
  echo "  kv-mem:  hit=$kv_hit" | tee -a "$OUT"
  
  sleep 3
done

echo "" | tee -a "$OUT"
echo "=============================================" | tee -a "$OUT"
echo "RESULTS: FTS5=$FTS5_SCORE/5  kv-memory=$KV_SCORE/5" | tee -a "$OUT"
echo "=============================================" | tee -a "$OUT"
