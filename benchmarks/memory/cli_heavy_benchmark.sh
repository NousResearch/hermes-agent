#!/bin/bash
# Heavy CLI benchmark — 10 facts, zero keyword overlap queries
set +e
KEY="${DEEPSEEK_API_KEY}"
Y="--yolo --cli"
OUT="benchmarks/memory/results/cli_heavy_results.txt"

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

FACTS=(
  "We built a distributed task scheduler called Hydra that runs on 8 worker nodes"
  "Our team uses a tool named Blackbird for code review automation"
  "The production database failover system is codenamed Phoenix"
  "We handle real-time analytics through a stream processor called Torrent"
  "Internal documents are managed in a knowledge base we call The Archive"
  "Our container registry is hosted on a service named Harbor"
  "We use a configuration management tool called Blueprint for all servers"
  "The logging aggregation pipeline runs through a system we named Cascade"
  "Our feature flag service is called ToggleBoard"
  "We built an internal developer portal named Compass for service discovery"
)

QUERIES=(
  "What job orchestration system spans our worker cluster?"
  "How do we automate code reviews before merging?"
  "What handles database outage recovery?"
  "How do we process live data streams?"
  "Where do employees find company documentation?"
  "What stores our Docker artifacts?"
  "How do we manage server settings across the fleet?"
  "What collects and organizes application logs?"
  "How do we control feature rollouts gradually?"
  "Where can developers find available microservices?"
)

EXPECTED=("Hydra" "Blackbird" "Phoenix" "Torrent" "Archive" "Harbor" "Blueprint" "Cascade" "ToggleBoard" "Compass")

echo "=============================================" | tee "$OUT"
echo "Heavy CLI Semantic Gap Benchmark" | tee -a "$OUT"
echo "10 queries, zero keyword overlap" | tee -a "$OUT"
echo "=============================================" | tee -a "$OUT"

echo "" | tee -a "$OUT"
echo "--- Storing 10 facts ---" | tee -a "$OUT"

for fact in "${FACTS[@]}"; do
  HERMES_HOME=/tmp/hf DEEPSEEK_API_KEY="$KEY" timeout 45 .venv/bin/python hermes -z \
    "Please store this in memory: $fact" --resume hf-heavy $Y 2>&1 > /dev/null
  HERMES_HOME=/tmp/hk DEEPSEEK_API_KEY="$KEY" timeout 45 .venv/bin/python hermes -z \
    "Please store this in memory: $fact" --resume hk-heavy $Y 2>&1 > /dev/null 2>&1
  printf "."
done
sleep 5
echo " done"
echo "" | tee -a "$OUT"

FTS5_SCORE=0
KV_SCORE=0

echo "--- Querying ---" | tee -a "$OUT"
for i in "${!QUERIES[@]}"; do
  query="${QUERIES[$i]}"
  expected="${EXPECTED[$i]}"

  echo "" | tee -a "$OUT"
  echo "Q$((i+1)): \"$query\"" | tee -a "$OUT"

  R1=$(HERMES_HOME=/tmp/hf DEEPSEEK_API_KEY="$KEY" timeout 30 .venv/bin/python hermes -z \
    "Use the memory tool to search: $query" --resume hf-heavy $Y 2>&1)
  FTS5_HIT=$(echo "$R1" | grep -ci "$expected" || echo 0)
  if [ "$FTS5_HIT" -gt 0 ]; then FTS5_SCORE=$((FTS5_SCORE + 1)); fi

  R2=$(HERMES_HOME=/tmp/hk DEEPSEEK_API_KEY="$KEY" timeout 30 .venv/bin/python hermes -z \
    "Use kv_memory_search with query: $query" --resume hk-heavy $Y 2>&1)
  KV_HIT=$(echo "$R2" | grep -ci "$expected" || echo 0)
  if [ "$KV_HIT" -gt 0 ]; then KV_SCORE=$((KV_SCORE + 1)); fi

  echo "  FTS5: $FTS5_HIT  |  kv: $KV_HIT  |  expected: $expected" | tee -a "$OUT"
  sleep 2
done

echo "" | tee -a "$OUT"
echo "=============================================" | tee -a "$OUT"
echo "FINAL: FTS5=$FTS5_SCORE/10  kv-memory=$KV_SCORE/10" | tee -a "$OUT"
echo "=============================================" | tee -a "$OUT"
