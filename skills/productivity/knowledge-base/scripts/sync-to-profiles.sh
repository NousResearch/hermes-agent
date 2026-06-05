#!/bin/bash
# Sync knowledge-base skill to worker profiles
for p in researcher researcher-videos; do
  rsync -a --delete \
    /root/.hermes/skills/productivity/knowledge-base/ \
    /root/.hermes/profiles/$p/skills/productivity/knowledge-base/ && echo "$p: OK" || echo "$p: FAIL"
done
