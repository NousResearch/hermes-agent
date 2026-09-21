# Migration Audit Checklist

## Post-Migration Verification

### Plugin Discovery
- [ ] `~/.hermes/plugins/memory-zvec/` exists with `plugin.yaml` (category: memory)
- [ ] Symlink exists: `~/.hermes/profiles/<target>/plugins/memory-zvec` → `~/.hermes/plugins/memory-zvec`
- [ ] `find_provider_dir('memory-zvec')` resolves correctly in target profile context
- [ ] `discover_memory_providers()` lists `memory-zvec: available=True`

### Config
- [ ] `config.yaml` has `memory.provider: memory-zvec`
- [ ] `memory.memory_enabled: true`

### Data Integrity
- [ ] Zvec collection opens without error
- [ ] `doc_count` matches source LanceDB table row count
- [ ] Vector dimension matches (1024 for bge-m3:latest)

### Index Health
- [ ] HNSW index completeness = 1.0 (run `collection.optimize()` if 0.0)
- [ ] FTS index uses `jieba` tokenizer (not `standard`)
- [ ] FTS search returns results for English queries
- [ ] FTS search returns results for Chinese queries
- [ ] Scalar fields (role, session_id, created_at) have InvertIndex

### Search Functionality
- [ ] Vector search returns semantically relevant results
- [ ] FTS/keyword search returns keyword matches
- [ ] Hybrid search (vector + FTS + RRF) works
- [ ] Scalar filter (session_id, role) works

### Hermes Integration
- [ ] Gateway log shows `Memory provider 'memory-zvec' registered`
- [ ] Gateway log shows `ZvecMemoryProvider initialized`
- [ ] Gateway log shows `Memory provider 'memory-zvec' activated`
- [ ] `vec_memory_stats` returns `backend: zvec`
- [ ] New memories can be added via `vec_memory_add`
- [ ] Memories can be recalled via `vec_memory_search`
- [ ] Session turn sync writes work (check `on_session_end` in log)

### Rollback Readiness
- [ ] LanceDB data directory still exists (if migrating from LanceDB)
- [ ] Can switch back by changing `memory.provider` in config.yaml
