# Qualification Packet Timestamp Migration Guide

Version: 1.0  
Date: 2026-04-03  
Scope: OpenClaw qualification packet timestamp migration

## 1. Overview

This guide explains how to migrate legacy qualification packets to include required timestamps:

- **`evaluated_at`**: When the qualification was evaluated
- **`scorecard.scored_at`**: When the scorecard was generated  
- **`scorecard.cost.provider_evidence_updated_at`**: When provider evidence was last updated

### Why migrate?
- **Fail-closed behavior**: Legacy packets fail routing (correct behavior)
- **Truth surface stability**: Eliminates "stale qualification" noise
- **Comparability**: Ensures all packets have consistent timestamp metadata
- **Freshness tracking**: Enables proper freshness checking in routing

## 2. Migration Tools

### Batch Migration Script
```bash
python scripts/qualification_packet_batch_backfill.py
```

### Individual Packet Backfill
```bash
python scripts/qualification_packet_timestamp_backfill.py
```

## 3. Migration Steps

### Step 1: Dry-run (preview)
Preview what will be migrated:

```bash
python scripts/qualification_packet_batch_backfill.py \
  --directory state/ \
  --dry-run \
  --json
```

**Output example:**
```json
{
  "timestamp": "2026-04-03T10:30:00Z",
  "directory": "state/",
  "apply": false,
  "force_fresh": false,
  "validate": true,
  "backup": true,
  "total_found": 3,
  "migrated": 0,
  "failed": 0,
  "details": [
    {
      "path": "state/continuity/latest/xe306_deepseek_helper_lane_qualification_packet_2026-03-29.json",
      "success": true,
      "backfilled_fields": ["scored_at", "provider_evidence_updated_at"],
      "dry_run": true
    }
  ]
}
```

### Step 2: Apply migration
Apply the migration (creates backups):

```bash
python scripts/qualification_packet_batch_backfill.py \
  --directory state/ \
  --apply \
  --validate
```

**Safety features:**
- Creates `.backup` files before modifying
- Validates schema after backfill
- Continues on error with detailed reporting

### Step 3: Verify migration
Verify no legacy packets remain:

```bash
python scripts/qualification_packet_batch_backfill.py \
  --directory state/ \
  --dry-run \
  --json
# Should show "total_found": 0
```

## 4. Migration Scenarios

### Scenario A: Standard migration
Backfill only missing timestamps:

```bash
python scripts/qualification_packet_batch_backfill.py \
  --directory state/ \
  --apply \
  --validate
```

### Scenario B: Force fresh migration  
Update all timestamps to current time (useful for consistency):

```bash
python scripts/qualification_packet_batch_backfill.py \
  --directory state/ \
  --force-fresh \
  --apply \
  --validate
```

### Scenario C: Targeted migration
Migrate specific directories:

```bash
# Migrate only continuity directory
python scripts/qualification_packet_batch_backfill.py \
  --directory state/continuity/ \
  --apply \
  --validate

# Migrate only latest snapshots
python scripts/qualification_packet_batch_backfill.py \
  --directory state/continuity/latest/ \
  --apply \
  --validate
```

### Scenario D: Integration with existing backfill utility
Use batch mode in existing utility:

```bash
python scripts/qualification_packet_timestamp_backfill.py \
  --batch state/continuity/latest/*.json \
  --dry-run \
  --json
```

## 5. Risk Management

### Backups
- Original files backed up with `.backup` extension
- Manual restore: `cp file.json.backup file.json`
- Automatic cleanup not provided (manual decision)

### Validation
- Schema validation ensures compliance
- Invalid packets are skipped (not modified)
- Validation errors reported in summary

### Rollback
1. **Individual packet**: `cp packet.json.backup packet.json`
2. **Batch rollback**: Use backup files (`.backup` extension)
3. **Verification**: Check routing accepts rolled-back packets

### Error Recovery
- Script continues on error
- Failed packets reported in summary
- Errors include path and error message
- Can retry failed packets individually

## 6. Integration

### CI/CD Pipeline
Add to qualification packet generation pipelines:

```yaml
# Example GitHub Actions workflow
jobs:
  migrate-qualification-packets:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Migrate legacy packets
        run: |
          python scripts/qualification_packet_batch_backfill.py \
            --directory state/ \
            --apply \
            --validate
```

### Scheduled Migration
Schedule regular migration scans:

```bash
# Cron job (weekly)
0 2 * * 0 python /path/to/scripts/qualification_packet_batch_backfill.py \
  --directory /path/to/state/ \
  --apply \
  --validate \
  --json > /var/log/qualification_migration.log
```

### Monitoring
Monitor for new legacy packets:

```bash
# Check for legacy packets (alert if > 0)
python scripts/qualification_packet_batch_backfill.py \
  --directory state/ \
  --dry-run \
  --json | jq '.total_found'
```

## 7. Troubleshooting

### Common Issues

#### Issue: "No legacy qualification packets found"
**Cause**: Directory doesn't contain qualification packets or all packets already have timestamps
**Solution**: Verify directory path and check packet structure

#### Issue: Schema validation fails
**Cause**: Packet doesn't conform to qualification packet schema
**Solution**: 
1. Check packet structure manually
2. Fix schema violations
3. Retry migration

#### Issue: Permission denied
**Cause**: Insufficient permissions to write files
**Solution**: Check file permissions and run with appropriate user

#### Issue: Backup files accumulating
**Cause**: Multiple migrations creating multiple backups
**Solution**: Manually clean up old backup files when confident

### Debug Commands

```bash
# Check individual packet
python scripts/qualification_packet_timestamp_backfill.py \
  --input state/continuity/latest/packet.json \
  --dry-run \
  --json

# Validate packet schema
python scripts/qualification_packet_timestamp_backfill.py \
  --input state/continuity/latest/packet.json \
  --validate-only \
  --json

# Check timestamp completeness
python -c "
import json, sys
with open('state/continuity/latest/packet.json') as f:
    data = json.load(f)
missing = []
if not data.get('evaluated_at'): missing.append('evaluated_at')
scorecard = data.get('scorecard', {})
if not scorecard.get('scored_at'): missing.append('scored_at')
cost = scorecard.get('cost', {})
if not cost.get('provider_evidence_updated_at'): missing.append('provider_evidence_updated_at')
print('Missing:', missing)
"
```

## 8. Post-Migration Verification

### Routing Test
Test that migrated packets pass routing:

```bash
# Use session topology router to test
python scripts/session_topology_router.py \
  --request-json '{"task_class": "implementation", "risk_tier": "medium"}' \
  --qualification-signal state/continuity/latest/migrated_packet.json \
  --dry-run
```

### Freshness Check
Verify timestamps are within acceptable ranges:

```bash
python -c "
import json, datetime, sys
with open('state/continuity/latest/packet.json') as f:
    data = json.load(f)

now = datetime.datetime.now(datetime.timezone.utc)
eval_at = datetime.datetime.fromisoformat(data['evaluated_at'].replace('Z', '+00:00'))
scored_at = datetime.datetime.fromisoformat(data['scorecard']['scored_at'].replace('Z', '+00:00'))
provider_at = datetime.datetime.fromisoformat(data['scorecard']['cost']['provider_evidence_updated_at'].replace('Z', '+00:00'))

print(f'evaluated_at age: {(now - eval_at).total_seconds() / 3600:.1f}h')
print(f'scored_at age: {(now - scored_at).total_seconds() / 3600:.1f}h')
print(f'provider_evidence_updated_at age: {(now - provider_at).total_seconds() / 3600:.1f}h')
"
```

### Schema Compliance
Verify all migrated packets comply with schema:

```bash
python scripts/qualification_packet_timestamp_backfill.py \
  --batch state/continuity/latest/*.json \
  --validate-only \
  --json | jq '.details[] | select(.validation.valid == false)'
```

## 9. Maintenance

### Regular Scans
Schedule regular scans to catch new legacy packets:

```bash
# Weekly scan (dry-run)
0 3 * * 0 python scripts/qualification_packet_batch_backfill.py \
  --directory state/ \
  --dry-run \
  --json > /tmp/qualification_scan.log

# Alert if legacy packets found
if grep -q '"total_found": [1-9]' /tmp/qualification_scan.log; then
  echo "WARNING: Legacy qualification packets found"
  # Send alert
fi
```

### Cleanup Old Backups
After confirming migration success, clean up backup files:

```bash
# List backup files
find state/ -name "*.json.backup" -type f

# Remove backup files (after verification)
# find state/ -name "*.json.backup" -type f -delete
```

### Update Qualification Packet Generation
Ensure new qualification packets include all required timestamps:

1. Use template: `docs/ops/templates/model_qualification_packet.template.json`
2. Integrate timestamp validation into generation pipelines
3. Add pre-commit hooks to validate new packets

## 10. Support

### Getting Help
- Check existing qualification packet examples
- Review test cases in `tests/test_qualification_packet_batch_backfill.py`
- Consult routing policy: `docs/ops/session_topology_routing_policy_v1.json`

### Reporting Issues
1. Run with `--json` flag for detailed output
2. Include error message and packet path
3. Check permissions and disk space
4. Verify Python dependencies are installed

### Emergency Rollback
If migration causes issues:

```bash
# Restore from backups
find state/ -name "*.json.backup" -exec sh -c 'cp "$1" "${1%.backup}"' _ {} \;

# Verify restoration
python scripts/qualification_packet_batch_backfill.py \
  --directory state/ \
  --dry-run \
  --json
```

---

**Migration Complete When:**  
✓ No legacy packets found in dry-run scan  
✓ All migrated packets validate against schema  
✓ Routing accepts migrated packets  
✓ Backup files can be safely removed (after verification)