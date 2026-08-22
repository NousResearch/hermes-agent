# Memory JSONL Repair Pattern (cartelogic-v2)

## Contexte

`memory/operational_memory.jsonl` contient des records JSON line-delimited. Chaque ligne doit respecter le schéma :

```json
{"action":"add","record":{"fact_id":"...","fact_type":"milestone","record_type":"operational_fact","project":"lah-stack","repo":"lah-stack-tools",...}}
```

## Problème connu

Les records générés par le pipeline d'autonomie peuvent avoir deux déviations :

1. **Format plat** : `{"ts":"...","type":"mission_complete","mission":"...",...}` — pas de `action`/`record` wrapper, pas de `fact_id`
2. **Record_type manquant** : `{"action":"add","record":{"fact_id":"...",...}}` — le champ `record_type` est vide ou absent

## Pattern de réparation (commit 1af315e)

Remplacer un record plat par l'enveloppe standard :

```javascript
const fs = require('fs');
const lines = fs.readFileSync('memory/operational_memory.jsonl', 'utf8').split('\n');
const oldLine = lines[index]; // index = lineNumber - 1
const parsed = JSON.parse(oldLine);
const newLine = JSON.stringify({
  action: 'add',
  record: {
    fact_id: parsed.verdict || parsed.mission || 'UNKNOWN',
    fact_type: 'milestone',
    record_type: 'operational_fact',
    project: 'lah-stack',
    repo: 'lah-stack-tools',
    ...parsed, // conserve toutes les données originales
  }
});
lines[index] = newLine;
fs.writeFileSync('memory/operational_memory.jsonl', lines.join('\n'), 'utf8');
```

## Vérification

```bash
python3 -m v2.operational.cli verify
# Attendu : integrity_status=OK records=<N>
```

## Prior repair reference

- Commit `1af315e` : repair invalid log action at record 479
- Commit `8090cdf` : MEM2 repair — 10 records fixed (494 + 9 related)
