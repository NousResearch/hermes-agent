# Authority Transfer Storyboard Example

This example exercises Kubrick's complete standalone Hermes pipeline:

```text
brief → retrieval → private graph → structured audit → storyboard propagation
→ transition comparison → neutral adapter packet → Grok Imagine prompt packet
```

Run from the repository root:

```bash
python scripts/kubrick.py compile \
  --brief examples/authority-transfer-storyboard/brief.yaml \
  --ledger examples/authority-transfer-storyboard/symbolic-ledger.yaml \
  --mode storyboard \
  --storyboard-plan examples/authority-transfer-storyboard/storyboard-plan.yaml \
  --provider grok-imagine \
  --out out/kubrick/authority-transfer
```

The three frames preserve an empty command center and a visibly cracked badge while ownership and doorway access transfer from a supervisor to a subordinate. The former supervisor remains outside as persistent residue.

Expected final artifacts include the private graph, structured audits, storyboard state, transition report, neutral model-adapter packet, Grok Imagine prompt packet, schema reports, and compile receipt.
