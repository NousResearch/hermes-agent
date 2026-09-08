# Example: shared origins and unresolved evidence

This is a synthetic illustration, not a real study or a factual performance claim.
Replace SKILL and RUN with absolute paths. Initialize a run before these commands.

```sh
python3 SKILL/scripts/evidence_ledger.py init RUN/evidence.json --topic "Synthetic pilot evaluation" --question "Q1: Did the pilot reduce processing time?" --question "Q2: Does this generalize?"
python3 SKILL/scripts/evidence_ledger.py add-source RUN/evidence.json --url https://example.com/pilot --title "Synthetic pilot report" --source-type primary --question "Q1: Did the pilot reduce processing time?" --independence-group pilot-dataset --independence-note "Original pilot dataset; self-reported by its operator" --summary "The operator reports a shorter median in its pilot." --evidence '{"excerpt":"Median processing time fell from 10 to 8 minutes in this pilot.","locator":"Results, paragraph 2"}' --limitation "Self-reported; no independent replication"
python3 SKILL/scripts/evidence_ledger.py add-source RUN/evidence.json --url https://example.org/news --title "Synthetic news coverage" --source-type secondary --question "Q1: Did the pilot reduce processing time?" --independence-group pilot-dataset --independence-note "This article repeats the first report; no new data" --summary "Coverage repeats the pilot result." --evidence '{"excerpt":"The operator reports a reduction from 10 to 8 minutes.","locator":"Paragraph 4; cites the pilot report"}'
python3 SKILL/scripts/evidence_ledger.py add-claim RUN/evidence.json --text "The pilot operator reports that median processing time fell from 10 to 8 minutes." --question "Q1: Did the pilot reduce processing time?" --status single-source --source 1,2 --evidence S1E1 --evidence S2E1 --reasoning "Both publications depend on the same pilot dataset." --caveat "This does not establish general effectiveness."
python3 SKILL/scripts/evidence_ledger.py add-claim RUN/evidence.json --text "The pilot result generalizes to other organizations." --question "Q2: Does this generalize?" --status unsupported --caveat "No independent replication was retrieved."
python3 SKILL/scripts/evidence_ledger.py check RUN/evidence.json
python3 SKILL/scripts/evidence_ledger.py brief RUN/evidence.json
```

Expected interpretation: the first narrow claim has direct passage support but only one
origin group. The checker warns about insufficient independent support. The second is an
unresolved gap, not a report conclusion. Two hostnames do not turn one study into two studies.
For an unassessed source, omit both independence arguments; it contributes no verified group.
For a counterargument, use `--counter-source` with its corresponding `--evidence` passage.

Report wording: “The operator reported a shorter median in its pilot [1]. News coverage [2]
repeats the same dataset. Independent replication was not found; generalizability is unresolved.”
