# AlphaHunt Research Dry Run

Hermes can produce AlphaHunt project research YAML without using AlphaHunt API
keys or live endpoints.

Validate a Hermes sample first:

```bash
python -m gateway.alphahunt.research_yaml --validate docs/alphahunt/samples/protocol_ethena.yaml
```

Then feed the YAML to the AlphaHunt P08 dry-run/import script:

```bash
python3 /home/ubuntu/alphahunt/scripts/create_project_from_research.py docs/alphahunt/samples/protocol_ethena.yaml
python3 /home/ubuntu/alphahunt/scripts/create_project_from_research.py docs/alphahunt/samples/stock_example.yaml
python3 /home/ubuntu/alphahunt/scripts/create_project_from_research.py docs/alphahunt/samples/commodity_copper.yaml
python3 /home/ubuntu/alphahunt/scripts/create_project_from_research.py docs/alphahunt/samples/macro_fomc.yaml
python3 /home/ubuntu/alphahunt/scripts/create_project_from_research.py docs/alphahunt/samples/market_worldcup.yaml
```

This integration is dry-run only from the Hermes side:

- No AlphaHunt API key is required.
- Hermes does not write AlphaHunt production DBs.
- Hermes does not call live AlphaHunt endpoints.
- Hermes does not connect cron or notifications.
- Hermes does not execute trades, place wagers, sign transactions, or manage funds.

Any live handoff requires separate user authorization and a separate H7 design.
