# BAC CIRT monitor source

`bac_cirt_monitor.py` is a stable-input collector for Hermes cron, not a valuation or thesis model. It reads Bank of America Investor Relations' latest current report and latest quarterly/annual filing, OCC RSS entries that name Bank of America, and FinCEN's official enforcement-action list entries for Bank of America. The output includes its immutable release-directory name but no fetch timestamp, so Hermes' existing `monitor-script` hash gate suppresses unchanged source snapshots and run receipts show which release supplied the data.

The cron prompt owns source review, comparison with the dated CIRT baseline and prior forecasts, recording the evidence delta and verdict in that job's durable notepad, and deciding whether to deliver. It must return `[SILENT]` for immaterial or unchanged evidence. Only a material thesis or valuation change is sent through the user's existing Telegram route.

Source failures exit nonzero rather than presenting an empty snapshot. Current coverage is deliberately narrow: this watches BAC's two latest filing slots, BAC-named OCC notices, and FinCEN enforcement-list entries for BAC. It does not poll live prices or every regulator's general notices; on a filing or regulator wake, the analyst rechecks relevant primary sources and uses only a timestamped official quote for P/TBV.
