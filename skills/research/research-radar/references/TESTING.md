# Testing

Manual test steps:

1. Start Hermes CLI
2. Invoke the `research-radar` skill
3. Request a briefing for a topic such as "AI coding agents"
4. Confirm the Markdown report is created successfully
5. Confirm output includes:
   - Key Developments
   - Emerging Themes
   - Risks
   - Opportunities
   - Recommended Actions
6. Run the skill again for the same topic with a dated filename
7. Confirm the newer report includes:
   - What Changed Since Last Run
8. Confirm dated output filenames can be created as expected

Example output files:
- ai_agents_briefing.md
- ai_agents_briefing_YYYY-MM-DD.md
- ai_agents_briefing_YYYY-MM-DD_v2.md
