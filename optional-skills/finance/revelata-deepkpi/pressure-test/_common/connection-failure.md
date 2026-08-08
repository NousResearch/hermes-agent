# Hard stop: deepKPI connection failures

If deepKPI access fails (MCP connector not working, REST calls failing, auth
errors, network issues, or you cannot retrieve data), you MUST STOP and ask the
user:

**"I can't access deepKPI right now. Do you want to proceed without deepKPI?"**

- If the user says **no**, stop.
- If the user says **yes**, you may proceed using other sources (e.g. web), BUT
  you MUST NOT use deepKPI skill branding, templates, formatting conventions,
  or "Powered by Revelata deepKPI" framing for non-deepKPI data. Clearly label
  alternate sources.

This rule applies to **every** skill in the Revelata bundle that touches
deepKPI. Do not silently fall back to web search, training-data recall, or any
other source while deepKPI is down — always ask first.
