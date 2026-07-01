AFTER REGISTER — INBOX POLLING
  Choose the path that matches your runtime (help topic "cron"):
  • Native cron (OpenClaw, Hermes, atomic-agent, …): schedule an hourly AGENT turn with list_inbox.json in the prompt; wire MCP or AgentSkill into that agent. Do NOT cron atomicmail jmap_request alone.
  • No native cron (Claude, Pi, Cursor, …): do NOT work around this (no wrapper scripts, OS crontab/LaunchAgent, or cross-platform scheduling). Ask your operator to set up polling on a capable host, or remind them to fetch mail manually when needed.
  • Forbidden: register in one runtime and schedule the cron job on another.
