# HERMES//HUB — zero-dependency stdlib server. The only optional extra is the
# official `anthropic` SDK, which unlocks the live Jarvis agent when
# HERMES_HUB_API_KEY is provided; everything else runs without it.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir anthropic

COPY server.py assistant.py automations.py ics.py sample_data.json ./
COPY public ./public

# Persist notes/tasks/sync state, feeds, calendars, automations and memory.
VOLUME /data

EXPOSE 8787

# HERMES_HUB_TOKEN   → access code required by browsers/API clients (recommended)
# HERMES_HUB_API_KEY → Anthropic API key for the live agent (optional)
# HERMES_HUB_MODEL   → override the Claude model (optional)
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8787", "--data-dir", "/data"]
