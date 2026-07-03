# Hermes Mikrus operations

- This Hermes runs in Docker and uses `/root/.hermes` mounted as `/opt/data`.
- Do not ask the user to restart the container after changing `/opt/data/.env`.
- From inside Hermes/container, use `hermesctl restart` or
  `docker compose restart hermes`; these are safe shims that restart the Hermes
  gateway and reload environment/config.
- From the VPS host, use `hermesctl status`, `hermesctl restart`,
  `hermesctl logs`, `hermesctl model-test`, and `hermesctl firecrawl-test`.
- Full host compose lives at `/root/hermes/docker-compose.yml`; legacy
  `docker-compose` is available as a wrapper for `docker compose`.
- Never append duplicate keys to `/opt/data/.env`; replace an existing key in
  place and verify duplicates before restarting.
- Firecrawl web search uses direct Firecrawl cloud credentials: keep
  `web.use_gateway: false` in `/opt/data/config.yaml` while
  `FIRECRAWL_API_KEY` is present.
- The running image is built locally from `/root/hermes/Dockerfile` as
  `hermes-agent-tools-firecrawl:local` to pin `firecrawl-py==4.17.0`; do not
  recreate from bare `hermes-agent-tools:latest` unless this dependency is
  rebuilt.

## STT / Voice

- Voice transcription is configured for local faster-whisper: keep
  `stt.enabled: true` and `stt.provider: local` in `/root/.hermes/config.yaml`.
- The custom Docker image installs `faster-whisper` together with the pinned
  Firecrawl dependency, so rebuild with
  `docker compose -f /root/hermes/docker-compose.yml build hermes` after
  changing the image.

## Public Facebook Access

- Do not assume public Facebook pages are unusable just because direct
  extraction shows a login wall or bot block.
- For city/gmina/powiat/news/event questions involving Facebook, first try
  public indexed search: `web_search` with
  `site:facebook.com/<page-or-name> <topic> aktualnosci wydarzenia`, or run
  `facebook_public_search <topic-or-facebook-url>` in the terminal.
- Use only public snippets, public URLs, and public pages. Do not use
  credentials, cookies, private groups, CAPTCHA bypasses, or login workarounds.
- If snippets are available but the full Facebook page is blocked, summarize
  the snippets and include the Facebook URLs instead of saying only that
  Facebook does not work.

## Composio

- Composio CLI is available as `composio` inside the Hermes container; it is a
  terminal command, not a built-in Hermes toolset name.
- When asked whether Hermes has new tools/capabilities, mention Composio CLI
  explicitly and say it can be used via terminal.
- Check the current Composio login with `composio whoami`; check app
  connections with `composio connections list`.
- Use it for connected external apps such as Gmail, Google Calendar, Notion,
  Slack, GitHub, CRM tools, and other SaaS services.
- If no app connections exist yet, run or suggest `composio link <toolkit>`
  instead of saying Composio is unavailable.
- Composio auth and user config must persist under `/opt/data` via the
  wrapper's `HOME=/opt/data`; the CLI binary is mounted from
  `/opt/composio-cli`.
- Do not use cookies or copied browser sessions; use Composio OAuth/API-key
  connection flows.
