# Ops — provozní soubory hetzner serveru (46.225.31.161)

Source-of-truth pro ručně nasazené provozní soubory, které **nežijí jinde v gitu**.
Toto je archiv/reference — **není to auto-deploy**. Po editaci je nutné soubor
ručně zkopírovat na cílové místo na serveru (viz níže) a restartovat příslušnou službu.

## `systemd/hermes-agent.service`

Cíl na serveru: `/etc/systemd/system/hermes-agent.service`

Telegram gateway Hermes agenta. Obsahuje **DNS readiness guard** (`ExecStartPre`) —
`network-online.target` negarantuje funkční DNS resolver, takže při bootu hrozil race,
kdy gateway nestihl resolvovat `api.telegram.org`, uvízl na sticky fallback IP a leaknul
mrtvý socket. Guard čeká až 60 s na resolver (viditelný log, žádný tichý fallback), pak
spustí stejně.

Deploy:
```bash
sudo cp systemd/hermes-agent.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl restart hermes-agent
```

## `paperclip/` — Paperclip denní/týdenní Telegram report

Cíl na serveru: `/home/leos/paperclip-agent-daemon/` (tento adresář **není git repo**).
Cron: `0 8 * * *` daily, `0 9 * * 1` weekly.

- `daily_report.py` — generuje denní přehled AI firem z Paperclip control plane.
  **Control plane běží na `localhost:3100`** (ne na veřejné `paperclip.frigeble.com` —
  tu port 443 zabral Plane PM proxy). URL se čte z `PAPERCLIP_API_URL` v `.env`.
  Login credentials (`PAPERCLIP_BOARD_EMAIL` / `PAPERCLIP_BOARD_PASSWORD`) se čtou
  **výhradně z `.env`** — žádný hardcoded fallback (fail-fast `exit 2` když chybí).
- `run_daily_report.sh` — wrapper: načte secrety z `.env`, spustí `daily_report.py`,
  pošle výsledek přes Telegram Bot API. Logy: `~/.hermes/cron/output/`.
- `run_weekly_analysis.sh` — týdenní varianta nad `trajectory.py`.

Vyžadované klíče v `~/paperclip-agent-daemon/.env` (chmod 600, **necommitovat**):
`PAPERCLIP_API_URL`, `PAPERCLIP_DAEMON_SECRET`, `PAPERCLIP_BOARD_EMAIL`,
`PAPERCLIP_BOARD_PASSWORD` + `TELEGRAM_BOT_TOKEN` v `~/.hermes/.env`.
