#!/usr/bin/env bash
# Idempotent installer: publish the browser-hosted Hermes Desktop on a headless host.
#
# Usage:
#   ./install-desktop.sh --desk-host desk.example.com [--login admin] [--password]
#                        [--port 9120] [--bind 127.0.0.1]
#                        [--trusted-proxy 172.16.0.0/12] [--no-caddy] [--skip-build]
#                        [--restart] [--dry-run] [--check]
#
# The gateway serves the Webapp itself (webapp.* config); this script only
# configures it, builds the renderer, and puts Caddy in front. It never writes a
# systemd unit for the Webapp — the gateway already is that process.
#
# Sign-in defaults to Portal OAuth through the Webapp's OWN client
# (hermes_cli/webapp_access: plan -> register -> serve), which keeps a dashboard
# on the same host working. It falls back to `hermes dashboard register` only on
# builds without webapp_access, and says so.
#
# Password mode reads the password from $DASH_PASSWORD: never pass a secret as
# an argument (it lands in shell history and `ps`).
set -euo pipefail

DESK_HOST=""; PORT=9120; LOGIN=""; AUTH=oauth; BIND=127.0.0.1
CHECK_ONLY=0; SKIP_BUILD=0; DO_RESTART=0; DRY_RUN=0; WANT_CADDY=1
PROXIES=()

while [ $# -gt 0 ]; do
  case "$1" in
    --desk-host) DESK_HOST=${2:?}; shift 2 ;;
    --port)      PORT=${2:?}; shift 2 ;;
    --login)     LOGIN=${2:?}; shift 2 ;;
    --bind)      BIND=${2:?}; shift 2 ;;
    --trusted-proxy) PROXIES+=("${2:?}"); shift 2 ;;
    --password)  AUTH=password; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --restart)   DO_RESTART=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --no-caddy)  WANT_CADDY=0; shift ;;
    --check)     CHECK_ONLY=1; shift ;;
    -h|--help)   sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$DESK_HOST" ] || { echo "--desk-host is required" >&2; exit 2; }
[ "$AUTH" = password ] && [ -z "$LOGIN" ] && { echo "--login is required with --password" >&2; exit 2; }
case "$BIND" in 127.0.0.1|0.0.0.0|::1|::) : ;; *) echo "--bind must be 127.0.0.1, ::1, 0.0.0.0 or ::" >&2; exit 2 ;; esac
if [ "$BIND" != 127.0.0.1 ] && [ "$BIND" != ::1 ] && [ ${#PROXIES[@]} -eq 0 ]; then
  echo "warning: --bind $BIND exposes the Webapp to the network; add --trusted-proxy <CIDR> so the sign-in callback and cookies are built as https" >&2
fi

HERE=$(cd "$(dirname "$0")/.." && pwd)
HERMES_HOME=${HERMES_HOME:-$HOME/.hermes}
CHECKOUT=$HERMES_HOME/hermes-agent
ENV_FILE=$HERMES_HOME/.env
CADDY=${CADDYFILE:-/etc/caddy/Caddyfile}
REG_FILE=$HERMES_HOME/webapp/sign-in-registration.json

say() { printf '\n== %s\n' "$*"; }
ok()  { printf '   ok  %s\n' "$*"; }
die() { printf '   !!  %s\n' "$*" >&2; exit 1; }
plan_only() { [ "$DRY_RUN" = 1 ]; }

resolve() {
  if command -v dig >/dev/null 2>&1; then
    dig +short "$1" A 2>/dev/null | tail -1
  elif command -v getent >/dev/null 2>&1; then
    getent ahosts "$1" 2>/dev/null | awk 'NR==1 {print $1}'
  else
    python3 -c 'import socket,sys; print(socket.gethostbyname(sys.argv[1]))' "$1" 2>/dev/null
  fi
}

say "preflight"
HERMES_BIN=$(command -v hermes || true)
[ -n "$HERMES_BIN" ] || die "hermes not on PATH"
ok "launcher $HERMES_BIN"

# The Webapp landed upstream on 2026-10-25 (463af9d5); older builds (incl. v2026.9.24) lack it.
"$HERMES_BIN" webapp --help >/dev/null 2>&1 || die \
  "this Hermes has no 'webapp' subcommand — update to a build that includes 463af9d5 (2026-10-25)"
ok "hermes webapp available"

[ -d "$CHECKOUT/apps/desktop" ] || die "no apps/desktop under $CHECKOUT — not a git install?"
ok "checkout $CHECKOUT"
[ -f "$ENV_FILE" ] || die "missing $ENV_FILE — run: hermes setup"
command -v caddy >/dev/null 2>&1 || die "caddy not installed (apt-get install -y caddy)"
ok "caddy $(caddy version 2>/dev/null | head -1)"

ip=$(resolve "$DESK_HOST")
[ -n "$ip" ] || die "$DESK_HOST does not resolve — create the A record first"
ok "$DESK_HOST -> $ip  (bind $BIND, port $PORT)"
[ "$CHECK_ONLY" = 1 ] && { echo; echo "check-only: preflight passed"; exit 0; }

VENV=$(ls -d "$HERMES_HOME"/installs/*/environments/*/venv/bin/python 2>/dev/null | head -1 || true)

if [ "$AUTH" = password ]; then
  say "password provider"
  [ -n "${DASH_PASSWORD:-}" ] || die "set DASH_PASSWORD in the environment"
  [ -n "$VENV" ] || die "no install venv found — hash manually (see SKILL.md step 2)"
  if plan_only; then ok "would write 3 HERMES_DASHBOARD_BASIC_AUTH_* keys to $ENV_FILE"; else
    HASH=$(cd "$CHECKOUT" && DASH_PASSWORD="$DASH_PASSWORD" "$VENV" -c \
      'import os; from plugins.dashboard_auth.basic import hash_password; print(hash_password(os.environ["DASH_PASSWORD"]))')
    [ -n "$HASH" ] || die "hash generation failed"
    ok "scrypt hash generated (${#HASH} chars)"
    LOGIN="$LOGIN" HASH="$HASH" ENV_FILE="$ENV_FILE" python3 - <<'PY'
import os, pathlib, re, secrets
path = pathlib.Path(os.environ["ENV_FILE"])
text = path.read_text() if path.exists() else ""
want = {
    "HERMES_DASHBOARD_BASIC_AUTH_USERNAME": os.environ["LOGIN"],
    "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH": os.environ["HASH"],
    "HERMES_DASHBOARD_BASIC_AUTH_SECRET": os.environ.get("HERMES_DASHBOARD_BASIC_AUTH_SECRET") or secrets.token_urlsafe(32),
}
lines = text.splitlines()
for key, value in want.items():
    pat = re.compile(rf"^{re.escape(key)}=.*$")
    hits = [i for i, line in enumerate(lines) if pat.match(line)]
    if hits:
        lines[hits[0]] = f"{key}={value}"
        for i in reversed(hits[1:]):
            del lines[i]
    else:
        lines.append(f"{key}={value}")
path.write_text("\n".join(lines).rstrip("\n") + "\n")
path.chmod(0o600)
print("   ok  3 basic-auth keys present")
PY
  fi
else
  say "Portal OAuth provider (Webapp's own client)"
  if [ -f "$REG_FILE" ]; then
    ok "Webapp sign-in already registered: $REG_FILE"
  elif [ -z "$VENV" ]; then
    die "no install venv found to run hermes_cli.webapp_access"
  else
    set +e
    ( cd "$CHECKOUT" && HERMES_HOME="$HERMES_HOME" "$VENV" - "$DESK_HOST" "$PORT" "$BIND" "$DRY_RUN" <<'PY'
import sys
from hermes_cli.webapp_access import WebappAccessError, plan, register, serve
host, port, url, callback = plan("public", "https://" + sys.argv[1])
bind, port_arg, dry = sys.argv[3], int(sys.argv[2]), sys.argv[4] == "1"
if dry:
    print(f"   ok  would register {callback}")
    print(f"   ok  would serve on {bind}:{port_arg} as {url}")
    raise SystemExit(0)
try:
    register(callback)
except WebappAccessError as exc:
    raise SystemExit(str(exc))
except Exception as exc:
    raise SystemExit(f"registration failed: {exc}")
serve(bind, url, port_arg)
print(f"   ok  registered {callback}")
print(f"   ok  webapp written: {bind}:{port_arg} {url}")
PY
    )
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      echo "   !! Webapp-client registration failed (rc=$rc)" >&2
      echo "      Falling back to 'hermes dashboard register'. That reuses the dashboard's" >&2
      echo "      client: a dashboard on this host loses Nous sign-in. See SKILL.md." >&2
      [ "$DRY_RUN" = 1 ] || "$HERMES_BIN" dashboard register --name "$DESK_HOST" \
        --redirect-uri "https://$DESK_HOST/auth/callback" || die "registration failed"
    fi
  fi
fi

say "webapp config"
if plan_only; then
  ok "would set webapp.enabled=true host=$BIND port=$PORT public_url=https://$DESK_HOST"
else
  "$HERMES_BIN" config set webapp.enabled true >/dev/null
  "$HERMES_BIN" config set webapp.host "$BIND" >/dev/null
  "$HERMES_BIN" config set webapp.port "$PORT" >/dev/null
  "$HERMES_BIN" config set webapp.public_url "https://$DESK_HOST" >/dev/null
  ok "webapp.enabled=true host=$BIND port=$PORT public_url=https://$DESK_HOST"
fi

if [ ${#PROXIES[@]} -gt 0 ]; then
  say "trusted proxies"
  JOINED=$(printf '%s\n' "${PROXIES[@]}" | paste -sd, -)
  if plan_only; then
    ok "would set dashboard.trusted_proxies = [$JOINED]"
  else
    ( cd "$CHECKOUT" && HERMES_HOME="$HERMES_HOME" LIST="$JOINED" "$VENV" - <<'PY'
import os
from hermes_cli.config import load_config, save_config
cfg = load_config()
dash = cfg.setdefault("dashboard", {})
dash["trusted_proxies"] = [p for p in os.environ["LIST"].split(",") if p]
save_config(cfg)
print("   ok  dashboard.trusted_proxies = " + ", ".join(dash["trusted_proxies"]))
PY
    )
  fi
fi

if [ "$SKIP_BUILD" = 1 ]; then
  say "renderer build (skipped)"
  [ -f "$CHECKOUT/apps/desktop/dist-webapp/index.html" ] || die "--skip-build but no dist-webapp bundle exists"
  ok "existing bundle present"
elif plan_only; then
  say "renderer build (dry run)"
  ok "would run: hermes webapp --build-only"
else
  say "renderer build (cold build is minutes)"
  "$HERMES_BIN" webapp --build-only || die "bundle build failed — see the log above"
  [ -f "$CHECKOUT/apps/desktop/dist-webapp/index.html" ] || die "build finished without dist-webapp/index.html"
  ok "bundle built"
fi

say "caddy"
if [ "$WANT_CADDY" = 0 ]; then
  ok "skipped (--no-caddy): point your own proxy at $BIND:$PORT"
elif [ ! -f "$CADDY" ]; then
  [ "$DRY_RUN" = 1 ] && { ok "would append a site block to $CADDY (missing now)"; } || die "no Caddyfile at $CADDY"
else
  if plan_only; then
    ok "would add/keep the site block for $DESK_HOST -> 127.0.0.1:$PORT"
  else
    cp -a "$CADDY" "$CADDY.bak-$(date +%Y%m%dT%H%M%SZ)"
    DESK_HOST="$DESK_HOST" PORT="$PORT" CADDY="$CADDY" HERE="$HERE" python3 - <<'PY'
import os, pathlib
path = pathlib.Path(os.environ["CADDY"])
text = path.read_text()
host, port = os.environ["DESK_HOST"], os.environ["PORT"]
if ("\n" + text).count("\n%s {" % host) > 0:
    print("   ok  site block exists: %s" % host)
else:
    template = pathlib.Path(os.environ["HERE"], "templates", "Caddyfile.site").read_text()
    path.write_text(text.rstrip("\n") + "\n\n" + template.replace("<DESK_HOST>", host).replace("<PORT>", port))
    print("   ok  site block added: %s" % host)
PY
    caddy validate --config "$CADDY" --adapter caddyfile >/dev/null 2>&1 || {
      cp -a "$(ls -t "$CADDY".bak-* | head -1)" "$CADDY"; die "caddy config invalid — restored backup"; }
    ok "caddy config valid"
    systemctl reload caddy 2>/dev/null || systemctl restart caddy
    ok "caddy reloaded"
  fi
fi

say "next step"
if plan_only; then
  ok "dry run: nothing changed"
elif [ "$DO_RESTART" = 1 ]; then
  "$HERMES_BIN" gateway hermes gateway restart || die "gateway restart failed"
  ok "gateway restarted"
  for _ in $(seq 1 10); do
    ss -ltn 2>/dev/null | grep -q ":$PORT" && break
    sleep 3
  done
else
  echo "   The gateway reads webapp.* and the sign-in provider at startup only."
  echo "   Restart it from a shell outside the agent, or send /restart in a Hermes chat:"
  echo "     $HERMES_BIN gateway hermes gateway restart"
fi

if ! plan_only; then
  say "verification"
  DASH_USER="${LOGIN:-${DASH_USER:-}}" DASH_PASS="${DASH_PASSWORD:-}" \
    bash "$HERE/scripts/verify-desktop.sh" "$DESK_HOST" || true
fi
