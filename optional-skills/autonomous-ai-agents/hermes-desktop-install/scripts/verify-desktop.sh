#!/usr/bin/env bash
# Verify a live browser-hosted Hermes Desktop deployment.
#
# Usage: verify-desktop.sh <DESK_HOST> [USER] [PASS]
#   USER/PASS default to $DASH_USER / $DASH_PASS. Without credentials the
#   password checks report SKIP and the exit code stays 0.
#
# Checks (each independent):
#   1 unauthenticated /           -> 30x to /login
#   2 unauthenticated /api/health -> 401
#   3 /login renders a sign-in form
#   4 wrong password              -> 401
#   5 correct password            -> 200 + session cookie
#   6 authed /api/health          -> 200 with a version
#   7 gateway serves it: webapp --status + PID on the loopback port
#   8 TLS certificate matches the hostname and is valid
#   9 renderer artifact exists
set -uo pipefail

DESK=${1:?usage: verify-desktop.sh <DESK_HOST> [USER] [PASS]}
USER_NAME=${2:-${DASH_USER:-}}
PASS=${3:-${DASH_PASS:-}}
PORT=${WEBAPP_PORT:-9120}
HERMES_HOME=${HERMES_HOME:-$HOME/.hermes}
JAR=$(mktemp); trap 'rm -f "$JAR"' EXIT

fails=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; fails=$((fails+1)); }
skip() { echo "SKIP  $*"; }

code() { curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$@"; }
redir() { curl -s -o /dev/null -w '%{redirect_url}' --max-time 20 "$@"; }

echo "== gate"
c=$(code "https://$DESK/"); r=$(redir "https://$DESK/")
case "$c" in
  30[1237]) case "$r" in *login*) pass "unauthenticated / -> $c $r";;
             *) fail "/ -> $c but redirect '$r' is not /login";; esac ;;
  000) fail "/ -> 000 (no response: cert pending, DNS, or proxy down)" ;;
  *) fail "/ unauthenticated -> $c (expected 30x to /login)" ;;
esac

# /api/health and /api/status are public banners by design; use a gated route.
c=$(code "https://$DESK/api/files")
[ "$c" = "401" ] && pass "unauthenticated /api/files -> 401" || fail "unauth /api/files -> $c (expected 401)"

hi=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://$DESK/login")
body=$(curl -s --max-time 20 "https://$DESK/login")
if [ "$hi" = "200" ] && printf '%s' "$body" | grep -qi 'sign in'; then
  pass "/login renders a sign-in form (200)"
else
  fail "/login -> $hi, sign-in form not detected"
fi

echo "== password provider"
if [ -n "$USER_NAME" ] && [ -n "$PASS" ]; then
  json() { printf '{"provider":"basic","username":"%s","password":"%s","next":"/"}' "$USER_NAME" "$1"; }
  ok=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST \
        -H 'Content-Type: application/json' -d "$(json "$PASS")" "https://$DESK/auth/password-login")
  bad=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST \
        -H 'Content-Type: application/json' -d "$(json 'wrong-on-purpose')" "https://$DESK/auth/password-login")
  [ "$ok" = "200" ] && pass "correct password -> 200" || fail "correct password -> $ok (expected 200)"
  [ "$bad" = "401" ] && pass "wrong password -> 401" || fail "wrong password -> $bad (expected 401)"
  if [ "$ok" = "200" ]; then
    curl -s -c "$JAR" -o /dev/null --max-time 20 -X POST -H 'Content-Type: application/json' \
      -d "$(json "$PASS")" "https://$DESK/auth/password-login"
    hb=$(curl -s -b "$JAR" -w '\n%{http_code}' --max-time 20 "https://$DESK/api/health")
    hcode=$(printf '%s' "$hb" | tail -1)
    if [ "$hcode" = "200" ] && printf '%s' "$hb" | grep -q '"version"'; then
      pass "session cookie accepted: /api/health 200"
    else
      fail "session cookie rejected: /api/health -> $hcode"
    fi
  fi
else
  skip "password checks (pass USER PASS or set DASH_USER/DASH_PASS)"
fi

echo "== gateway listener"
if command -v hermes >/dev/null 2>&1; then
  st=$(hermes webapp --status 2>&1 | tail -2)
  case "$st" in
    *Serving*) pass "webapp --status: $st" ;;
    *) fail "webapp --status: $st" ;;
  esac
else
  skip "hermes not on PATH"
fi
listen_pid=$(ss -ltnp 2>/dev/null | awk -v p="127.0.0.1:$PORT" '$4 ~ p' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
main_pid=$(systemctl --user show hermes-gateway.service -p MainPID --value 2>/dev/null)
[ -n "${main_pid:-}" ] && [ "$main_pid" != "0" ] || main_pid=$(systemctl show hermes-gateway.service -p MainPID --value 2>/dev/null)
if [ -z "${listen_pid:-}" ]; then
  fail "nothing listening on 127.0.0.1:$PORT"
elif [ -n "${main_pid:-}" ] && [ "$listen_pid" = "$main_pid" ]; then
  pass "gateway (MainPID $main_pid) owns 127.0.0.1:$PORT"
else
  fail "127.0.0.1:$PORT held by pid $listen_pid, gateway MainPID=${main_pid:-unknown}"
fi

echo "== tls"
end=$(echo | openssl s_client -connect "$DESK:443" -servername "$DESK" 2>/dev/null \
        | openssl x509 -noout -subject -enddate 2>/dev/null)
subj=$(printf '%s' "$end" | sed -n 's/^subject=//p')
notafter=$(printf '%s' "$end" | sed -n 's/^notAfter=//p')
case "$subj" in
  *"CN = $DESK"*|*"CN=$DESK"*) pass "certificate subject is $DESK" ;;
  *) fail "certificate subject does not name $DESK (${subj:-no cert})" ;;
esac
if [ -n "$notafter" ]; then
  exp=$(date -d "$notafter" +%s 2>/dev/null || echo 0)
  now=$(date +%s)
  if [ "$exp" -gt "$now" ]; then pass "certificate valid until $notafter"; else fail "certificate expired: $notafter"; fi
else
  fail "no certificate end date"
fi

echo "== renderer"
bundle="$HERMES_HOME/hermes-agent/apps/desktop/dist-webapp/index.html"
[ -f "$bundle" ] && pass "bundle present: $bundle" || fail "bundle missing: $bundle (run: hermes webapp --build-only)"

echo
echo "failures: $fails"
exit $(( fails > 0 ? 1 : 0 ))
