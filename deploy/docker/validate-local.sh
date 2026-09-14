#!/usr/bin/env bash
# Local validation of the NOVA control-plane image. Run this before pushing anything to a
# registry: it exercises the built image rather than the source tree, which is the only way
# to catch the class of defect that only exists once the code is in a container — a missing
# dependency, a signal nobody handles, a module root copied half-way.
#
# It starts two tenants side by side on the real image, drives the real control API, and
# checks the properties that would be a security incident if they regressed: cross-tenant
# reads, unauthenticated access, role separation, cross-site writes, credential material in
# the image. Every check names its expected value, so a change of behaviour shows up as a
# diff rather than as a script that quietly stops testing anything.
#
# Usage:  deploy/docker/validate-local.sh nova-control-plane:<tag>
#
# The tokens below are throwaway fixtures for a container this script creates and destroys.
# They authenticate nothing outside this run.

set -uo pipefail
IMG="${1:?usage: validate-local.sh <image:tag>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="$(mktemp -d -t nova-validate-XXXXXX)"
trap 'docker rm -f nova-a nova-b >/dev/null 2>&1; rm -rf "$ROOT"' EXIT
AT=tok-a-admin-0123456789abcdef; AV=tok-a-viewer-0123456789abcdef; BT=tok-b-admin-0123456789abcdef
pass=0; fail=0
ck() { # ck <label> <expected> <actual>
  if [ "$2" = "$3" ]; then printf '  PASS  %-58s %s\n' "$1" "$3"; pass=$((pass+1));
  else printf '  FAIL  %-58s expected %s got %s\n' "$1" "$2" "$3"; fail=$((fail+1)); fi
}
code() { curl -s -o /tmp/vb -w '%{http_code}' "$@"; }

echo "### 0. clean state"
docker rm -f nova-a nova-b >/dev/null 2>&1
mkdir -p "$ROOT"
for t in a b; do
  mkdir -p "$ROOT/tenant-$t/bundle" "$ROOT/tenant-$t/home"
  cp -r "$REPO_ROOT"/nova/examples/acme/. "$ROOT/tenant-$t/bundle/"
  sed -i "s/^tenant_id: acme/tenant_id: tenant-$t/" "$ROOT/tenant-$t/bundle/organization.yaml"
  python3 - "$ROOT/tenant-$t/home/control-principals.yaml" "$t" <<'PY'
import hashlib, pathlib, sys
p, t = pathlib.Path(sys.argv[1]), sys.argv[2]
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text("principals:\n"
  f"  - name: admin-{t}\n    role: admin\n    token_sha256: {h(f'tok-{t}-admin-0123456789abcdef')}\n"
  f"  - name: viewer-{t}\n    role: viewer\n    token_sha256: {h(f'tok-{t}-viewer-0123456789abcdef')}\n")
PY
done
chmod -R a+rwX "$ROOT"

echo "### 1. NOVA starts in the container"
ck "nova --version"        "nova 0.1.0"  "$(docker run --rm --entrypoint sh "$IMG" -c 'python -m nova --version' 2>&1)"
ck "runs as a non-root user" "nova"      "$(docker run --rm --entrypoint sh "$IMG" -c 'id -un' 2>&1)"

echo "### 2. Hermes starts in the container"
out=$(docker run --rm --entrypoint sh -e HERMES_HOME=/tmp/h "$IMG" -c 'mkdir -p /tmp/h && python - <<PY 2>/dev/null
from cron import jobs
from hermes_cli import kanban_db_connect as kc, kanban_db as kb, kanban_tenant as kt
conn = kc.connect()
a = kb.create_task(conn, title="alpha", tenant="tenant-a")
b = kb.create_task(conn, title="beta",  tenant="tenant-b")
tt = lambda rows: sorted(r.title for r in rows)
with kt.use("tenant-a"): x = (tt(kb.list_tasks(conn)), kb.get_task(conn, b))
with kt.use("tenant-b"): y = (tt(kb.list_tasks(conn)), kb.get_task(conn, a))
print("OK" if x==(["alpha"],None) and y==(["beta"],None) else f"BAD {x} {y}")
print("cron", hasattr(jobs,"create_job"))
PY')
ck "kanban store + tenant scoping in-container" "OK" "$(echo "$out" | head -1)"
ck "cron.jobs write API present"                "cron True" "$(echo "$out" | sed -n 2p)"

echo "### 3. apply materialises agents onto the volume"
ap=$(docker run --rm -v "$ROOT/tenant-a:/var/lib/nova" "$IMG" apply 2>&1 | head -1)
ck "tenant-a apply" "tenant=tenant-a runtime=hermes created=3 changed=0 unchanged=0" "$ap"
ap=$(docker run --rm -v "$ROOT/tenant-b:/var/lib/nova" "$IMG" apply 2>&1 | head -1)
ck "tenant-b apply" "tenant=tenant-b runtime=hermes created=3 changed=0 unchanged=0" "$ap"

echo "### 4. exposure guard"
# A home with NO principals file: the anonymous-exposure refusal.
rm -rf "$ROOT/bare"; mkdir -p "$ROOT/bare/home"; cp -r "$ROOT/tenant-a/bundle" "$ROOT/bare/bundle"; chmod -R a+rwX "$ROOT/bare"
g=$(docker run --rm -e NOVA_BIND_HOST=0.0.0.0 -v "$ROOT/bare:/var/lib/nova" "$IMG" 2>&1)
case "$g" in *"refusing to bind 0.0.0.0 with no principals file"*) r=refused;; *) r="started or wrong error";; esac
ck "0.0.0.0 with no principals file is refused" "refused" "$r"
# The same home WITH principals but no TLS statement: the cleartext refusal.
g=$(docker run --rm -e NOVA_BIND_HOST=0.0.0.0 -v "$ROOT/tenant-a:/var/lib/nova" "$IMG" 2>&1)
case "$g" in *"refusing to bind 0.0.0.0 without TLS"*) r=refused;; *) r="started or wrong error";; esac
ck "0.0.0.0 with principals but no TLS is refused" "refused" "$r"

echo "### 5. two tenants served concurrently"
docker run -d --name nova-a -p 18787:8787 -e NOVA_BIND_HOST=0.0.0.0 -e NOVA_BEHIND_TLS_PROXY=1 -v "$ROOT/tenant-a:/var/lib/nova" "$IMG" >/dev/null
docker run -d --name nova-b -p 18788:8787 -e NOVA_BIND_HOST=0.0.0.0 -e NOVA_BEHIND_TLS_PROXY=1 -v "$ROOT/tenant-b:/var/lib/nova" "$IMG" >/dev/null
for i in $(seq 1 40); do
  [ "$(docker inspect --format '{{.State.Health.Status}}' nova-a 2>/dev/null)" = healthy ] &&
  [ "$(docker inspect --format '{{.State.Health.Status}}' nova-b 2>/dev/null)" = healthy ] && break; sleep 2; done
ck "container A reports healthy" "healthy" "$(docker inspect --format '{{.State.Health.Status}}' nova-a)"
ck "container B reports healthy" "healthy" "$(docker inspect --format '{{.State.Health.Status}}' nova-b)"
A=http://127.0.0.1:18787/platform/v1; B=http://127.0.0.1:18788/platform/v1

echo "### 6. authentication and RBAC"
ck "no token -> /agents"                 401 "$(code $A/agents)"
ck "garbage token -> /agents"            401 "$(code -H 'Authorization: Bearer nope' $A/agents)"
ck "tenant-B token -> tenant-A /agents"  401 "$(code -H "Authorization: Bearer $BT" $A/agents)"
ck "tenant-A token -> tenant-B /agents"  401 "$(code -H "Authorization: Bearer $AT" $B/agents)"
ck "viewer -> /agents"                   200 "$(code -H "Authorization: Bearer $AV" $A/agents)"
ck "viewer -> /policy"                   403 "$(code -H "Authorization: Bearer $AV" $A/policy)"
ck "viewer -> /decisions"                403 "$(code -H "Authorization: Bearer $AV" $A/decisions)"
ck "viewer -> /budget"                   403 "$(code -H "Authorization: Bearer $AV" $A/budget)"
ck "admin  -> /policy"                   200 "$(code -H "Authorization: Bearer $AT" $A/policy)"

echo "### 7. write surface"
J='Content-Type: application/json'
SPEC='{"id":"nightly-check","title":"Nightly check","agent":"operations","schedule":"every day at 02:00","objective":"Summarise yesterday inventory exceptions.","enabled":true,"permissions":["read_inventory"],"reason":"ops requested"}'
ck "viewer POST /automations (create)"   403 "$(code -X POST -H "Authorization: Bearer $AV" -H "$J" -d "$SPEC" $A/automations)"
ck "viewer POST /automations/x/pause"    403 "$(code -X POST -H "Authorization: Bearer $AV" -H "$J" -d '{}' $A/automations/x/pause)"
ck "viewer POST /automations/x/delete"   403 "$(code -X POST -H "Authorization: Bearer $AV" -H "$J" -d '{}' $A/automations/x/delete)"
ck "no token POST /automations"          401 "$(code -X POST -H "$J" -d "$SPEC" $A/automations)"
ck "admin form-encoded write (CSRF)"     415 "$(code -X POST -H "Authorization: Bearer $AT" -H 'Content-Type: application/x-www-form-urlencoded' -d 'a=1' $A/automations)"
ck "admin cross-Origin write"            403 "$(code -X POST -H "Authorization: Bearer $AT" -H "$J" -H 'Origin: https://evil.example' -d "$SPEC" $A/automations)"
ck "admin POST /automations (create)"    201 "$(code -X POST -H "Authorization: Bearer $AT" -H "$J" -d "$SPEC" $A/automations)"

echo "### 8. tenant isolation of the created automation"
ck "tenant-A sees 1 automation" "1" "$(curl -s -H "Authorization: Bearer $AT" $A/automations | python3 -c 'import json,sys;print(json.load(sys.stdin)["counts"]["total"])')"
ck "tenant-B sees 0 automations" "0" "$(curl -s -H "Authorization: Bearer $BT" $B/automations | python3 -c 'import json,sys;print(json.load(sys.stdin)["counts"]["total"])')"
ck "objective text withheld from viewer" "0" "$(curl -s -H "Authorization: Bearer $AV" $A/automations | grep -c 'inventory exceptions')"
ck "tenant-B has no NOVA automation registry" "absent" "$([ -e "$ROOT/tenant-b/home/nova/automations.json" ] && echo present || echo absent)"

echo "### 9. audit: intent -> committed, with the human actor"
ck "last two audit phases" "intent committed" "$(tail -2 "$ROOT/tenant-a/home/nova/audit.jsonl" | python3 -c 'import sys,json;print(" ".join(json.loads(l)["phase"] for l in sys.stdin))')"
ck "audit actor is the authenticated principal" "admin-a" "$(tail -1 "$ROOT/tenant-a/home/nova/audit.jsonl" | python3 -c 'import sys,json;print(json.load(sys.stdin)["actor"])')"
ck "audit tenant" "tenant-a" "$(tail -1 "$ROOT/tenant-a/home/nova/audit.jsonl" | python3 -c 'import sys,json;print(json.load(sys.stdin)["tenant_id"])')"

echo "### 10. Control Center"
hdr=$(curl -s -D- -o /tmp/idx -H "Authorization: Bearer $AT" http://127.0.0.1:18787/)
ck "GET / serves the dashboard" "200" "$(code -H "Authorization: Bearer $AT" http://127.0.0.1:18787/)"
ck "CSP header unchanged" "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'" "$(printf '%s' "$hdr" | tr -d '\r' | sed -n 's/^Content-Security-Policy: //p')"
ck "X-Frame-Options" "DENY" "$(printf '%s' "$hdr" | tr -d '\r' | sed -n 's/^X-Frame-Options: //p')"
n=0; bad=0
for a in $(grep -oE '(src|href)="/assets/[^"]+"' /tmp/idx | sed 's/.*="//;s/"//' | sort -u); do
  n=$((n+1)); [ "$(code -H "Authorization: Bearer $AT" "http://127.0.0.1:18787$a")" = 200 ] || bad=$((bad+1)); done
ck "all $n referenced assets resolve" "0" "$bad"

echo "### 11. lifecycle"
before=$(curl -s -H "Authorization: Bearer $AT" $A/automations | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["automations"][0]["state"], d["automations"][0]["next_run_at"])')
t0=$(date +%s); docker stop -t 30 nova-a >/dev/null; t1=$(date +%s)
ck "graceful stop exit code" "0" "$(docker inspect --format '{{.State.ExitCode}}' nova-a)"
ck "graceful stop under 5s" "yes" "$([ $((t1-t0)) -lt 5 ] && echo yes || echo "no ($((t1-t0))s)")"
docker start nova-a >/dev/null
for i in $(seq 1 40); do [ "$(docker inspect --format '{{.State.Health.Status}}' nova-a)" = healthy ] && break; sleep 2; done
after=$(curl -s -H "Authorization: Bearer $AT" $A/automations | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["automations"][0]["state"], d["automations"][0]["next_run_at"])')
ck "automation survives a restart unchanged" "$before" "$after"
ck "provenance survives a restart" "1" "$(curl -s -H "Authorization: Bearer $AT" $A/automations | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["governance"]))')"

echo "### 12. no secrets in the image"
ck "no .env / key / credential files"  "0" "$(docker run --rm --entrypoint sh "$IMG" -c 'find / -xdev \( -name ".env" -o -name "*.env" -o -name "id_rsa*" -o -name ".netrc" -o -name ".git-credentials" -o -name "*.p12" -o -name "*.pfx" \) 2>/dev/null | wc -l')"
ck "no /run/secrets left in the image" "absent" "$(docker run --rm --entrypoint sh "$IMG" -c '[ -e /run/secrets ] && echo present || echo absent')"
ck "no .git directory"                 "absent" "$(docker run --rm --entrypoint sh "$IMG" -c '[ -e /opt/nova/.git ] && echo present || echo absent')"
ck "no tenant bundle baked in"         "0" "$(docker run --rm --entrypoint sh "$IMG" -c 'ls -A /var/lib/nova 2>/dev/null | wc -l')"
ck "no tfvars / tfstate"               "0" "$(docker run --rm --entrypoint sh "$IMG" -c 'find / -xdev \( -name "*.tfvars" -o -name "*.tfstate" \) 2>/dev/null | wc -l')"
# AWS's own documented placeholder (AKIAIOSFODNN7EXAMPLE) appears in a comment explaining
# why a shape check alone is not enough. Excluded by name; any OTHER key-shaped string fails.
ck "no AWS access keys in shipped code" "0" "$(docker run --rm --entrypoint sh "$IMG" -c 'grep -rEoh "(AKIA|ASIA)[0-9A-Z]{16}" /opt/nova 2>/dev/null | grep -v "^AKIAIOSFODNN7EXAMPLE$" | wc -l')"
ck "build-time CA never entered a layer" "0" "$(docker run --rm --entrypoint sh "$IMG" -c 'grep -rl "ccr-agent-proxy" / --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev 2>/dev/null | wc -l')"
# GPG_KEY in the upstream python base is a public key *fingerprint* used to verify the
# CPython tarball, not a credential. Excluded by exact name; anything else fails.
ck "image config carries no secret env" "0" "$(docker image inspect "$IMG" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -Ei "secret|password|token|_key=" | grep -vc "^GPG_KEY=" )"

echo
echo "=== $pass passed, $fail failed ==="
docker rm -f nova-a nova-b >/dev/null 2>&1
exit $((fail > 0))
