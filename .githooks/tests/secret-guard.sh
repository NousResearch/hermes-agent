#!/bin/sh
# .githooks/tests/secret-guard.sh — tests for the secret-guard FILENAME blocklist.
# Throwaway git repos under a temp dir; no network, no effect on this repo.
# Exit 0 = all pass.  Companion to run.sh (which tests content-scan).
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
GUARD="$HERE/../secret-guard"
TMP=$(mktemp -d 2>/dev/null || { d=/tmp/nfsg-tests.$$; mkdir -p "$d"; printf '%s' "$d"; })
trap 'rm -rf "$TMP"' EXIT

pass=0 fail=0
ok()  { pass=$((pass+1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad() { fail=$((fail+1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

newrepo() {
    d="$TMP/$1"; mkdir -p "$d"
    ( cd "$d" && git init -q && git config user.email t@t && git config user.name t && git config core.autocrlf false )
    printf '%s' "$d"
}

# stage_check <repo> <relpath> ; then assert on the exit of `secret-guard --staged`
stage_file() {
    _r=$1; _p=$2
    ( cd "$_r" && mkdir -p "$(dirname "$_p")" 2>/dev/null; printf 'x\n' > "$_p" && git add -f -- "$_p" )
}

# name, path, expect: "block" or "allow"
one_case() {
    _name=$1; _path=$2; _want=$3
    _r=$(newrepo "case_$(printf '%s' "$_name" | tr -c 'A-Za-z0-9' _)")
    stage_file "$_r" "$_path"
    if ( cd "$_r" && "$GUARD" --staged ) >/dev/null 2>&1; then _got=allow; else _got=block; fi
    if [ "$_got" = "$_want" ]; then ok "$_name ($_path -> $_want)"; else bad "$_name ($_path: want $_want, got $_got)"; fi
}

# --- blocked: real credential material ------------------------------------
one_case ".env"                    ".env"                          block
one_case ".env.production"         ".env.production"               block
one_case "nested .env"             "deploy/config/.env.staging"    block
one_case "id_rsa"                  "keys/id_rsa"                    block
one_case "id_ed25519"             "id_ed25519"                     block
one_case "pkcs12"                  "certs/client.p12"              block
one_case "pfx"                     "app.pfx"                        block
one_case "java keystore"           "config/app.jks"               block
one_case "keystore ext"           "release.keystore"              block
one_case "credentials.json"        "credentials.json"              block
one_case "gcp service account"     "svc/my-project-service-account.json" block
one_case "gcloud key"             "gcloud-service-key.json"       block
one_case ".netrc"                  ".netrc"                        block
one_case ".pgpass"                 "home/.pgpass"                  block
one_case ".htpasswd"              "nginx/.htpasswd"               block
one_case "bare private key"        "secrets/server.key"           block
one_case "pem private key"         "deploy/privkey.pem"           block
one_case "pk8"                     "signing/app.pk8"              block

# --- allowed: not secrets, or deliberately safe -------------------------
one_case ".env.example"            ".env.example"                  allow
one_case ".env.sample"             "config/.env.sample"           allow
one_case ".envrc"                  ".envrc"                        allow
one_case "public ssh key"          "keys/id_rsa.pub"              allow
one_case "CA bundle"               "vendor/certifi/cacert.pem"    allow
one_case "fullchain bundle"        "certs/fullchain.pem"          allow
one_case "pem test fixture"        "tests/fixtures/server.pem"    allow
one_case "key test fixture"        "src/pkg/testdata/tls.key"     allow
one_case "top-level tests pem"     "tests/data/localhost.pem"     allow
one_case "ordinary json"           "package/credentials.schema.json" allow
one_case "source file"             "app/keychain.py"              allow
one_case "cert (not key)"          "certs/server.crt"             allow

# --- allowlist escape hatch --------------------------------------------
r=$(newrepo allowlisted)
# inject a one-line allowlist entry into a COPY of the guard, exercise it
sed 's#^ALLOWLIST="#ALLOWLIST="\ntest/fixtures/legit-but-flagged.key#' "$GUARD" > "$r/guard-with-allow"
chmod +x "$r/guard-with-allow"
stage_file "$r" "test/fixtures/legit-but-flagged.key"
if ( cd "$r" && sh "./guard-with-allow" --staged ) >/dev/null 2>&1; then
    ok "explicit ALLOWLIST path passes despite matching a rule"
else
    bad "explicit ALLOWLIST path passes despite matching a rule"
fi

printf '\n%s passed, %s failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
