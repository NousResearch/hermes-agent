#!/usr/bin/env bash
# test_verify_done.sh — 4-case acceptance test for verify_done.sh
#
# Each scenario must pass. Exits 0 only if all pass.

set -euo pipefail

HERE="$(cd "$(dirname "$0")"/.. && pwd)"
SCRIPT="$HERE/scripts/verify_done.sh"

if [[ ! -x "$SCRIPT" ]]; then
    echo "FATAL: $SCRIPT not executable"; exit 2
fi

pass=0
fail=0

run_case() {
    local name="$1"; shift
    local expect_exit="$1"; shift
    local input="$1"; shift

    local actual
    echo "$input" | bash "$SCRIPT" - > /tmp/verify_out 2>&1 && actual=0 || actual=$?

    if [[ "$actual" == "$expect_exit" ]]; then
        echo "✓ $name"
        pass=$((pass + 1))
    else
        echo "✗ $name  (expected exit $expect_exit, got $actual)"
        echo "  --- output ---"
        cat /tmp/verify_out | sed 's/^/  | /'
        echo "  --- end ---"
        fail=$((fail + 1))
    fi
}

# Case 1: bare claim, no proof → expect 1
run_case "case 1: bare claim → flag" 1 "Listo, ya quedo todo configurado."

# Case 2: claim with paired $ command → expect 0
input2=$(cat <<'EOF'
Listo, ya quedo todo.

$ docker ps
CONTAINER ID   IMAGE         STATUS
abc123         vaultwarden   Up 5 minutes
EOF
)
run_case "case 2: claim with proof → pass" 0 "$input2"

# Case 3: no claims at all → expect 0
run_case "case 3: no claims → pass" 0 "Voy a verificar el estado del servidor."

# Case 4: one good claim + one bare claim → expect 1 (flags only the bare one)
input4=$(cat <<'EOF'
Listo, configure la migración.

$ kubectl get pods
NAME          READY   STATUS
web-abc       1/1     Running

Tambien arregle el ingress.
EOF
)
run_case "case 4: one bare claim mixed → flag it" 1 "$input4"

# Case 5: claim with adjacent exit code → expect 0
input5=$(cat <<'EOF'
Listo, build OK.

$ ./build.sh
exit code: 0
EOF
)
run_case "case 5: exit code as proof → pass" 0 "$input5"

# Case 6: claim with HTTP status → expect 0
input6=$(cat <<'EOF'
Repo created and ready.

HTTP/1.1 200 OK
EOF
)
run_case "case 6: HTTP status as proof → pass" 0 "$input6"

# Case 7: empty message → expect 0
run_case "case 7: empty message → pass" 0 ""

# Case 8: stdin interactive (-h) → expect 0
if bash "$SCRIPT" --help > /dev/null 2>&1; then
    echo "✓ case 8: --help works"
    pass=$((pass + 1))
else
    echo "✗ case 8: --help failed"
    fail=$((fail + 1))
fi

echo
echo "=== $pass passed, $fail failed ==="
[[ "$fail" -eq 0 ]] || exit 1
exit 0
