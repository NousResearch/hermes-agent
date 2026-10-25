---
name: api-fuzzing
description: Use when fuzzing API paths, parameters, values, and headers.
version: 1.0.0
metadata:
  hermes:
    tags: [api, fuzzing, ffuf, kiterunner]
---

# API Fuzzing

## Overview
Systematically probes an API surface by mutating paths, parameters, values, and headers, then triaging results by status and response size. It pairs content discovery tools with parameter mining and value fuzzing. Requires a baseline response for each fuzzed target to filter noise.

## When to Use
- Route inventory is incomplete and hidden endpoints likely exist.
- You need to discover undocumented parameters on a known endpoint.
- Validating input handling for injection or type confusion.
- **Don't use for:** blind fuzzing without a baseline, which floods results with false positives.

## Procedure

### Step 1: Choose the target surface
Fix a base URL and an auth token, then capture a baseline request.

```bash
export BASE='https://target.example.com/api/v1'
curl -s -o /dev/null -w 'baseline %{http_code} %{size_download}\n' "$BASE/users" \
  -H "Authorization: Bearer $TOKEN"
```

### Step 2: Fuzz paths
Use ffuf and kiterunner with API-oriented wordlists.

```bash
ffuf -u "$BASE/FUZZ" -w /usr/share/seclists/Discovery/Web-Content/api/objects.txt \
  -H "Authorization: Bearer $TOKEN" -mc all -fc 404 -fs 0 -of json -o paths.json
kr scan "$BASE" -w routes-large.kite -H 'Authorization: Bearer TOKEN' -x 20
```

### Step 3: Fuzz parameters and values
Mine hidden parameters, then mutate values for each.

```bash
arjun -u "$BASE/users" -m GET,POST --headers "Authorization: Bearer $TOKEN" -oJ params.json
ffuf -u "$BASE/users?id=FUZZ" -w /usr/share/seclists/Fuzzing/Integers.txt -mc all -fc 404
ffuf -u "$BASE/users" -X POST -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" -d '{"role":"FUZZ"}' -w roles.txt -mc all
```

### Step 4: Fuzz headers, then triage
Mutate headers and rank findings by status and length deltas.

```bash
ffuf -u "$BASE/users" -H "Authorization: Bearer $TOKEN" -H "FUZZ: 1" \
  -w /usr/share/seclists/Discovery/Web-Content/BurpSuite-ParamMiner/Headers.txt -mc all -fc 404
jq -r '.results[] | select(.status!=404) | "\(.status) \(.length) \(.url)"' paths.json | sort -u | head -50
```

## Common Pitfalls
1. No baseline filter — always record the nominal status and size so false positives can be excluded.
2. Hammering without throttling — set `-rate` or `-p` to avoid tripping WAFs and skewing results.
3. Trusting one wordlist — combine API-specific lists with generic and parameter lists.

## Verification Checklist
- [ ] Baseline request and response captured per target.
- [ ] Path fuzzing completed and deduplicated.
- [ ] Parameter discovery run for GET and POST.
- [ ] Value fuzzing performed on discovered parameters.
- [ ] Header fuzzing executed.
- [ ] Results triaged by status, length, and words.
- [ ] Candidate findings verified manually.

## References
- [ffuf](https://github.com/ffuf/ffuf)
- [kiterunner](https://github.com/assetnote/kiterunner)
- [arjun](https://github.com/s0md3v/Arjun)
- [SecLists](https://github.com/danielmiessler/SecLists)

> **Authorization notice:** Use only against systems you own or are explicitly authorized to test.
