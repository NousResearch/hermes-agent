# Vulnerability Taxonomy

Two classification systems used during analysis. Both come from Shannon
(concepts only; rewritten here). Both exist to make the question
"is this exploitable?" mechanical instead of vibes-based.

## Injection: Slot Types

Every injection sink has a **slot type** — the lexical position the
attacker payload lands in. Each slot type has a small set of
**required defenses**. A mismatch is a vulnerability. The same defense
applied to the wrong slot is also a vulnerability.

| Slot | Example | Required defense |
|------|---------|------------------|
| `SQL-val` | `SELECT * FROM u WHERE id = :v` | Parameterized binding |
| `SQL-ident` | `SELECT * FROM ${table}` | Allowlist on identifier values |
| `SQL-keyword` | `ORDER BY ${col} ${dir}` | Allowlist on column AND direction |
| `CMD-argument` | `subprocess.run(["ls", v])` | argv list (never shell=True) |
| `CMD-shell` | `os.system("ls " + v)` | DON'T — refactor to argv list |
| `PATH-segment` | `open("/data/" + v)` | Normalize + allowlist + base-relative check |
| `URL-host` | redirect to `https://${v}/x` | Allowlist of acceptable hosts |
| `URL-fetch` | `requests.get(v)` | Allowlist + block private/metadata IPs (SSRF) |
| `TEMPLATE-string` | `Template("Hello {{ v }}")` | Autoescape ON, no user-controlled template syntax |
| `DESERIALIZE-pickle` | `pickle.loads(v)` | DON'T — use JSON / msgpack |
| `DESERIALIZE-yaml` | `yaml.load(v)` | `yaml.safe_load`, never `yaml.load` |
| `XPATH-expr` | `tree.xpath("//u[@id='" + v + "']")` | Parameterized XPath or escape |
| `LDAP-filter` | `(uid=${v})` | LDAP filter escaping |
| `REGEX-pattern` | `re.search(v, text)` | Don't take pattern from user (ReDoS too) |
| `LOG-record` | `log.info("got " + v)` | Encode CR/LF/control chars before logging |
| `EMAIL-header` | `Subject: ${v}` | Reject CR/LF |
| `HTTP-header` | `Set-Cookie: ${v}` | Reject CR/LF (response splitting) |

When you classify a finding:
1. Identify the slot type
2. Identify the actual defense in the code (if you have source)
3. If defense doesn't match the required-defense set: vulnerable

## XSS: Render Contexts

XSS exploitability depends on **where** in the HTML/JS the value lands.
Encoding for one context doesn't protect another.

| Context | Example | Required encoding |
|---------|---------|-------------------|
| `HTML_BODY` | `<div>{{ v }}</div>` | HTML entity encode `<>&"'` |
| `HTML_ATTR_QUOTED` | `<a href="{{ v }}">` | HTML attr encode |
| `HTML_ATTR_UNQUOTED` | `<a href={{ v }}>` | Almost impossible to safely encode; quote the attr |
| `URL_ATTR` (href/src) | `<a href="{{ v }}">` | Validate scheme allowlist + attr encode |
| `JAVASCRIPT_STRING` | `<script>var x = "{{ v }}";</script>` | JS string escape + ensure quote consistency |
| `JAVASCRIPT_BLOCK` | `<script>{{ v }}</script>` | DON'T — refactor; no safe encoding |
| `CSS_VALUE` | `<style>color: {{ v }};</style>` | CSS encode + allowlist scheme/format |
| `CSS_BLOCK` | `<style>{{ v }}</style>` | DON'T — refactor |
| `JSON_RESPONSE` (consumed by JS) | `JSON.parse(response)` | JSON encode + correct content-type header |
| `EVENT_HANDLER` | `<div onclick="{{ v }}">` | JS string escape *inside* HTML attr encode |
| `URL_PATH` (router-driven) | route param echoed unencoded | URL-encode + HTML-encode |
| `DOM_INNERHTML` | `el.innerHTML = v` (DOM XSS) | Use `textContent` instead, or DOMPurify |
| `DOM_DOC_WRITE` | `document.write(v)` | DON'T — refactor |

When you classify:
1. Identify the render context where user input lands
2. Identify the encoding applied
3. Mismatch = vulnerable. Even "HTML encoded" output in
   `JAVASCRIPT_STRING` is exploitable (`</script><script>` evasion).

## OWASP Top 10 (2021) Mapping

For reporting:

| OWASP | Slot/context covered |
|-------|----------------------|
| A01 Broken Access Control | authz class (IDOR, vertical/horizontal) |
| A02 Cryptographic Failures | infra class (weak TLS, plaintext storage) |
| A03 Injection | injection class (all slot types except deserialize) |
| A04 Insecure Design | reported in findings narrative |
| A05 Security Misconfiguration | infra class |
| A06 Vulnerable Components | infra class (whatweb output) |
| A07 Auth Failures | auth class |
| A08 Software/Data Integrity | DESERIALIZE-* slots, also supply chain |
| A09 Logging/Monitoring | infra class (out of scope for active testing) |
| A10 SSRF | ssrf class |
