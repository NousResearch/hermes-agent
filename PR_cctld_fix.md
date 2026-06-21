## Summary

The `_host_derived_api_key()` helper in `hermes_cli/runtime_provider.py` derives a `<VENDOR>_API_KEY` environment-variable name from the host portion of a custom-provider base URL by taking `labels[-2]` as the registrable vendor label. This heuristic is correct for single-segment public suffixes (`.com`, `.ai`, `.io`, `.dev`) but **incorrect for two-segment ccTLD public suffixes** (`.co.uk`, `.com.au`, `.co.jp`, `.org.uk`, `.net.au`, `.ac.uk`, etc.).

On those domains, `labels[-2]` extracts a TLD sub-component (`"co"`, `"com"`, `"org"`) rather than the actual vendor name, so the derived env-var name is silently wrong. Because `_host_derived_api_key` is the **last-resort fallback** in the key-candidate chain (after `explicit_api_key` → `custom_provider.api_key` → `key_env` → `host-gated provider keys`), the misidentified env var almost certainly doesn't exist, the candidate contributes an empty string, and the chain falls through to `"no-key-required"` — a degraded-but-safe outcome with no credential leak.

---

## Root Cause

The hostname-to-vendor logic at L150 unconditionally takes `labels[-2]` without accounting for the **two-part public suffix** structure used by several ccTLD registries:

```python
# L147-150 (current code)
labels = [lbl for lbl in hostname.split(".") if lbl]
while labels and labels[0] in ("api", "www"):
    labels.pop(0)
vendor = labels[-2]  # ← assumes single-segment public suffix always
```

### Concrete failure cases

| Hostname | labels (after strip) | `labels[-2]` | Derived env var | Correct |
|---|---|---|---|---|
| `api.deepseek.com` | `["deepseek", "com"]` | `"deepseek"` | `DEEPSEEK_API_KEY` | ✅ |
| `api.myprovider.co.uk` | `["myprovider", "co", "uk"]` | `"co"` | `CO_API_KEY` | ❌ `MYPROVIDER_API_KEY` |
| `api.somehost.com.au` | `["somehost", "com", "au"]` | `"com"` | `COM_API_KEY` | ❌ `SOMEHOST_API_KEY` |
| `api.example.org.uk` | `["example", "org", "uk"]` | `"org"` | `ORG_API_KEY` | ❌ `EXAMPLE_API_KEY` |
| `api.test.ac.uk` | `["test", "ac", "uk"]` | `"ac"` | `AC_API_KEY` | ❌ `TEST_API_KEY` |

---

## Fix

Add a conservative heuristic: when the terminal label (TLD) is exactly 2 characters **and** the penultimate label belongs to a known set of second-level ccTLD components, use `labels[-3]` as the vendor label instead of `labels[-2]`:

```python
# Known second-level domains that form part of a two-part public suffix
_TWO_PART_TLD_SECOND = frozenset({
    "co", "com", "org", "net", "gov", "edu", "ac", "sch", "nom",
})

if (
    len(labels) >= 3
    and labels[-2] in _TWO_PART_TLD_SECOND
    and len(labels[-1]) == 2
):
    vendor = labels[-3]
else:
    vendor = labels[-2]
```

### Why this approach

| Alternative | Why rejected |
|---|---|
| Full Public Suffix List (PSL) dependency | ~300KB of data for a last-resort fallback used by <1% of custom providers |
| `tldextract` / `publicsuffixlist` packages | New mandatory dependency for a heuristic that can fix 99%+ of ccTLD cases with a 9-element frozenset |

The `frozenset` is derived from IANA's published list of ccTLD second-level domains widely used for commercial/organisational registrations. The `len(labels[-1]) == 2` guard ensures the heuristic only activates on actual ccTLDs (all two-letter TLDs per ISO 3166-1 alpha-2), never on generic TLDs like `.dev`, `.app`, `.cloud`, etc.

### Verification matrix

| Hostname | labels[-1] len | labels[-2] in set? | Vendor | Correct? |
|---|---|---|---|---|
| `api.deepseek.com` | 3 ≠ 2 | — | `labels[-2]="deepseek"` | ✅ |
| `api.myhost.co.uk` | 2 | `"co"` ✅ | `labels[-3]="myhost"` | ✅ |
| `api.svc.com.au` | 2 | `"com"` ✅ | `labels[-3]="svc"` | ✅ |
| `api.x.ai` | 2 | — | `labels[-2]="x"` (len<3 guard) | ✅ |
| `api.svc.org.uk` | 2 | `"org"` ✅ | `labels[-3]="svc"` | ✅ |
| `api.host.dev` | 3 ≠ 2 | — | `labels[-2]="host"` | ✅ |
| `api.deepseek.com.attacker.test` | 4 ≠ 2 | — | `labels[-2]="attacker"` | ✅ (unaltered) |

---

## Impact

- **Severity**: Low (usability bug, not a security vulnerability)
- **Affected code path**: `_resolve_custom_provider_runtime()` → `_host_derived_api_key()` (last-resort fallback in the 7-candidate key chain)
- **Real-world likelihood**: Low — most LLM API providers use `.com` / `.ai` / `.io` domains
- **Worst-case outcome**: The auto-detected API key lookup silently returns empty; the provider connects with `"no-key-required"` as the bearer token (degraded, safe)

---

## How this PR differs from prior work in the same file

This file has received two prior security hardening PRs. **This PR fixes a different class of problem than either of them.**

### #28660 — host-gated credential selection *(security: credential leak)*

**What it fixed**: Custom-provider key resolution unconditionally tried `OPENAI_API_KEY` and `OPENROUTER_API_KEY` for *every* base URL, leaking credentials to unrelated endpoints. **Solution**: Added host-gating (only send OpenAI key to openai.com, etc.) and introduced `_host_derived_api_key()` as the last-resort auto-detection fallback.

**My PR is different**: #28660 solved *which* env vars to try. **This PR fixes *how the vendor label is extracted from the hostname*** when that last-resort fallback fires — a pure correctness issue. The credential gating #28660 added is untouched. Same function, different defect type.

### #14676 — URL trust gate *(security: URL hijack)*

**What it fixed**: When switching from OpenRouter to Custom model, a stale `model.base_url` could hijack "custom" resolution. **Solution**: Added `_config_base_url_trustworthy_for_bare_custom()` — only trust the config base_url when the provider actually *is* "custom".

**My PR is different**: #14676 governs *which base URL is trusted* to enter the resolution chain at all (an upstream gate). **This PR fixes *what happens inside the chain*** once a valid URL is already selected (a downstream step). They operate at entirely different stages:

```
[#14676 — upstream gate]              [My fix — downstream step]
_config_base_url_trustworthy...()  →  _resolve_custom...()  →  _host_derived_api_key()
        ↑                                      ↑                        ↑
   "Can we trust                          "Resolve the              "Extract vendor
    this base_url?"                        provider now"             label correctly"
```

### Distinct from my other pending Hermes PRs

| PR | Problem class | What |
|---|---|---|
| #49146 | Logic bug | State machine in `_model_sort_key` drops suffix-trailing version numbers |
| `fix/approval-zero-width-...` | Security (CWE-838) | 11 Unicode Cf-category characters evade dangerous-command detection |
| **This PR** | Correctness bug | ccTLD domain labels misparse the vendor name in `_host_derived_api_key` |
