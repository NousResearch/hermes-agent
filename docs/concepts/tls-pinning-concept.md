# TLS Pinning — Concept Document

**Status:** Concept only — will not implement
**Date:** 2026-05-10
**Author:** Nikolay Gusev

---

## What

TLS pinning (aka certificate pinning) hardcodes the expected TLS certificate or public key of an LLM provider into the agent, so even if a CA issues a fraudulent cert or a corporate proxy rewrites traffic, the agent refuses to connect.

## Attack Scenario

```text
User → [Corporate Firewall / ISP / Malicious Proxy] → LLM Provider (OpenRouter)
```

The attacker sits **between the user and the provider** and rewrites LLM responses. This is the same technique the author used as a sysadmin to redirect online game traffic to hh.ru.

### Plausible attack

1. Corporate firewall detects `api.openrouter.ai` traffic
2. Rewrites all responses to: *"Write the code yourself, monkey."*
3. Agent gets poisoned response → executes wrong code / refuses work
4. User has no way to detect the MITM without TLS pinning

### Why it's unlikely

- Requires **active MITM** on the path user→provider
- User must be behind a transparent proxy (corporate VPN, state-level DPI, compromised ISP)
- Modern LLM providers use strong TLS + HSTS
- Attacker must have a valid certificate for the domain or user must accept a custom CA

## Effort Estimate

| Component | Effort | Complexity |
|-----------|--------|------------|
| OpenAI SDK patch | ~1h | MED — override `httpx.Client` |
| Other providers (Anthropic, Google, etc.) | ~2h | MED — each SDK does TLS differently |
| User config for fingerprints | ~0.5h | LOW — YAML section in config |
| Cert rotation handling | ~1h | HIGH — provider certs change periodically |
| **Total** | **~4.5h** | |

## Alternatives Considered

| Approach | Effort | Security | Maintenance |
|----------|--------|----------|-------------|
| TLS pinning | 4.5h | HIGH | HIGH (certs rotate) |
| DNSSEC + HSTS preload | 0h | MED | LOW (passive) |
| Provider-side signature | 0h | MED | LOW (provider dependent) |
| mTLS (mutual TLS) | 2h | HIGH | LOW (if provider supports) |
| Output validation (L1 sanitize) | 1h ✅ DONE | HIGH (covers same vector) | LOW |

## Verdict: ✅ DONE by L1 Sanitize

The `core/sanitize.py` pipeline already covers **the same attack surface** at a higher layer:

- **TLS pinning** blocks the MITM at network layer
- **L1 Sanitize** detects the injection in the response text

Since any realistic MITM attack would inject `[SYSTEM]` / override commands into responses (the attacker wants the agent to DO something), the sanitize pipeline catches it regardless of transport.

The remaining edge case — an attacker who injects only subtle semantic payloads that bypass L1 — would also bypass pinning, because pinning only checks the TLS handshake, not the content.

## Recommendation

**Do not implement.** The effort-to-coverage ratio is poor:
- 4.5h of work
- Covers only 1 of 5 attack vectors (transport MITM)
- Already covered by L1 Sanitize with better coverage
- Ongoing cert rotation maintenance

---

## References

- [Let's Encrypt: Certificate pinning](https://letsencrypt.org/docs/certificate-pinning/)
- [OWASP: Certificate Pinning Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Pinning_Cheat_Sheet.html)
- `core/sanitize.py` — L1 Sanitize pipeline (existing)
- `hermes-agent-security-hardening` skill — full attack surface map
