# Media proxy origin boundary

The private `_validate_media_proxy_url` validates the entire URL against the
fixed HTTP(S) CDN authority before returning it. Its anchored `re.fullmatch`
accepts the supported CDN roots and subdomains, optional trailing dots, valid
ports and media paths. It rejects userinfo, authority suffix tricks, malformed
ports, backslashes and literal control characters that URL parsers can discard.

The earlier return-value model pack was removed. Explicit extension resolution
confirmed its row was loaded, and a forced query evaluation still reported the
real local validator flow. Retaining that ineffective model would misstate
acceptance. The executable origin guard now carries the boundary directly;
no model marks generic URL-safety helpers or HTTP clients as safe.

DNS and connection safety remain implementation contracts: the request hook
guards every redirect, the canonical transport re-resolves and pins the vetted
IP while preserving Host/SNI, and `trust_env=False` excludes unguarded proxy
mounts. The configured private-URL/fake-IP policy remains explicit; the origin
guard does not claim those opt-outs are disabled or replace transport checks.

`tests/hermes_cli/test_web_server.py` exercises the real router, HTTPX client,
guarded transport and policy with offline DNS/wire fixtures, including public
and private answers, mixed answers, redirects, rebinding and environment
proxies. Validator cases include domain-prefix attacks, schemes and credential
authorities, case, trailing dots, ports and parser normalization. These tests
must remain green when the origin boundary changes.

Local CodeQL acceptance uses the actual router and transport source plus a
separate unguarded FastAPI/HTTPX route. With CLI 2.27.1 and Python queries 1.8.11,
the full-SSRF query detects the unguarded route and no longer reports the media
proxy, without a model pack. Hosted acceptance still requires a scan of the
published head. This comparison does not certify all source paths or dismiss
unrelated code-scanning alerts.

References:

- [Python SSRF customization and barrier kind](https://github.com/github/codeql/blob/main/python/ql/lib/semmle/python/security/dataflow/ServerSideRequestForgeryCustomizations.qll)
