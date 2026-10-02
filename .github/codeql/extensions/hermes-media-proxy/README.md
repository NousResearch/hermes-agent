# Media proxy origin model

This model describes the private `_validate_media_proxy_url` return value: an
HTTP(S) URL whose hostname matches the fixed CDN allowlist and whose authority
contains no credentials. It applies only to that returned value. Passing the
original request URL to another HTTP client remains unmodeled.

The Python library maps the `request-forgery` barrier kind to
`FullUrlControlSanitizer`, so the model describes a restricted origin, not the
safety of every URL component. It does not suppress the partial SSRF query.

DNS and connection safety remain implementation contracts: the request hook
guards every redirect, the canonical transport re-resolves and pins the vetted
IP while preserving Host/SNI, and `trust_env=False` excludes unguarded proxy
mounts. The configured private-URL/fake-IP policy remains explicit; the model
does not claim those opt-outs are disabled or model generic URL safety as safe.

`tests/hermes_cli/test_web_server.py` exercises the real router, HTTPX client,
guarded transport and policy with offline DNS/wire fixtures, including public
and private answers, mixed answers, redirects, rebinding and environment
proxies. Validator cases include domain-prefix attacks, schemes and credential
authorities. These tests must remain green when this model changes.

Default setup discovers repository model packs from this directory. Acceptance
requires a current-head hosted scan; creating this pack alone is not evidence
that an alert closed.

References:

- [Repository model packs in default setup](https://docs.github.com/en/code-security/how-tos/find-and-fix-code-vulnerabilities/manage-your-configuration/edit-default-setup)
- [Python data extensions](https://codeql.github.com/docs/codeql-language-guides/customizing-library-models-for-python/)
- [Python SSRF customization and barrier kind](https://github.com/github/codeql/blob/main/python/ql/lib/semmle/python/security/dataflow/ServerSideRequestForgeryCustomizations.qll)
