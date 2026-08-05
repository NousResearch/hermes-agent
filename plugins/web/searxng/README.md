# SearXNG provider configuration

Hermes calls `GET <SEARXNG_URL>/search` with `q`, `format=json`, and `pageno=1`.
The SearXNG instance must explicitly enable JSON responses:

```yaml
search:
  formats:
    - html
    - json
```

If SearXNG bot detection is enabled, allow the trusted API client in
`limiter.toml`, or configure the reverse proxy to set a trustworthy client IP.
Do not add forwarding headers in Hermes: a client-supplied `X-Forwarded-For` or
`X-Real-IP` can spoof trust and does not fix a server-side proxy policy.

For a local-only Docker deployment, bind the host listener to loopback and mount
the two configuration files directly. Do not also mount a volume over their
parent directory, because `/etc/searxng` can mask the file mounts and silently
restore SearXNG's HTML-only defaults:

```yaml
ports:
  - "127.0.0.1:8082:8080"
volumes:
  - ./settings.yml:/etc/searxng/settings.yml:ro
  - ./limiter.toml:/etc/searxng/limiter.toml:ro
```

After changing SearXNG configuration, recreate the container and verify the
same endpoint Hermes uses:

```bash
docker compose up -d --force-recreate
curl -fsS 'http://127.0.0.1:8082/search?q=hermes&format=json' \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["results"]))'
```

An HTTP 403 with an HTML response while ordinary HTML search succeeds usually
means the effective container configuration does not enable JSON. A 403 behind
a reverse proxy can instead indicate bot-detection or trusted-proxy rejection.
Hermes reports the status, method, query-free endpoint, server header, and
content type so these cases can be distinguished without logging search terms.