# Local classic-CLI route presentation

Local-only UI change; no upstream PR, deployment, or live configuration change.
It covers the classic terminal `/model` picker and its switch confirmation, not
`hermes model` setup, Ink/desktop pickers, the startup banner, or status bar.

## Optional provider settings

A provider may opt into read-only route metadata and declare stable labels:

```yaml
custom_providers:
  - name: Turbofit
    base_url: https://provider.example/v1 # use your existing endpoint unchanged
    discover_models: false              # fixed native route catalogue
    picker_metadata: true
    models:
      auto: {picker_label: 'Turbofit:Auto'}
      active:main: {picker_label: 'Turbofit:Main'}
      active:aux: {picker_label: 'Turbofit:Aux'}
```

The equivalent keyed `providers` representation is supported by normalization.
Keep existing authentication, model context, defaults and endpoint settings. This
example is not an installer or a request to modify any live profile.

`picker_label` is presentation only. The provider stays one provider. Selection,
filter-to-selection mapping, API requests, session state and global configuration
continue to use original provider/model IDs. Unloaded/unknown rows remain selectable.

## Generic wire contract

The configured base URL's existing `/models` endpoint may include optional metadata:

```json
{"data":[{"id":"active:aux","metadata":{"role":"aux","backing_model":"exact-main-alias","residency":"ready","mode":"shared-main","observed_at":1700000000,"freshness":{"age_s":2,"max_age_s":15,"stale":false}}}]}
```

The richer TF8 contract supplies `role`, `backing_model` (string or null),
`residency` (`ready`, `loading`, `idle`, `error`, or `unknown`), `mode`
(`local` or `shared-main`), a Unix `observed_at`, and `freshness` as above.
Aux shared-main uses the actual main alias as its backing model. API policy labels
are not physical identity. Hermes does not infer identity/residency from a role,
port, health response or `/status`.

The consumer retains `unavailable` compatibility and an optional `mode: shared-main`
description. Dots always accompany words, not color alone. Old metadata without
the freshness contract preserves backing identity but shows unknown residency.

## Freshness and failure

Each `/model` open starts one background GET per explicitly opted-in matching
provider. A response is limited to 64 KiB and a one-second socket timeout; redirects
are refused to avoid forwarding credentials. No inference, lifecycle, wake, repair
or route-selection API is invoked. Ordinary providers add no metadata requests.

The picker initially shows stable labels with `Unknown model · ? Unknown`. It
invalidates when its worker completes. Residency expires using producer `age_s`
plus monotonic time since receipt, bounded by both producer `max_age_s` and the
UI's 15-second cap. `stale: true` or malformed/missing freshness immediately yields
unknown residency; the published backing identity and shared-main mode remain.
The producer timestamp is validated, not subtracted from the client's wall clock.
A new GET never resets producer age. There is no automatic polling: close/reopen
`/model` to fetch the latest producer observation. Each opening owns its
rows, so an old worker cannot overwrite a newer picker. This is an in-memory
observation, never a persisted runtime cache. Timeout/unsupported/malformed data
leaves unknown; it must never manufacture intentional idle or ready.

This initial consumer uses inline endpoint keys and configured extra headers;
key-command and custom TLS configuration are not implemented for the metadata
fetch. Such endpoints should leave metadata disabled or expect unknown, rather
than disable verification. Existing inference authentication is unchanged.

## Validation

`tests/hermes_cli/test_picker_route_presentation.py` drives the real config loader,
provider inventory, renderer, filtered selection and disk persistence in a
sandbox. A real ephemeral HTTP catalogue verifies GET-only metadata reads,
shared-main identity, reopen freshness and old-row isolation. Adjacent picker and
provider-switch tests cover ordinary providers.

The closeout evidence additionally records actual prompt_toolkit output through a
PTY at 40 and 60 columns. Synthetic catalogue metadata is labelled as such: it is
UI/HTTP-contract evidence, not a live deployed TF8 or GPU-residency claim.
