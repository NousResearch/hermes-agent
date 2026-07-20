"""``hermes model`` subcommand parser.

Interactive picker (TTY) plus Model Desk subcommands (M01–Q05).
"""

from __future__ import annotations

from typing import Callable


def build_model_parser(subparsers, *, cmd_model: Callable) -> None:
    """Attach the ``model`` subcommand to ``subparsers``."""
    model_parser = subparsers.add_parser(
        "model",
        help="Select default model/provider or run Model Desk ops",
        description=(
            "Interactively select inference provider/model, or run non-interactive "
            "Model Desk commands (doctor, preflight, dry-run, ports, …)."
        ),
    )
    model_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Wipe the model picker disk cache and re-fetch every provider's live /v1/models list.",
    )
    model_parser.add_argument(
        "--portal-url",
        help="Portal base URL for Nous login (default: production portal)",
    )
    model_parser.add_argument(
        "--inference-url",
        help="Inference API base URL for Nous login (default: production inference API)",
    )
    model_parser.add_argument(
        "--client-id",
        default=None,
        help="OAuth client id to use for Nous login (default: hermes-cli)",
    )
    model_parser.add_argument(
        "--scope", default=None, help="OAuth scope to request for Nous login"
    )
    model_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not attempt to open the browser automatically during Nous login",
    )
    model_parser.add_argument(
        "--manual-paste",
        action="store_true",
        help=(
            "For loopback OAuth providers: skip the local callback listener and paste "
            "the failed callback URL from your browser instead."
        ),
    )
    model_parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP request timeout in seconds for Nous login (default: 15)",
    )
    model_parser.add_argument(
        "--ca-bundle", help="Path to CA bundle PEM file for Nous TLS verification"
    )
    model_parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for Nous login (testing only)",
    )

    sub = model_parser.add_subparsers(dest="model_command")

    def _desk_dispatch(args):
        """Route nested ``hermes model <sub>`` commands to the Model Desk CLI."""
        from hermes_cli.model_desk_cli import run_model_desk_command

        raise SystemExit(run_model_desk_command(args))

    d = sub.add_parser("desk", help="Aggregated Model Desk status (M01)")
    d.add_argument("--deep", action="store_true")
    d.set_defaults(func=_desk_dispatch)
    s = sub.add_parser("status", help="Alias for desk")
    s.set_defaults(func=_desk_dispatch)
    doc = sub.add_parser("doctor", help="Model desk + golden + soft preflight")
    doc.set_defaults(func=_desk_dispatch)
    sub.add_parser("features", help="Import-check M01–Q05 modules")

    pf = sub.add_parser("preflight", help="Preflight hard-gate (M03)")
    pf.add_argument("--soft", dest="hard", action="store_false", default=True)
    pf.set_defaults(func=_desk_dispatch)

    dr = sub.add_parser("dry-run", help="Dry-run resolve — no API call (C09)")
    dr.add_argument("--provider", default="")
    dr.set_defaults(func=_desk_dispatch)
    ex = sub.add_parser("explain", help="Why-this-provider explain (Q04)")
    ex.add_argument("--provider", default="")
    ex.set_defaults(func=_desk_dispatch)

    p = sub.add_parser("ports", help="Local port collision detector (M04)")
    p.set_defaults(func=_desk_dispatch)

    sc = sub.add_parser("sync-ctx", help="Sync n_ctx → context_length (M05)")
    sc.add_argument("--host", default="127.0.0.1")
    sc.add_argument("--port", type=int, default=8080)
    sc.add_argument("--apply", action="store_true")
    sc.set_defaults(func=_desk_dispatch)

    nt = sub.add_parser("normalize-tag", help="Ollama tag normalizer (M06)")
    nt.add_argument("model_id", nargs="?", default="")
    nt.set_defaults(func=_desk_dispatch)

    sub.add_parser("privacy", help="Aux never-leave-local status (M07)")
    fa = sub.add_parser("fallback-audit", help="Fallback audit summary (M08)")
    fa.add_argument("--limit", type=int, default=50)
    fa.set_defaults(func=_desk_dispatch)

    aw = sub.add_parser("atomic-write", help="Atomic provider+model+url write (M09)")
    aw.add_argument("--provider", required=True)
    aw.add_argument("--model-id", dest="model_id", required=True)
    aw.add_argument("--base-url", dest="base_url", default=None)
    aw.add_argument("--api-mode", dest="api_mode", default=None)
    aw.add_argument("--apply", action="store_true")
    aw.set_defaults(func=_desk_dispatch)

    am = sub.add_parser("api-mode", help="api_mode self-test (M10)")
    am.add_argument("--live", action="store_true")
    am.set_defaults(func=_desk_dispatch)

    un = sub.add_parser("unify", help="Unify providers ↔ custom_providers (M02)")
    un.add_argument("--apply", action="store_true")
    un.set_defaults(func=_desk_dispatch)

    sub.add_parser("serve-plan", help="Local serve plan (L01)")
    sub.add_parser("resources", help="RAM budget gate (L05)")
    qn = sub.add_parser("quant", help="Quantization recommender (L10)")
    qn.add_argument("--params-b", dest="params_b", type=float, default=7.0)
    qn.set_defaults(func=_desk_dispatch)

    ol = sub.add_parser("ollama-list", help="List Ollama tags (L04)")
    ol.add_argument("--base-url", dest="base_url", default="http://127.0.0.1:11434")
    ol.set_defaults(func=_desk_dispatch)
    lm = sub.add_parser("lmstudio", help="LM Studio catalog probe (L03)")
    lm.add_argument("--base-url", dest="base_url", default=None)
    lm.set_defaults(func=_desk_dispatch)

    sub.add_parser("smart-route", help="Smart route score (C01)")
    sub.add_parser("capabilities", help="Vision/audio capability matrix (C07)")
    pr = sub.add_parser("parity", help="Streaming/tool parity matrix (C06)")
    pr.add_argument("--provider", default=None)
    pr.set_defaults(func=_desk_dispatch)
    sub.add_parser("aux-pins", help="Auxiliary task pins (C02)")
    po = sub.add_parser("pool", help="Credential pool status (C03)")
    po.add_argument("--provider", default=None)
    po.set_defaults(func=_desk_dispatch)

    sub.add_parser("golden", help="Golden resolve scenarios (Q03)")
    sub.add_parser("chaos", help="Chaos local-down drill (Q02)")
    sub.add_parser("slo", help="Provider resolve SLO (Q01)")

    sub.add_parser("slots", help="Local n_parallel / slots (L02)")
    hs = sub.add_parser("hot-swap", help="Hot-swap serve advice (L06)")
    hs.add_argument("--model-path", dest="model_path", default=None)
    hs.set_defaults(func=_desk_dispatch)
    sub.add_parser("embeddings", help="Embeddings route status (L07)")
    sd = sub.add_parser("spec-decode", help="Speculative decode advice (L08)")
    sd.add_argument("--model", dest="model_id", default=None)
    sd.set_defaults(func=_desk_dispatch)
    so = sub.add_parser("structured", help="Structured-output capability (L09)")
    so.add_argument("--provider", default=None)
    so.set_defaults(func=_desk_dispatch)
    dep = sub.add_parser("deprecation", help="Model deprecation warnings (C05)")
    dep.add_argument("--model", dest="model_id", default=None)
    dep.set_defaults(func=_desk_dispatch)
    rh = sub.add_parser("regional", help="Regional endpoint hints (C04)")
    rh.add_argument("--provider", default=None)
    rh.set_defaults(func=_desk_dispatch)
    sub.add_parser("spend", help="Spend/token ledger snapshot (C08)")
    tp = sub.add_parser("toolset-policy", help="Toolset↔model policy (C10)")
    tp.add_argument("toolset", nargs="?", default="")
    tp.add_argument("--provider", default=None)
    tp.set_defaults(func=_desk_dispatch)
    rd = sub.add_parser("redact", help="Redact secrets from JSON stdin/args (Q05)")
    rd.add_argument("--json", dest="json_payload", default=None)
    rd.set_defaults(func=_desk_dispatch)

    # Bare ``hermes model`` (no nested subcommand) routes to the interactive picker.
    model_parser.set_defaults(func=cmd_model)
